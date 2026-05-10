"""실시간 위험 소리 감지 CLI 진입점 — `python -m src.cli` 형태로 실행."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# YAMNetWrapper는 tensorflow 의존성이 있어 --help 시에도 import가 시도된다.
# argparse 파싱까지는 heavy import를 지연시켜 --help가 항상 동작하도록 한다.
# (DangerFilter, Trigger는 numpy/yaml만 필요하므로 최상단 import 유지)
from src.model.danger_filter import DangerFilter
from src.postprocess.trigger import Trigger, TriggerEvent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YAMNet 기반 위험 소리 감지기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="FILE_OR_MIC",
        help="WAV 파일 경로 또는 'mic'",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="전체 클래스 공통 임계값 오버라이드",
    )
    parser.add_argument(
        "--config",
        default="config/whitelist.yaml",
        help="whitelist YAML 설정 파일 경로",
    )
    parser.add_argument(
        "--hop",
        type=float,
        default=0.48,
        help="hop 길이(초) - 실시간 지연 조정",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="매 윈도우마다 12종 전체 score 및 debounce 큐 상태 출력",
    )
    parser.add_argument(
        "--log",
        metavar="JSONL_PATH",
        default=None,
        help="결과를 JSONL 파일로 저장할 경로",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="마이크 장치 인덱스 (mic 모드 한정)",
    )
    # M2: debounce 파라미터
    parser.add_argument(
        "--debounce-window",
        type=int,
        default=None,
        metavar="N",
        help="슬라이딩 윈도우 크기 N (기본값: whitelist.yaml 또는 3)",
    )
    parser.add_argument(
        "--debounce-k",
        type=int,
        default=None,
        metavar="K",
        help="트리거 최소 양성 투표 수 K, 1 <= K <= N (기본값: whitelist.yaml 또는 2)",
    )
    parser.add_argument(
        "--no-debounce",
        action="store_true",
        help="debounce 비활성화 - 단일 윈도우 트리거 (M1 동작과 동일). 비교 디버깅용",
    )
    return parser.parse_args()


def _validate_debounce_args(window: int, k: int) -> None:
    """debounce 파라미터 유효성을 검사한다. 위반 시 오류 메시지를 출력하고 종료한다."""
    if window < 1 or k < 1:
        print(
            f"[ERROR] debounce 파라미터는 1 이상이어야 합니다. "
            f"(window={window}, k={k})",
            file=sys.stderr,
        )
        sys.exit(1)
    if k > window:
        print(
            f"[ERROR] --debounce-k ({k}) 은 --debounce-window ({window}) 이하여야 합니다.",
            file=sys.stderr,
        )
        sys.exit(1)


def _fmt_ts(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _print_danger(event: TriggerEvent) -> None:
    print(f"[{_fmt_ts(event.timestamp)}] DANGER: {event.key} (score={event.score:.4f})")


def _print_no_danger(timestamp: float, scores: dict[str, float]) -> None:
    if not scores:
        return
    top_key = max(scores, key=lambda k: scores[k])
    top_val = scores[top_key]
    print(f"[{_fmt_ts(timestamp)}] -- no danger (top: {top_key}={top_val:.4f})")


def _print_verbose(
    timestamp: float,
    scores: dict[str, float],
    trigger: Trigger,
    debounce_enabled: bool,
) -> None:
    """매 윈도우마다 클래스별 score 및 debounce 큐 상태를 출력한다."""
    ts_str = _fmt_ts(timestamp)
    print(f"[{ts_str}] WINDOW scores:")

    for key, score in scores.items():
        state = trigger.get_debounce_state(key)
        votes = state.snapshot()
        vote_sum = sum(votes)
        n = state.N
        k = state.K

        if debounce_enabled:
            passed = vote_sum >= k
            cooldown_active = state.is_cooldown_active(timestamp, 5.0)
            if passed and cooldown_active:
                status = "COOLDOWN"
            elif passed:
                status = "PASS"
            else:
                status = "--"
            votes_str = "[" + ",".join(str(v) for v in votes) + "]"
            print(
                f"  {key:<22}: {score:.4f}  votes={votes_str}  sum={vote_sum}/{n}  {status}"
            )
        else:
            # no-debounce 모드는 단순 score만 표시
            print(f"  {key:<22}: {score:.4f}")


def _make_log_record(
    timestamp: float,
    scores: dict[str, float],
    events: list[TriggerEvent],
    trigger: Trigger,
) -> dict:
    return {
        "timestamp": timestamp,
        "window_duration_ms": 960,
        "scores": scores,
        "triggered": [e.key for e in events],
        "top_score": max((e.score for e in events), default=None),
        # M2: 매 윈도우 레코드에 debounce_votes 항상 포함
        "debounce_votes": trigger.get_debounce_snapshot(),
    }


def run_file_mode(
    path: str,
    yamnet: YAMNetWrapper,
    danger_filter: DangerFilter,
    trigger: Trigger,
    hop_sec: float,
    verbose: bool,
    debounce_enabled: bool,
    log_file: Optional[object],
) -> None:
    from src.audio_io.file_reader import iter_file_frames

    print(f"[INFO] 파일 분석 시작: {path}", file=sys.stderr)
    wall_start = time.time()

    for ts_sec, frame in iter_file_frames(path, hop_sec=hop_sec):
        mean_scores = yamnet.infer_mean_scores(frame)
        scores = danger_filter.extract(mean_scores)
        now = wall_start + ts_sec
        events = trigger.evaluate(scores, now=now)

        if verbose:
            _print_verbose(now, scores, trigger, debounce_enabled)

        if events:
            for ev in events:
                _print_danger(ev)
        else:
            if not verbose:
                _print_no_danger(now, scores)

        if log_file is not None:
            record = _make_log_record(now, scores, events, trigger)
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()

    print("[INFO] 파일 분석 완료.", file=sys.stderr)


def run_mic_mode(
    yamnet: YAMNetWrapper,
    danger_filter: DangerFilter,
    trigger: Trigger,
    hop_sec: float,
    verbose: bool,
    debounce_enabled: bool,
    log_file: Optional[object],
    device: Optional[int],
) -> None:
    from src.audio_io.mic_stream import MicStream

    hop_samples = int(hop_sec * 16000)
    mic = MicStream(device=device, hop_samples=hop_samples)
    mic.start()
    print("[INFO] 마이크 스트림 시작. Ctrl+C로 종료.", file=sys.stderr)

    try:
        for _elapsed, frame in mic.iter_frames():
            now = time.time()
            mean_scores = yamnet.infer_mean_scores(frame)
            scores = danger_filter.extract(mean_scores)
            events = trigger.evaluate(scores, now=now)

            if verbose:
                _print_verbose(now, scores, trigger, debounce_enabled)

            if events:
                for ev in events:
                    _print_danger(ev)
            else:
                if not verbose:
                    _print_no_danger(now, scores)

            if log_file is not None:
                record = _make_log_record(now, scores, events, trigger)
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()

    except KeyboardInterrupt:
        print("\n[INFO] 마이크 스트림 종료.", file=sys.stderr)
    finally:
        mic.stop()


def main() -> None:
    args = _parse_args()

    # 화이트리스트 로드 + 임계값 오버라이드
    danger_filter = DangerFilter(config_path=args.config)
    if args.threshold != 0.5:
        danger_filter.override_threshold(args.threshold)

    # debounce 파라미터: CLI > YAML > 기본값 순서로 적용
    debounce_window = args.debounce_window or danger_filter.debounce_config.window
    debounce_k = args.debounce_k or danger_filter.debounce_config.k
    debounce_enabled = not args.no_debounce

    if not args.no_debounce:
        _validate_debounce_args(debounce_window, debounce_k)

    trigger = Trigger(
        danger_filter,
        debounce_window=debounce_window,
        debounce_k=debounce_k,
        debounce_enabled=debounce_enabled,
    )

    # YAMNet은 실제 추론이 필요한 시점에만 import (--help는 여기에 도달하지 않음)
    from src.model.yamnet_wrapper import YAMNetWrapper  # noqa: PLC0415

    # YAMNet 로드
    print("[INFO] YAMNet 로딩 중 (최초 실행 시 다운로드 발생)...", file=sys.stderr)
    yamnet = YAMNetWrapper()
    print("[INFO] YAMNet 로딩 완료.", file=sys.stderr)

    log_file = None
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")

    try:
        if args.input.lower() == "mic":
            run_mic_mode(
                yamnet, danger_filter, trigger,
                args.hop, args.verbose, debounce_enabled, log_file, args.device,
            )
        else:
            run_file_mode(
                args.input, yamnet, danger_filter, trigger,
                args.hop, args.verbose, debounce_enabled, log_file,
            )
    finally:
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    main()
