"""실시간 위험 소리 감지 CLI 진입점 — `python -m src.cli` 형태로 실행."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

# YAMNetWrapper는 tensorflow 의존성이 있어 --help 시에도 import가 시도된다.
# argparse 파싱까지는 heavy import를 지연시켜 --help가 항상 동작하도록 한다.
# (DangerFilter, Trigger는 numpy/yaml만 필요하므로 최상단 import 유지)
from src.model.danger_filter import DangerFilter
from src.postprocess.trigger import Trigger, TriggerEvent


# ─────────────────────────────────────────────────────────────────────────────
# LiveDisplay: 마이크 모드 전용 in-place 터미널 갱신 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _ansi_supported() -> bool:
    """ANSI escape code 사용 가능 여부를 판정한다.

    조건:
      1. stdout 이 TTY 일 것.
      2. 환경변수 NO_COLOR 가 설정되지 않을 것.
    Windows Terminal / PowerShell 7+ 은 기본적으로 ANSI 를 지원한다.
    cmd.exe 구형 환경에서는 os.system("") 호출로 VT 처리 모드를 활성화한다.
    """
    if not sys.stdout.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    # Windows 에서 ANSI VT 처리 모드 활성화 (no-op on non-Windows)
    if sys.platform == "win32":
        os.system("")  # 빈 명령으로 ConEmu/WT VT 처리 초기화
    return True


class LiveDisplay:
    """마이크 모드에서 터미널 같은 자리를 in-place 로 갱신하는 디스플레이.

    TTY 가 아니거나 NO_COLOR 가 설정되면 fallback 으로 기존 스크롤 출력을 사용한다.

    non-verbose 모드:
        헤더 1줄 + 클래스 N줄 블록을 in-place 로 갱신한다. verbose 모드와 동일한
        블록 갱신 로직을 사용하며, 차이는 debounce 정보(votes, sum, 상태) 표시 여부뿐이다.
        DANGER 이벤트는 블록 위쪽에 스크롤 출력되도록 블록을 지우고 DANGER 를 출력한 뒤
        블록을 다시 인쇄한다.

    verbose 모드:
        non-verbose 와 동일한 블록 구조이지만 각 클래스 줄 끝에 debounce 큐 상태
        (votes=[...] sum=K/N PASS/COOLDOWN/--) 가 추가된다.
    """

    # ANSI escape 상수
    _ERASE_LINE = "\x1b[K"        # 커서 위치부터 줄 끝까지 지우기
    _ERASE_WHOLE_LINE = "\x1b[2K" # 줄 전체 지우기
    _MOVE_UP = "\x1b[{}A"         # N줄 위로 (format 사용)
    _CR = "\r"                    # 줄 처음으로

    def __init__(self, verbose: bool) -> None:
        self._verbose = verbose
        self._ansi = _ansi_supported()
        # 이전에 인쇄한 블록의 줄 수 (0이면 아직 미출력). non-verbose/verbose 공용.
        self._block_lines: int = 0

    # ------------------------------------------------------------------
    # 공개 메서드
    # ------------------------------------------------------------------

    def update_no_danger(
        self,
        timestamp: float,
        scores: dict[str, float],
        trigger: Trigger,
        debounce_enabled: bool,
    ) -> None:
        """DANGER 없는 윈도우 결과를 라이브 갱신한다.

        non-verbose: 클래스 점수만 표시 (debounce 정보 없음).
        verbose: update_verbose() 를 대신 호출하므로 여기서는 non-verbose 경로만 처리.
        """
        if not self._verbose:
            lines = self._build_block_lines(timestamp, scores, trigger, debounce_enabled=False)
            if self._ansi:
                self._redraw_block(lines)
            else:
                for line in lines:
                    print(line)

    def update_verbose(
        self,
        timestamp: float,
        scores: dict[str, float],
        trigger: Trigger,
        debounce_enabled: bool,
    ) -> None:
        """verbose 모드: 클래스별 score + debounce 상태 블록을 in-place 갱신한다."""
        lines = self._build_block_lines(timestamp, scores, trigger, debounce_enabled)
        if self._ansi:
            self._redraw_block(lines)
        else:
            # fallback: 그냥 줄 단위 출력
            for line in lines:
                print(line)

    def emit_danger(self, event: TriggerEvent) -> None:
        """DANGER 이벤트를 라이브 영역 위쪽(스크롤로 남는 줄)으로 출력한다."""
        danger_line = (
            f"[{_fmt_ts(event.timestamp)}] DANGER: {event.key} (score={event.score:.4f})"
        )
        if self._ansi:
            self._insert_above_block(danger_line)
        else:
            # fallback: 그냥 인쇄
            print(danger_line)

    def finalize(self) -> None:
        """Ctrl+C 종료 시 커서를 라이브 영역 아래로 내린다."""
        if self._ansi:
            if self._block_lines > 0:
                # 블록이 출력된 상태이면 블록 아래로 커서 이동
                sys.stdout.write("\n")
                sys.stdout.flush()

    # ------------------------------------------------------------------
    # 공용 블록 빌더 (non-verbose / verbose 통합)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_block_lines(
        timestamp: float,
        scores: dict[str, float],
        trigger: Trigger,
        debounce_enabled: bool,
    ) -> list[str]:
        """헤더 1줄 + 클래스 N줄 블록을 생성한다.

        debounce_enabled=True(verbose) 이면 각 클래스 줄 끝에 debounce 정보를 추가한다.
        debounce_enabled=False(non-verbose) 이면 점수만 표시한다.
        """
        lines: list[str] = []
        ts_str = _fmt_ts(timestamp)
        lines.append(f"[{ts_str}] WINDOW scores:")

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
                lines.append(
                    f"  {key:<22}: {score:.4f}  votes={votes_str}  sum={vote_sum}/{n}  {status}"
                )
            else:
                lines.append(f"  {key:<22}: {score:.4f}")

        return lines

    # ------------------------------------------------------------------
    # 블록 in-place 갱신 내부 메서드
    # ------------------------------------------------------------------

    def _redraw_block(self, lines: list[str]) -> None:
        """ANSI: 이전 블록 위치로 이동해 줄 단위로 덮어쓴다."""
        out = []
        if self._block_lines > 0:
            # 이전 블록 첫 줄로 이동
            out.append(self._MOVE_UP.format(self._block_lines))

        for line in lines:
            out.append(self._CR + self._ERASE_LINE + line + "\n")

        # 새 블록이 이전 블록보다 줄 수가 적을 때 남은 줄을 지운다 (일반적으로 없음)
        leftover = self._block_lines - len(lines)
        for _ in range(leftover):
            out.append(self._CR + self._ERASE_WHOLE_LINE + "\n")

        sys.stdout.write("".join(out))
        sys.stdout.flush()
        self._block_lines = len(lines)

    def _insert_above_block(self, danger_line: str) -> None:
        """ANSI: 블록을 일시적으로 지우고 DANGER 줄을 위에 인쇄한 뒤 블록을 다시 그린다.

        블록이 아직 미출력(_block_lines=0)이면 단순 인쇄.
        """
        if self._block_lines == 0:
            # 블록이 아직 인쇄되지 않은 상태이면 그냥 인쇄
            sys.stdout.write(danger_line + "\n")
            sys.stdout.flush()
            return

        out = []
        # 블록 첫 줄로 이동
        out.append(self._MOVE_UP.format(self._block_lines))
        # 블록 줄들을 모두 지운다 (위에서 아래로)
        for _ in range(self._block_lines):
            out.append(self._CR + self._ERASE_WHOLE_LINE + "\n")
        # 블록 아래에 와 있으므로 다시 첫 줄로 이동
        out.append(self._MOVE_UP.format(self._block_lines))

        # DANGER 줄 출력 (스크롤 히스토리로 확정)
        out.append(danger_line + "\n")
        sys.stdout.write("".join(out))
        sys.stdout.flush()

        # 블록 줄 수를 리셋해 다음 _redraw_block 이 처음부터 인쇄하게 한다
        self._block_lines = 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI 인자 파싱 및 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    # CLI 인자 파서 정의. ArgumentDefaultsHelpFormatter로 --help 시 기본값을 자동 표시한다.
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
    # Unix epoch 시각을 "YYYY-MM-DD HH:MM:SS.mmm" 문자열로 변환 (밀리초 3자리까지 표시).
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# ─────────────────────────────────────────────────────────────────────────────
# 파일 모드용 출력 함수 (변경 없음 — 스크롤 방식 유지)
# ─────────────────────────────────────────────────────────────────────────────

def _print_danger(event: TriggerEvent) -> None:
    # 위험 이벤트 한 건을 콘솔에 한 줄로 출력한다.
    print(f"[{_fmt_ts(event.timestamp)}] DANGER: {event.key} (score={event.score:.4f})")


def _print_no_danger(timestamp: float, scores: dict[str, float]) -> None:
    # 트리거가 없을 때 가장 점수가 높은 클래스를 표시하여 모니터링 가시성을 확보한다.
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

    # 12종 클래스 각각에 대해 score, debounce 큐 상태, 판정 결과를 출력한다.
    for key, score in scores.items():
        state = trigger.get_debounce_state(key)
        votes = state.snapshot()       # 현재 슬라이딩 윈도우 투표 이력
        vote_sum = sum(votes)          # 양성 투표 수 = sum
        n = state.N                    # 윈도우 크기 N
        k = state.K                    # 트리거 최소 양성 수 K

        if debounce_enabled:
            # 디바운스 통과/쿨다운 활성 여부를 별도로 판단해 상태 문자열을 결정.
            passed = vote_sum >= k
            cooldown_active = state.is_cooldown_active(timestamp, 5.0)
            if passed and cooldown_active:
                status = "COOLDOWN"   # K 충족했지만 쿨다운으로 억제됨
            elif passed:
                status = "PASS"       # 트리거 가능 상태
            else:
                status = "--"         # K 미충족
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


# ─────────────────────────────────────────────────────────────────────────────
# 파일 모드 (스크롤 출력 유지)
# ─────────────────────────────────────────────────────────────────────────────

def run_file_mode(
    path: str,
    yamnet,
    danger_filter: DangerFilter,
    trigger: Trigger,
    hop_sec: float,
    verbose: bool,
    debounce_enabled: bool,
    log_file: Optional[object],
) -> None:
    # 파일 모드는 무거운 audio_io import를 함수 진입 시점으로 지연한다.
    from src.audio_io.file_reader import iter_file_frames

    print(f"[INFO] 파일 분석 시작: {path}", file=sys.stderr)
    wall_start = time.time()  # 파일 내부 timestamp(0초 기준)를 벽시계로 변환할 때 사용

    # 0.96s 윈도우를 순차로 받아 YAMNet 추론 → 위험 클래스 필터 → 트리거 판정 파이프라인 실행.
    for ts_sec, frame in iter_file_frames(path, hop_sec=hop_sec):
        mean_scores = yamnet.infer_mean_scores(frame)   # (521,) 평균 score 벡터
        scores = danger_filter.extract(mean_scores)     # 12종 위험 클래스만 추출
        now = wall_start + ts_sec                       # 실시간 로그 시각 일관성 유지
        events = trigger.evaluate(scores, now=now)      # debounce/cooldown 적용

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


# ─────────────────────────────────────────────────────────────────────────────
# 마이크 모드 (라이브 디스플레이 적용)
# ─────────────────────────────────────────────────────────────────────────────

def run_mic_mode(
    yamnet,
    danger_filter: DangerFilter,
    trigger: Trigger,
    hop_sec: float,
    verbose: bool,
    debounce_enabled: bool,
    log_file: Optional[object],
    device: Optional[int],
) -> None:
    # 마이크 모드 진입 시점에 sounddevice import (--help 경로 보호).
    from src.audio_io.mic_stream import MicStream

    hop_samples = int(hop_sec * 16000)  # 초 단위 hop을 16kHz 샘플 수로 변환
    mic = MicStream(device=device, hop_samples=hop_samples)
    mic.start()  # 비동기 콜백 스레드로 PCM 캡처 시작
    print("[INFO] 마이크 스트림 시작. Ctrl+C로 종료.", file=sys.stderr)

    display = LiveDisplay(verbose=verbose)

    try:
        # 마이크는 무한 스트림이므로 KeyboardInterrupt(Ctrl+C)로만 종료된다.
        for _elapsed, frame in mic.iter_frames():
            now = time.time()  # 마이크는 실시간이므로 벽시계를 그대로 사용
            mean_scores = yamnet.infer_mean_scores(frame)
            scores = danger_filter.extract(mean_scores)
            events = trigger.evaluate(scores, now=now)

            # DANGER 이벤트: 라이브 영역 위쪽에 확정 출력 (스크롤로 남음)
            for ev in events:
                display.emit_danger(ev)

            # 라이브 영역 갱신
            if verbose:
                display.update_verbose(now, scores, trigger, debounce_enabled)
            else:
                display.update_no_danger(now, scores, trigger, debounce_enabled)

            # JSONL 로그는 기존과 동일하게 매 윈도우 기록
            if log_file is not None:
                record = _make_log_record(now, scores, events, trigger)
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()

    except KeyboardInterrupt:
        display.finalize()
        print("\n[INFO] 마이크 스트림 종료.", file=sys.stderr)
    finally:
        mic.stop()


def main() -> None:
    # 1) CLI 인자 파싱.
    args = _parse_args()

    # 2) 화이트리스트(YAML)에서 12종 위험 클래스 + 임계값 + cooldown 로드.
    danger_filter = DangerFilter(config_path=args.config)
    if args.threshold != 0.5:
        # 사용자가 명시적으로 --threshold를 준 경우만 전 클래스 일괄 오버라이드.
        danger_filter.override_threshold(args.threshold)

    # 3) debounce 파라미터: CLI > YAML > 기본값(3, 2) 순으로 우선 적용.
    debounce_window = args.debounce_window or danger_filter.debounce_config.window
    debounce_k = args.debounce_k or danger_filter.debounce_config.k
    debounce_enabled = not args.no_debounce

    if not args.no_debounce:
        # debounce 활성 시 K <= N 등 유효성 사전 검증 (불일치면 sys.exit).
        _validate_debounce_args(debounce_window, debounce_k)

    # 4) 트리거 인스턴스 생성: 클래스별 DebounceState 초기화 포함.
    trigger = Trigger(
        danger_filter,
        debounce_window=debounce_window,
        debounce_k=debounce_k,
        debounce_enabled=debounce_enabled,
    )

    # 5) YAMNet은 텐서플로 의존이 무겁기 때문에 --help 경로를 보호하기 위해 늦게 import.
    from src.model.yamnet_wrapper import YAMNetWrapper  # noqa: PLC0415

    # 6) TF-Hub에서 YAMNet 로드 (최초 1회 ~수십 MB 다운로드 발생).
    print("[INFO] YAMNet 로딩 중 (최초 실행 시 다운로드 발생)...", file=sys.stderr)
    yamnet = YAMNetWrapper()
    print("[INFO] YAMNet 로딩 완료.", file=sys.stderr)

    # 7) --log 옵션이 있으면 JSONL 파일을 append 모드로 열어둔다.
    log_file = None
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")

    # 8) --input 값에 따라 마이크/파일 분기. log_file은 finally에서 안전하게 닫는다.
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
