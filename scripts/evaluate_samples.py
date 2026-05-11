"""실제 WAV 샘플 평가 스크립트 — 임계값 sweep 기반 precision/recall/F1/false alarm.

사용 예:
    python scripts/evaluate_samples.py --data-dir data/sample
    python scripts/evaluate_samples.py --data-dir data/sample \
        --manifest data/sample/manifest.csv \
        --thresholds 0.1,0.2,0.3,0.4,0.5 \
        --out-dir experiments/eval_custom

매니페스트 CSV 형식 (헤더 필수):
    filename,label
    scream_01.wav,screaming
    bgm_01.wav,negative

라벨 규칙:
- 매니페스트가 있으면 우선.
- 없으면 하위 폴더명을 라벨로 추정 (예: data/sample/screaming/foo.wav → "screaming").
- 폴더가 평탄(파일만 있음)하면 파일명 토큰 매칭 fallback: 위험 클래스 key가 파일명에 포함되면 그 라벨로, 그 외는 'negative'.
- 라벨이 12종 위험 클래스 key가 아니면 'negative'로 취급 (즉, 비위험).

평가 결과:
- predictions.jsonl: 파일별 12종 score / max score / 라벨 / per-threshold trigger 여부
- metrics.csv: threshold별 precision/recall/F1/false alarm rate (전체 위험 vs 비위험 기준)
- summary.md: 사람이 읽기 좋은 요약 (클래스별 score 분포 + 임계값 sweep 표)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 프로젝트 루트를 sys.path에 추가 (scripts/ 직접 실행 대비)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_THRESHOLDS = [round(0.1 * i, 2) for i in range(1, 10)]  # 0.1 ~ 0.9


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YAMNet 위험 소리 감지 파이프라인 WAV 평가",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="data/sample",
                   help="WAV 파일 검색 루트 디렉터리 (재귀)")
    p.add_argument("--manifest", default=None,
                   help="filename,label CSV. 없으면 폴더명/파일명에서 라벨 추론.")
    p.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
                   help="콤마 구분 임계값 리스트")
    p.add_argument("--out-dir", default=None,
                   help="결과 저장 디렉터리 (기본: experiments/eval_YYYYMMDD_HHMMSS)")
    p.add_argument("--hop", type=float, default=0.48,
                   help="hop 길이(초)")
    p.add_argument("--config", default="config/whitelist.yaml",
                   help="화이트리스트 YAML")
    p.add_argument("--per-class-threshold", action="store_true",
                   help="sweep 대신 whitelist.yaml의 클래스별 threshold로 평가")
    return p.parse_args()


def _find_wavs(data_dir: Path) -> List[Path]:
    """data_dir 하위에서 .wav 파일을 재귀 탐색."""
    if not data_dir.exists():
        return []
    return sorted([p for p in data_dir.rglob("*.wav") if p.is_file()])


def _load_manifest(path: Path) -> Dict[str, str]:
    """filename → label 매핑."""
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("filename", "").strip()
            label = row.get("label", "").strip()
            if fname:
                mapping[fname] = label
    return mapping


def _infer_label(
    wav_path: Path,
    data_dir: Path,
    manifest: Dict[str, str],
    danger_keys: List[str],
) -> str:
    """매니페스트 → 폴더명 → 파일명 토큰 매칭 순으로 라벨 추론."""
    if wav_path.name in manifest:
        return manifest[wav_path.name]
    rel = wav_path.relative_to(data_dir)
    # 하위 폴더가 있으면 첫 번째 폴더명이 라벨
    if len(rel.parts) > 1:
        return rel.parts[0]
    # 파일명 토큰 매칭
    stem_lower = wav_path.stem.lower()
    for key in danger_keys:
        # key의 일부(언더스코어 분리)도 매칭에 사용
        tokens = [key] + key.split("_")
        for tok in tokens:
            if tok and tok in stem_lower:
                return key
    return "negative"


def _is_positive_label(label: str, danger_keys_set: set) -> bool:
    """위험 클래스 key 중 하나면 positive."""
    return label in danger_keys_set


def _compute_metrics(
    predictions: List[dict],
    thresholds: List[float],
    danger_keys_set: set,
) -> List[dict]:
    """임계값별 precision/recall/F1/false alarm rate 계산.

    파일 단위(any-class) 평가:
      - 양성: label이 위험 클래스 key
      - 예측 양성: 어떤 위험 클래스라도 score >= threshold
    """
    metrics = []
    for thr in thresholds:
        tp = fp = fn = tn = 0
        for rec in predictions:
            true_pos = _is_positive_label(rec["label"], danger_keys_set)
            pred_pos = rec["max_score"] >= thr
            if true_pos and pred_pos:
                tp += 1
            elif (not true_pos) and pred_pos:
                fp += 1
            elif true_pos and (not pred_pos):
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics.append({
            "threshold": thr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_alarm_rate": round(far, 4),
        })
    return metrics


def _compute_per_class_metrics(
    predictions: List[dict],
    class_thresholds: Dict[str, float],
    danger_keys_set: set,
) -> dict:
    """클래스별 threshold 적용 단일 평가. 어떤 위험 클래스라도 자기 threshold 통과 시 trigger."""
    tp = fp = fn = tn = 0
    for rec in predictions:
        true_pos = _is_positive_label(rec["label"], danger_keys_set)
        pred_pos = any(
            rec["scores"].get(k, 0.0) >= thr for k, thr in class_thresholds.items()
        )
        if true_pos and pred_pos:
            tp += 1
        elif (not true_pos) and pred_pos:
            fp += 1
        elif true_pos and (not pred_pos):
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "mode": "per_class",
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alarm_rate": round(far, 4),
        "thresholds": class_thresholds,
    }


def _write_metrics_csv(path: Path, metrics: List[dict]) -> None:
    fields = ["threshold", "tp", "fp", "fn", "tn",
              "precision", "recall", "f1", "false_alarm_rate"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(metrics)


def _write_predictions_jsonl(path: Path, predictions: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _class_score_distribution(predictions: List[dict]) -> Dict[str, Dict[str, float]]:
    """라벨 그룹별 클래스 score 통계 (mean/max)."""
    grouped: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for rec in predictions:
        grouped[rec["label"]].append(rec["scores"])

    dist: Dict[str, Dict[str, float]] = {}
    for label, score_dicts in grouped.items():
        agg: Dict[str, List[float]] = defaultdict(list)
        for sd in score_dicts:
            for k, v in sd.items():
                agg[k].append(v)
        dist[label] = {
            k: {
                "n": len(vs),
                "mean": round(sum(vs) / len(vs), 4) if vs else 0.0,
                "max": round(max(vs), 4) if vs else 0.0,
            }
            for k, vs in agg.items()
        }
    return dist


def _write_summary_md(
    path: Path,
    out_dir: Path,
    data_dir: Path,
    n_files: int,
    n_positive: int,
    n_negative: int,
    metrics: List[dict],
    score_dist: Dict[str, Dict[str, float]],
    thresholds: List[float],
    label_counts: Dict[str, int],
) -> None:
    lines = []
    lines.append(f"# 평가 요약 — {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"- data_dir: `{data_dir}`")
    lines.append(f"- 총 파일 수: {n_files}")
    lines.append(f"- 위험(positive): {n_positive}, 비위험(negative): {n_negative}")
    lines.append(f"- 임계값 sweep: {thresholds}")
    lines.append("")
    lines.append("## 라벨 분포")
    for lbl, n in sorted(label_counts.items()):
        lines.append(f"- {lbl}: {n}")
    lines.append("")
    lines.append("## 임계값별 지표")
    lines.append("")
    lines.append("| threshold | TP | FP | FN | TN | precision | recall | F1 | FAR |")
    lines.append("|-----------|----|----|----|----|-----------|--------|----|-----|")
    for m in metrics:
        lines.append(
            f"| {m['threshold']:.2f} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['false_alarm_rate']:.4f} |"
        )
    lines.append("")
    lines.append("## 라벨 그룹별 클래스 score (mean / max)")
    for label, per_class in score_dist.items():
        lines.append("")
        lines.append(f"### label = `{label}` (n={list(per_class.values())[0]['n'] if per_class else 0})")
        lines.append("")
        lines.append("| class | mean | max |")
        lines.append("|-------|------|-----|")
        for cls_key, stats in per_class.items():
            lines.append(f"| {cls_key} | {stats['mean']:.4f} | {stats['max']:.4f} |")
    lines.append("")
    lines.append("산출 파일:")
    lines.append(f"- predictions.jsonl")
    lines.append(f"- metrics.csv")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()

    data_dir = Path(args.data_dir).resolve()
    wavs = _find_wavs(data_dir)

    # out_dir 결정
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "experiments" / f"eval_{stamp}"

    if not wavs:
        print(
            f"[안내] '{data_dir}' 하위에서 WAV 파일을 찾지 못했습니다.\n"
            f"      위험 소리 데이터셋을 아래에 배치하세요:\n"
            f"        - ESC-50:       https://github.com/karolpiczak/ESC-50  (glass_breaking, siren, car_horn, ...)\n"
            f"        - FSD50K:       https://zenodo.org/record/4060432       (Screaming, Gunshot, Smoke_detector, ...)\n"
            f"        - UrbanSound8K: https://urbansounddataset.weebly.com/   (siren, car_horn, ...)\n"
            f"      예) data/sample/screaming/scream_01.wav (폴더명이 라벨이 됩니다)",
            file=sys.stderr,
        )
        return 0

    # 무거운 import 지연
    from src.audio_io.file_reader import iter_file_frames
    from src.model.danger_filter import DangerFilter
    try:
        from src.model.yamnet_wrapper import YAMNetWrapper
    except ImportError as e:
        print(f"[ERROR] YAMNet import 실패 (tensorflow/tensorflow_hub 미설치?): {e}",
              file=sys.stderr)
        return 1

    # 임계값 파싱
    try:
        thresholds = sorted({float(t) for t in args.thresholds.split(",") if t.strip()})
    except ValueError as e:
        print(f"[ERROR] --thresholds 파싱 실패: {e}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    danger_filter = DangerFilter(config_path=args.config)
    danger_keys = [e.key for e in danger_filter.classes]
    danger_keys_set = set(danger_keys)

    manifest: Dict[str, str] = {}
    if args.manifest:
        manifest_path = Path(args.manifest)
        if manifest_path.exists():
            manifest = _load_manifest(manifest_path)
        else:
            print(f"[WARN] manifest 파일 없음: {manifest_path}", file=sys.stderr)

    print(f"[INFO] YAMNet 로딩 중...", file=sys.stderr)
    try:
        yamnet = YAMNetWrapper()
    except Exception as e:
        print(f"[ERROR] YAMNet 로딩 실패: {e}", file=sys.stderr)
        return 1
    print(f"[INFO] YAMNet 로딩 완료. 평가 파일 수: {len(wavs)}", file=sys.stderr)

    predictions: List[dict] = []
    label_counts: Dict[str, int] = defaultdict(int)

    for wav in wavs:
        label = _infer_label(wav, data_dir, manifest, danger_keys)
        label_counts[label] += 1
        # 파일 전체 윈도우 평균이 아닌, 윈도우별 max를 파일 단위 score로 사용 (위험 이벤트는 짧음)
        per_window_scores: List[Dict[str, float]] = []
        try:
            for ts_sec, frame in iter_file_frames(wav, hop_sec=args.hop):
                mean_scores = yamnet.infer_mean_scores(frame)
                scores = danger_filter.extract(mean_scores)
                per_window_scores.append({k: float(v) for k, v in scores.items()})
        except Exception as e:
            print(f"[WARN] {wav.name} 처리 실패: {e}", file=sys.stderr)
            continue

        if not per_window_scores:
            continue

        # 파일 단위 score: 각 클래스에 대해 윈도우 최댓값
        file_scores: Dict[str, float] = {}
        for key in danger_keys:
            file_scores[key] = max(ws[key] for ws in per_window_scores)
        max_key = max(file_scores, key=lambda k: file_scores[k])
        max_score = file_scores[max_key]

        # threshold별 trigger 여부 (어떤 위험 클래스라도 임계값 통과하면 trigger)
        per_threshold: Dict[str, dict] = {}
        for thr in thresholds:
            triggered_classes = [k for k, v in file_scores.items() if v >= thr]
            per_threshold[f"{thr:.2f}"] = {
                "triggered": bool(triggered_classes),
                "triggered_classes": triggered_classes,
            }

        rec = {
            "filename": str(wav.relative_to(data_dir)),
            "label": label,
            "n_windows": len(per_window_scores),
            "scores": file_scores,
            "max_class": max_key,
            "max_score": round(max_score, 4),
            "per_threshold": per_threshold,
        }
        predictions.append(rec)
        print(
            f"[{label:>22}] {wav.name:<40} max={max_key}:{max_score:.4f} "
            f"(n_windows={len(per_window_scores)})",
            file=sys.stderr,
        )

    if not predictions:
        print("[ERROR] 유효한 예측이 없습니다.", file=sys.stderr)
        return 1

    # 메트릭 계산
    metrics = _compute_metrics(predictions, thresholds, danger_keys_set)
    score_dist = _class_score_distribution(predictions)

    per_class_metrics: Optional[dict] = None
    if args.per_class_threshold:
        class_thresholds = {e.key: float(e.threshold) for e in danger_filter.classes}
        per_class_metrics = _compute_per_class_metrics(
            predictions, class_thresholds, danger_keys_set
        )
        with (out_dir / "per_class_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(per_class_metrics, f, indent=2, ensure_ascii=False)

    # 산출물 저장
    _write_predictions_jsonl(out_dir / "predictions.jsonl", predictions)
    _write_metrics_csv(out_dir / "metrics.csv", metrics)

    n_positive = sum(1 for p in predictions if _is_positive_label(p["label"], danger_keys_set))
    n_negative = len(predictions) - n_positive
    _write_summary_md(
        out_dir / "summary.md",
        out_dir, data_dir,
        n_files=len(predictions),
        n_positive=n_positive, n_negative=n_negative,
        metrics=metrics, score_dist=score_dist,
        thresholds=thresholds, label_counts=dict(label_counts),
    )

    # 콘솔에 표 요약
    print("\n=== Threshold sweep ===", file=sys.stderr)
    print("thr   TP  FP  FN  TN   P      R      F1     FAR", file=sys.stderr)
    for m in metrics:
        print(
            f"{m['threshold']:.2f}  {m['tp']:>3} {m['fp']:>3} {m['fn']:>3} {m['tn']:>3}  "
            f"{m['precision']:.3f}  {m['recall']:.3f}  {m['f1']:.3f}  {m['false_alarm_rate']:.3f}",
            file=sys.stderr,
        )
    if per_class_metrics is not None:
        m = per_class_metrics
        print("\n=== Per-class threshold (whitelist.yaml) ===", file=sys.stderr)
        print(
            f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}  "
            f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  FAR={m['false_alarm_rate']:.3f}",
            file=sys.stderr,
        )
        print(f"thresholds = {m['thresholds']}", file=sys.stderr)

    print(f"\n[INFO] 결과 저장: {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
