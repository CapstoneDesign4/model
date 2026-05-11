"""YAMNet 로드 + 더미/파일 오디오로 추론하여 환경을 검증하는 스크립트.

사용법:
    python scripts/verify_inference.py              # 더미 오디오
    python scripts/verify_inference.py --file data/sample/test.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 프로젝트 루트를 path에 추가 (scripts/ 위치에서 직접 실행 시)
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


# whitelist.yaml에 명시된 12종 위험 클래스의 YAMNet 521-차원 인덱스 모음.
# 검증 스크립트에서 점수 표시 대상이며, glass(435) + shatter(437)은 통합 클래스에 해당.
DANGER_INDICES = [11, 20, 302, 304, 390, 391, 393, 394, 420, 421, 435, 437, 464]


def main() -> None:
    parser = argparse.ArgumentParser(description="YAMNet 추론 환경 검증")
    parser.add_argument("--file", default=None, help="WAV 파일 경로 (없으면 더미 오디오 사용)")
    args = parser.parse_args()

    # 1. TF-Hub에서 YAMNet 로드
    print("[1/4] YAMNet 로딩 중...")
    import tensorflow_hub as hub
    import tensorflow as tf

    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    print("      로딩 완료.")

    # 2. 입력 준비
    print("[2/4] 입력 준비...")
    if args.file:
        # 실제 WAV 파일 첫 0.96s만 잘라 추론. 더 짧으면 zero-pad.
        import librosa
        audio, _ = librosa.load(args.file, sr=16000, mono=True)
        waveform = audio[:15360].astype(np.float32)
        if len(waveform) < 15360:
            waveform = np.pad(waveform, (0, 15360 - len(waveform)))
        print(f"      파일 입력: {args.file}  shape={waveform.shape}")
    else:
        # 파일이 없으면 무음(zeros)으로도 출력 shape이 정상인지 확인할 수 있다.
        waveform = np.zeros(15360, dtype=np.float32)
        print(f"      더미 입력 (zeros): shape={waveform.shape}")

    # 3. 추론
    print("[3/4] 추론 실행...")
    scores, embeddings, spectrogram = yamnet(tf.cast(waveform, tf.float32))
    scores_np = scores.numpy()
    embeddings_np = embeddings.numpy()

    print(f"      scores shape    : {scores_np.shape}      (기대: (N, 521))")
    print(f"      embeddings shape: {embeddings_np.shape}  (기대: (N, 1024))")

    assert scores_np.ndim == 2 and scores_np.shape[1] == 521, (
        f"scores shape 오류: {scores_np.shape}"
    )
    assert embeddings_np.ndim == 2 and embeddings_np.shape[1] == 1024, (
        f"embeddings shape 오류: {embeddings_np.shape}"
    )

    # 4. 위험 클래스 점수 출력
    print("[4/4] 위험 클래스 평균 score:")
    avg_scores = scores_np.mean(axis=0)  # shape (521,)

    # YAMNet class_map은 모델 내부 에셋에서 가져옴 (yamnet_class_map.csv).
    # 인덱스 ↔ 사람이 읽을 수 있는 라벨 매핑을 만들어 출력에 사용한다.
    try:
        class_map_path = yamnet.class_map_path().numpy().decode("utf-8")
        import csv
        class_names: dict[int, str] = {}
        with open(class_map_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_names[int(row["index"])] = row["display_name"]
    except Exception:
        # class_map 로드 실패 시에도 점수 표시는 가능하도록 빈 dict로 폴백.
        class_names = {}

    for idx in DANGER_INDICES:
        name = class_names.get(idx, f"index_{idx}")
        print(f"      [{idx:3d}] {name:<40s} score={avg_scores[idx]:.4f}")

    print()
    print("PASS: YAMNet inference OK")


if __name__ == "__main__":
    main()
