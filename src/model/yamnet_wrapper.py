"""TF-Hub YAMNet 로드 및 추론 래퍼."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# TF-Hub에 공개된 YAMNet v1 모델 URL. 최초 호출 시 ~수십 MB 캐시 다운로드 발생.
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 15360  # 0.96s * 16000 (YAMNet의 표준 입력 길이)


class YAMNetWrapper:
    """TF-Hub YAMNet을 로드하고 0.96s 윈도우 단위 추론을 수행한다."""

    def __init__(self, model_url: str = YAMNET_URL) -> None:
        # hub.load는 SavedModel을 캐시(TFHUB_CACHE_DIR)에 저장 후 callable 핸들을 반환.
        self._model = hub.load(model_url)

    def infer(
        self, waveform_16k_mono: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """0.96s float32 mono 파형을 입력받아 (scores, embeddings, spectrogram)을 반환한다.

        Args:
            waveform_16k_mono: shape (N,), dtype float32, 값 범위 [-1.0, 1.0].
                               N은 보통 15360 (0.96s @ 16kHz).

        Returns:
            scores:      shape (num_patches, 521)  — 패치별 521 클래스 확률
            embeddings:  shape (num_patches, 1024) — 패치별 임베딩 (M3 이후 사용)
            spectrogram: shape (num_patches, 64)   — mel spectrogram (참고용)
        """
        # YAMNet은 float32 1D 텐서를 기대한다. 명시적 캐스팅으로 dtype 호환성 확보.
        waveform = tf.cast(waveform_16k_mono, tf.float32)
        # 모델 호출 → (scores, embeddings, spectrogram) 3-튜플 반환.
        scores, embeddings, spectrogram = self._model(waveform)
        # TF Tensor → numpy 배열로 변환하여 호출 측에서 다루기 쉽게 한다.
        return (
            scores.numpy(),
            embeddings.numpy(),
            spectrogram.numpy(),
        )

    def infer_mean_scores(self, waveform_16k_mono: np.ndarray) -> np.ndarray:
        """추론 후 패치별 scores를 평균내어 shape (521,) 벡터로 반환한다.

        M1 단순 버전: num_patches 차원을 axis=0으로 평균.
        """
        scores, _, _ = self.infer(waveform_16k_mono)
        # YAMNet 내부적으로 0.96s 입력은 다수의 패치(0.48s씩)로 쪼개져 추론된다.
        # M1에서는 단순 평균으로 521차원 단일 벡터를 만든다.
        return scores.mean(axis=0)  # shape (521,)
