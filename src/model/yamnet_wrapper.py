"""TF-Hub YAMNet 로드 및 추론 래퍼."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 15360  # 0.96s * 16000


class YAMNetWrapper:
    """TF-Hub YAMNet을 로드하고 0.96s 윈도우 단위 추론을 수행한다."""

    def __init__(self, model_url: str = YAMNET_URL) -> None:
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
        waveform = tf.cast(waveform_16k_mono, tf.float32)
        scores, embeddings, spectrogram = self._model(waveform)
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
        return scores.mean(axis=0)  # shape (521,)
