"""WAV 파일을 로드하여 0.96s 슬라이딩 윈도우 프레임을 생성한다."""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Tuple

import librosa
import numpy as np

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 15360   # 0.96s
DEFAULT_HOP_SAMPLES = 7680  # 0.48s (50% 오버랩)


def load_and_resample(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """오디오 파일을 로드하고 16kHz mono float32 배열로 반환한다."""
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return audio.astype(np.float32)


def sliding_window_frames(
    audio: np.ndarray,
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = DEFAULT_HOP_SAMPLES,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """audio 배열을 슬라이딩 윈도우로 잘라 (frame_index, frame) 튜플을 yield한다.

    마지막 프레임이 window_samples보다 짧으면 제로패딩하여 반환한다.

    Args:
        audio:          float32 1D 배열 (16kHz mono).
        window_samples: 윈도우 길이 (기본 15360 = 0.96s).
        hop_samples:    hop 길이 (기본 7680 = 0.48s).

    Yields:
        (frame_index, frame_array):
            frame_index — 0-based 윈도우 번호
            frame_array — shape (window_samples,) float32
    """
    total = len(audio)
    frame_idx = 0
    start = 0

    while start < total:
        end = start + window_samples
        chunk = audio[start:end]

        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)))

        yield frame_idx, chunk.astype(np.float32)
        frame_idx += 1
        start += hop_samples


def iter_file_frames(
    path: str | Path,
    hop_sec: float = 0.48,
) -> Generator[Tuple[float, np.ndarray], None, None]:
    """파일 경로에서 직접 (timestamp_sec, frame) 제너레이터를 반환한다.

    Args:
        path:    WAV 파일 경로.
        hop_sec: hop 길이(초). CLI --hop 인자와 연결됨.

    Yields:
        (timestamp_sec, frame_array):
            timestamp_sec — 윈도우 시작 시각 (파일 기준, 초)
            frame_array   — shape (15360,) float32
    """
    audio = load_and_resample(path)
    hop_samples = int(hop_sec * SAMPLE_RATE)

    for frame_idx, frame in sliding_window_frames(audio, hop_samples=hop_samples):
        timestamp_sec = frame_idx * hop_sec
        yield timestamp_sec, frame
