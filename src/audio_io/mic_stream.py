"""sounddevice 콜백 기반 마이크 스트림 — 0.96s 윈도우를 yield한다."""

from __future__ import annotations

import queue
import threading
from typing import Generator, Optional, Tuple

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 15360   # 0.96s
DEFAULT_HOP_SAMPLES = 7680  # 0.48s


class MicStream:
    """sounddevice 콜백으로 PCM을 수집하고 0.96s 윈도우를 생성한다.

    M1: 최소 동작 수준. 링 버퍼는 deque 기반 단순 구현.
    M2 이후: VAD(음성 활동 감지) 연동 및 overflow 처리 강화 예정.
    """

    def __init__(
        self,
        device: Optional[int] = None,
        sr: int = SAMPLE_RATE,
        hop_samples: int = DEFAULT_HOP_SAMPLES,
    ) -> None:
        self._sr = sr
        self._hop_samples = hop_samples
        self._window_samples = WINDOW_SAMPLES
        self._device = device
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._buffer = np.zeros(0, dtype=np.float32)
        self._stop_event = threading.Event()

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice 콜백 — 수신 PCM 청크를 큐에 넣는다."""
        if status:
            # 오버플로 등 상태 경고를 stderr에 출력하지만 계속 동작
            import sys
            print(f"[MicStream] sounddevice status: {status}", file=sys.stderr)
        self._q.put(indata[:, 0].copy().astype(np.float32))

    def start(self) -> None:
        """마이크 스트림을 시작한다."""
        self._stop_event.clear()
        self._stream = sd.InputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            blocksize=self._hop_samples,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """마이크 스트림을 중단한다."""
        self._stop_event.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def iter_frames(self) -> Generator[Tuple[float, np.ndarray], None, None]:
        """(elapsed_sec, frame) 튜플을 yield하는 제너레이터.

        start()를 먼저 호출해야 한다. KeyboardInterrupt 또는 stop() 호출 시 종료.

        Yields:
            (elapsed_sec, frame_array):
                elapsed_sec — 스트림 시작 기준 경과 시간(초, 근사값)
                frame_array — shape (15360,) float32
        """
        import time
        start_time = time.time()
        window_count = 0

        while not self._stop_event.is_set():
            try:
                chunk = self._q.get(timeout=1.0)
            except queue.Empty:
                continue

            self._buffer = np.concatenate([self._buffer, chunk])

            # 윈도우 길이 이상 누적됐으면 앞에서부터 hop만큼 슬라이딩
            while len(self._buffer) >= self._window_samples:
                frame = self._buffer[: self._window_samples].copy()
                self._buffer = self._buffer[self._hop_samples :]
                elapsed_sec = window_count * (self._hop_samples / self._sr)
                window_count += 1
                yield elapsed_sec, frame
