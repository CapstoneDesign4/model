"""Send a WAV file as ESP32-style UDP PCM packets.

Example:
    python scripts/send_pcm_udp.py --file data/sample/siren/1-31482-A-42.wav \
        --device-id 1 --host 192.168.0.10 --port 5005
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_io.file_reader import load_and_resample  # noqa: E402
from src.audio_io.network_stream import (  # noqa: E402
    CHUNK_SAMPLES,
    SAMPLE_RATE,
    build_packet,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send WAV audio as ESP32 UDP PCM chunks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--file", required=True, help="WAV/audio file path")
    parser.add_argument("--host", required=True, help="Jetson receiver IP/host")
    parser.add_argument("--port", type=int, default=5005, help="Jetson UDP port")
    parser.add_argument("--device-id", type=int, required=True, choices=[1, 2])
    parser.add_argument("--seq-start", type=int, default=0, help="Initial uint32 seq")
    parser.add_argument(
        "--timestamp-start-ms",
        type=int,
        default=0,
        help="Initial ESP32 timestamp in milliseconds",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Repeat the file until interrupted",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Send packets as fast as possible instead of sleeping 0.48s per chunk",
    )
    return parser.parse_args()


def _float_to_pcm16(audio: np.ndarray) -> np.ndarray:
    """Convert float audio in [-1, 1] to signed int16."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2")


def _iter_pcm_chunks(audio: np.ndarray):
    """Yield fixed 0.48s int16 PCM chunks, zero-padding the final chunk."""
    pcm = _float_to_pcm16(audio)
    start = 0
    total = len(pcm)
    while start < total:
        chunk = pcm[start : start + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk))).astype("<i2")
        yield chunk
        start += CHUNK_SAMPLES


def main() -> int:
    args = _parse_args()
    path = Path(args.file)
    if not path.exists():
        print(f"[ERROR] file not found: {path}", file=sys.stderr)
        return 1

    audio = load_and_resample(path, sr=SAMPLE_RATE)
    chunks = list(_iter_pcm_chunks(audio))
    if not chunks:
        print(f"[ERROR] no audio samples in {path}", file=sys.stderr)
        return 1

    seq = args.seq_start & 0xFFFFFFFF
    timestamp_ms = args.timestamp_start_ms & 0xFFFFFFFF
    sleep_sec = CHUNK_SAMPLES / SAMPLE_RATE

    print(
        f"[INFO] sending {len(chunks)} chunks from {path} "
        f"as device_id={args.device_id} to {args.host}:{args.port}",
        file=sys.stderr,
    )

    sent = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            while True:
                for chunk in chunks:
                    packet = build_packet(
                        device_id=args.device_id,
                        seq=seq,
                        timestamp_ms=timestamp_ms,
                        pcm16=chunk,
                    )
                    sock.sendto(packet, (args.host, args.port))
                    sent += 1
                    seq = (seq + 1) & 0xFFFFFFFF
                    timestamp_ms = (timestamp_ms + int(sleep_sec * 1000)) & 0xFFFFFFFF
                    if not args.no_realtime:
                        time.sleep(sleep_sec)

                if not args.loop:
                    break
        except KeyboardInterrupt:
            print("\n[INFO] interrupted", file=sys.stderr)

    print(f"[INFO] sent packets: {sent}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

