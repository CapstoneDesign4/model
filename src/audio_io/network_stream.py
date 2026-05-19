"""UDP PCM network stream input for ESP32 audio nodes."""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass
from typing import Dict, Generator, Optional, Tuple

import numpy as np

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 15360
CHUNK_SAMPLES = 7680
PCM_PAYLOAD_BYTES = CHUNK_SAMPLES * 2

PACKET_MAGIC = 0xA501
PACKET_VERSION = 1
# Header fields use network byte order. PCM payload stays signed little-endian int16.
HEADER_FORMAT = "!HBBIIHH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAX_UINT32 = 0xFFFFFFFF
SEQ_OLDER_THAN_HALF_RANGE = 0x80000000


class PacketFormatError(ValueError):
    """Raised when an incoming UDP packet does not match the expected format."""


@dataclass(frozen=True)
class PacketHeader:
    """Fixed-size ESP32 audio packet header."""

    magic: int
    version: int
    device_id: int
    seq: int
    timestamp_ms: int
    payload_len: int
    flags: int = 0


@dataclass(frozen=True)
class DecodedPacket:
    """Decoded UDP packet with float32 PCM payload."""

    header: PacketHeader
    audio: np.ndarray


@dataclass(frozen=True)
class DeviceStats:
    """Immutable snapshot of per-device network stream counters."""

    packets_received: int
    packets_lost: int
    duplicate_packets: int
    out_of_order_packets: int
    frames_emitted: int
    last_seq: Optional[int]


@dataclass(frozen=True)
class NetworkFrame:
    """One 0.96s analysis frame assembled from network PCM chunks."""

    device_id: int
    timestamp: float
    frame: np.ndarray
    seq: int
    esp_timestamp_ms: int
    stats: DeviceStats


def pack_header(
    device_id: int,
    seq: int,
    timestamp_ms: int,
    payload_len: int = PCM_PAYLOAD_BYTES,
    flags: int = 0,
) -> bytes:
    """Pack the fixed 16-byte UDP header."""
    if not 0 <= seq <= MAX_UINT32:
        raise ValueError(f"seq must be uint32, got {seq}")
    if not 0 <= timestamp_ms <= MAX_UINT32:
        raise ValueError(f"timestamp_ms must be uint32, got {timestamp_ms}")
    if not 0 <= payload_len <= 0xFFFF:
        raise ValueError(f"payload_len must be uint16, got {payload_len}")
    if not 0 <= flags <= 0xFFFF:
        raise ValueError(f"flags must be uint16, got {flags}")
    if not 0 <= device_id <= 0xFF:
        raise ValueError(f"device_id must be uint8, got {device_id}")

    return struct.pack(
        HEADER_FORMAT,
        PACKET_MAGIC,
        PACKET_VERSION,
        device_id,
        seq,
        timestamp_ms,
        payload_len,
        flags,
    )


def unpack_header(data: bytes) -> PacketHeader:
    """Unpack and validate the fixed UDP header."""
    if len(data) < HEADER_SIZE:
        raise PacketFormatError(
            f"packet too short: got {len(data)} bytes, need at least {HEADER_SIZE}"
        )
    raw = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
    header = PacketHeader(*raw)
    if header.magic != PACKET_MAGIC:
        raise PacketFormatError(f"bad magic: 0x{header.magic:04x}")
    if header.version != PACKET_VERSION:
        raise PacketFormatError(f"unsupported version: {header.version}")
    return header


def pcm16_to_float32(payload: bytes) -> np.ndarray:
    """Convert signed little-endian PCM16 bytes to float32 in [-1.0, 1.0]."""
    if len(payload) % 2 != 0:
        raise PacketFormatError(f"PCM16 payload length must be even, got {len(payload)}")
    pcm = np.frombuffer(payload, dtype="<i2")
    return (pcm.astype(np.float32) / 32768.0).astype(np.float32)


def parse_packet(data: bytes, *, strict_payload_len: bool = True) -> DecodedPacket:
    """Parse one UDP packet into header and float32 PCM payload."""
    header = unpack_header(data)
    payload = data[HEADER_SIZE:]
    if header.payload_len != len(payload):
        raise PacketFormatError(
            f"payload_len mismatch: header={header.payload_len}, actual={len(payload)}"
        )
    if strict_payload_len and header.payload_len != PCM_PAYLOAD_BYTES:
        raise PacketFormatError(
            f"unexpected payload_len: {header.payload_len}, expected {PCM_PAYLOAD_BYTES}"
        )
    audio = pcm16_to_float32(payload)
    if strict_payload_len and audio.shape != (CHUNK_SAMPLES,):
        raise PacketFormatError(
            f"unexpected sample count: {audio.shape[0]}, expected {CHUNK_SAMPLES}"
        )
    return DecodedPacket(header=header, audio=audio)


def build_packet(
    device_id: int,
    seq: int,
    timestamp_ms: int,
    pcm16: np.ndarray | bytes,
    flags: int = 0,
) -> bytes:
    """Build one UDP packet from int16 PCM samples or raw PCM bytes."""
    if isinstance(pcm16, bytes):
        payload = pcm16
    else:
        payload = np.asarray(pcm16, dtype="<i2").tobytes()
    header = pack_header(
        device_id=device_id,
        seq=seq & MAX_UINT32,
        timestamp_ms=timestamp_ms & MAX_UINT32,
        payload_len=len(payload),
        flags=flags,
    )
    return header + payload


class DeviceStreamState:
    """Per-device ring buffer and sequence accounting."""

    def __init__(
        self,
        device_id: int,
        window_samples: int = WINDOW_SAMPLES,
        hop_samples: int = CHUNK_SAMPLES,
        max_gap_fill_packets: int = 2,
    ) -> None:
        self.device_id = device_id
        self.window_samples = window_samples
        self.hop_samples = hop_samples
        self.max_gap_fill_packets = max_gap_fill_packets
        self.buffer = np.zeros(0, dtype=np.float32)
        self.last_seq: Optional[int] = None
        self.packets_received = 0
        self.packets_lost = 0
        self.duplicate_packets = 0
        self.out_of_order_packets = 0
        self.frames_emitted = 0

    def stats(self) -> DeviceStats:
        """Return a snapshot of stream counters."""
        return DeviceStats(
            packets_received=self.packets_received,
            packets_lost=self.packets_lost,
            duplicate_packets=self.duplicate_packets,
            out_of_order_packets=self.out_of_order_packets,
            frames_emitted=self.frames_emitted,
            last_seq=self.last_seq,
        )

    def _accept_seq(self, seq: int) -> Optional[int]:
        """Accept new/in-order/ahead seq values.

        Returns:
            Number of missing packets before this packet, or None when the packet
            is duplicate/out-of-order and should be dropped.
        """
        seq &= MAX_UINT32
        if self.last_seq is None:
            self.last_seq = seq
            return 0

        diff = (seq - self.last_seq) & MAX_UINT32
        if diff == 0:
            self.duplicate_packets += 1
            return None
        if diff >= SEQ_OLDER_THAN_HALF_RANGE:
            self.out_of_order_packets += 1
            return None
        missing = 0
        if diff > 1:
            missing = diff - 1
            self.packets_lost += missing
        self.last_seq = seq
        return missing

    def push_packet(self, packet: DecodedPacket, received_at: float) -> list[NetworkFrame]:
        """Append one decoded packet and return any completed 0.96s frames."""
        header = packet.header
        if header.device_id != self.device_id:
            raise ValueError(
                f"packet device_id={header.device_id} does not match state={self.device_id}"
            )
        missing = self._accept_seq(header.seq)
        if missing is None:
            return []

        self.packets_received += 1
        if missing:
            if missing <= self.max_gap_fill_packets:
                silence = np.zeros(missing * self.hop_samples, dtype=np.float32)
                self.buffer = np.concatenate([self.buffer, silence])
            else:
                # A large gap means the stream likely restarted; keep real-time behavior.
                self.buffer = np.zeros(0, dtype=np.float32)
        self.buffer = np.concatenate([self.buffer, packet.audio.astype(np.float32)])

        frames: list[NetworkFrame] = []
        while len(self.buffer) >= self.window_samples:
            frame = self.buffer[: self.window_samples].copy()
            self.buffer = self.buffer[self.hop_samples :]
            self.frames_emitted += 1
            frames.append(
                NetworkFrame(
                    device_id=self.device_id,
                    timestamp=received_at,
                    frame=frame,
                    seq=header.seq,
                    esp_timestamp_ms=header.timestamp_ms,
                    stats=self.stats(),
                )
            )
        return frames


class NetworkAudioStream:
    """UDP receiver that yields per-device 0.96s analysis frames."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5005,
        device_count: int = 2,
        timeout: float = 1.0,
        recv_buffer_size: int = 1 << 20,
    ) -> None:
        self.host = host
        self.port = port
        self.device_count = device_count
        self.timeout = timeout
        self.recv_buffer_size = recv_buffer_size
        self._socket: Optional[socket.socket] = None
        self._states: Dict[int, DeviceStreamState] = {}
        self.bad_packets = 0
        self.unknown_device_packets = 0

    def start(self) -> None:
        """Bind the UDP socket."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.recv_buffer_size)
        sock.settimeout(self.timeout)
        sock.bind((self.host, self.port))
        self._socket = sock

    def stop(self) -> None:
        """Close the UDP socket."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _state_for(self, device_id: int) -> DeviceStreamState:
        if device_id not in self._states:
            self._states[device_id] = DeviceStreamState(device_id=device_id)
        return self._states[device_id]

    def iter_frames(self) -> Generator[NetworkFrame, None, None]:
        """Yield assembled frames indefinitely until interrupted or stopped."""
        if self._socket is None:
            raise RuntimeError("NetworkAudioStream.start() must be called first")

        while self._socket is not None:
            try:
                data, _addr = self._socket.recvfrom(HEADER_SIZE + PCM_PAYLOAD_BYTES + 1024)
            except socket.timeout:
                continue
            except OSError:
                break

            received_at = time.time()
            try:
                packet = parse_packet(data)
            except PacketFormatError:
                self.bad_packets += 1
                continue

            if packet.header.device_id < 1 or packet.header.device_id > self.device_count:
                self.unknown_device_packets += 1
                continue

            state = self._state_for(packet.header.device_id)
            for frame in state.push_packet(packet, received_at=received_at):
                yield frame

    def stats_snapshot(self) -> dict:
        """Return receiver and per-device counters for logging/debugging."""
        return {
            "bad_packets": self.bad_packets,
            "unknown_device_packets": self.unknown_device_packets,
            "devices": {
                device_id: state.stats().__dict__
                for device_id, state in sorted(self._states.items())
            },
        }


def open_network_stream(
    host: str = "0.0.0.0",
    port: int = 5005,
    device_count: int = 2,
    timeout: float = 1.0,
) -> NetworkAudioStream:
    """Create and start a NetworkAudioStream."""
    stream = NetworkAudioStream(
        host=host,
        port=port,
        device_count=device_count,
        timeout=timeout,
    )
    stream.start()
    return stream
