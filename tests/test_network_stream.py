"""Network UDP packet and per-device stream tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.audio_io.network_stream import (
    CHUNK_SAMPLES,
    HEADER_SIZE,
    PCM_PAYLOAD_BYTES,
    DeviceStreamState,
    PacketFormatError,
    build_packet,
    parse_packet,
    pcm16_to_float32,
)


def _pcm_chunk(value: int = 0) -> np.ndarray:
    return np.full(CHUNK_SAMPLES, value, dtype="<i2")


class TestPacketFormat:
    def test_build_and_parse_packet(self):
        pcm = _pcm_chunk(1234)
        data = build_packet(device_id=1, seq=7, timestamp_ms=480, pcm16=pcm)

        assert len(data) == HEADER_SIZE + PCM_PAYLOAD_BYTES
        decoded = parse_packet(data)
        assert decoded.header.device_id == 1
        assert decoded.header.seq == 7
        assert decoded.header.timestamp_ms == 480
        assert decoded.header.payload_len == PCM_PAYLOAD_BYTES
        assert decoded.audio.shape == (CHUNK_SAMPLES,)
        assert decoded.audio.dtype == np.float32
        assert decoded.audio[0] == pytest.approx(1234 / 32768.0)

    def test_bad_magic_raises(self):
        data = bytearray(build_packet(device_id=1, seq=0, timestamp_ms=0, pcm16=_pcm_chunk()))
        data[0] = 0x00
        data[1] = 0x00
        with pytest.raises(PacketFormatError):
            parse_packet(bytes(data))

    def test_payload_len_mismatch_raises(self):
        data = build_packet(device_id=1, seq=0, timestamp_ms=0, pcm16=_pcm_chunk())
        with pytest.raises(PacketFormatError):
            parse_packet(data[:-2])

    def test_unexpected_payload_size_raises(self):
        short_pcm = np.zeros(CHUNK_SAMPLES - 1, dtype="<i2")
        data = build_packet(device_id=1, seq=0, timestamp_ms=0, pcm16=short_pcm)
        with pytest.raises(PacketFormatError):
            parse_packet(data)

    def test_pcm16_conversion_range(self):
        payload = np.array([-32768, 0, 32767], dtype="<i2").tobytes()
        audio = pcm16_to_float32(payload)
        assert audio.tolist() == pytest.approx([-1.0, 0.0, 32767 / 32768.0])


class TestDeviceStreamState:
    def test_two_packets_emit_one_window(self):
        state = DeviceStreamState(device_id=1)
        packet0 = parse_packet(build_packet(1, 0, 0, _pcm_chunk(100)))
        packet1 = parse_packet(build_packet(1, 1, 480, _pcm_chunk(200)))

        assert state.push_packet(packet0, received_at=1000.0) == []
        frames = state.push_packet(packet1, received_at=1000.48)

        assert len(frames) == 1
        frame = frames[0]
        assert frame.device_id == 1
        assert frame.seq == 1
        assert frame.frame.shape == (CHUNK_SAMPLES * 2,)
        assert frame.frame[:CHUNK_SAMPLES].mean() == pytest.approx(100 / 32768.0)
        assert frame.frame[CHUNK_SAMPLES:].mean() == pytest.approx(200 / 32768.0)

    def test_three_packets_emit_overlapping_windows(self):
        state = DeviceStreamState(device_id=1)
        packets = [
            parse_packet(build_packet(1, 0, 0, _pcm_chunk(100))),
            parse_packet(build_packet(1, 1, 480, _pcm_chunk(200))),
            parse_packet(build_packet(1, 2, 960, _pcm_chunk(300))),
        ]

        emitted = []
        for packet in packets:
            emitted.extend(state.push_packet(packet, received_at=1.0))

        assert len(emitted) == 2
        np.testing.assert_array_equal(
            emitted[0].frame[CHUNK_SAMPLES:],
            emitted[1].frame[:CHUNK_SAMPLES],
        )

    def test_duplicate_packet_is_dropped(self):
        state = DeviceStreamState(device_id=1)
        packet0 = parse_packet(build_packet(1, 0, 0, _pcm_chunk(100)))
        duplicate = parse_packet(build_packet(1, 0, 0, _pcm_chunk(999)))

        state.push_packet(packet0, received_at=1.0)
        frames = state.push_packet(duplicate, received_at=1.1)

        assert frames == []
        assert state.duplicate_packets == 1
        assert state.packets_received == 1

    def test_gap_fills_silence_and_counts_loss(self):
        state = DeviceStreamState(device_id=1)
        packet0 = parse_packet(build_packet(1, 0, 0, _pcm_chunk(100)))
        packet2 = parse_packet(build_packet(1, 2, 960, _pcm_chunk(300)))

        state.push_packet(packet0, received_at=1.0)
        frames = state.push_packet(packet2, received_at=2.0)

        assert state.packets_lost == 1
        assert len(frames) == 2
        assert frames[0].frame[:CHUNK_SAMPLES].mean() == pytest.approx(100 / 32768.0)
        assert frames[0].frame[CHUNK_SAMPLES:].mean() == pytest.approx(0.0)
        assert frames[1].frame[:CHUNK_SAMPLES].mean() == pytest.approx(0.0)
        assert frames[1].frame[CHUNK_SAMPLES:].mean() == pytest.approx(300 / 32768.0)

    def test_out_of_order_packet_is_dropped(self):
        state = DeviceStreamState(device_id=1)
        packet1 = parse_packet(build_packet(1, 1, 480, _pcm_chunk(100)))
        packet0 = parse_packet(build_packet(1, 0, 0, _pcm_chunk(200)))

        state.push_packet(packet1, received_at=1.0)
        frames = state.push_packet(packet0, received_at=1.1)

        assert frames == []
        assert state.out_of_order_packets == 1

    def test_device_mismatch_raises(self):
        state = DeviceStreamState(device_id=1)
        packet = parse_packet(build_packet(2, 0, 0, _pcm_chunk(100)))
        with pytest.raises(ValueError):
            state.push_packet(packet, received_at=1.0)
