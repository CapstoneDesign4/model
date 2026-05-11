"""합성 신호 기반 audio_io 프레이밍 + danger_filter 매핑 단위 테스트.

네트워크/YAMNet 없이 실행 가능. 결정론적 random seed 사용.

검증 대상:
1. 합성 신호 헬퍼: 사인파, 화이트노이즈, 임펄스, 무음, chirp 생성 (16kHz mono float32).
2. audio_io.file_reader.sliding_window_frames 프레이밍:
   - 윈도우 길이 15360 (0.96s), hop 7680 (0.48s), dtype float32, shape 일치.
   - 마지막 프레임 zero-pad.
3. danger_filter.DangerFilter:
   - whitelist.yaml의 13개 YAMNet 인덱스 → 12개 이벤트 키 (glass_shatter 통합).
   - 모듈에서 config를 import 해 인덱스 단일 출처 확인 (재하드코딩 없음).
   - 클래스별 인덱스 핫원 주입 시 해당 클래스 점수가 max로 반영.

실행: pytest tests/test_synthetic_signals.py -v
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest

from src.audio_io.file_reader import (
    DEFAULT_HOP_SAMPLES,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    sliding_window_frames,
)
from src.model.danger_filter import DangerFilter


# ─────────────────────────────────────────────────────────────────────────────
# 합성 신호 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _sine(freq_hz: float, duration_sec: float, sr: int = SAMPLE_RATE,
          amplitude: float = 0.5) -> np.ndarray:
    """단일 주파수 사인파 (16kHz mono float32)."""
    n = int(duration_sec * sr)
    t = np.arange(n, dtype=np.float32) / sr
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def _white_noise(duration_sec: float, sr: int = SAMPLE_RATE,
                 amplitude: float = 0.1, seed: int = 0) -> np.ndarray:
    """결정론적 화이트노이즈."""
    rng = np.random.default_rng(seed)
    n = int(duration_sec * sr)
    return (amplitude * rng.standard_normal(n)).astype(np.float32)


def _impulse(duration_sec: float, sr: int = SAMPLE_RATE,
             position_ratio: float = 0.5, amplitude: float = 1.0) -> np.ndarray:
    """단일 임펄스 (deltafunction)."""
    n = int(duration_sec * sr)
    sig = np.zeros(n, dtype=np.float32)
    idx = int(n * position_ratio)
    if 0 <= idx < n:
        sig[idx] = amplitude
    return sig


def _silence(duration_sec: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """무음."""
    return np.zeros(int(duration_sec * sr), dtype=np.float32)


def _chirp(f0: float, f1: float, duration_sec: float,
           sr: int = SAMPLE_RATE, amplitude: float = 0.5) -> np.ndarray:
    """선형 처프 (f0 → f1)."""
    n = int(duration_sec * sr)
    t = np.arange(n, dtype=np.float32) / sr
    # phase = 2π ∫(f0 + (f1-f0)*t/T) dt = 2π(f0*t + (f1-f0)*t²/(2T))
    phase = 2.0 * np.pi * (f0 * t + (f1 - f0) * t * t / (2.0 * duration_sec))
    return (amplitude * np.sin(phase)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 합성 신호 헬퍼 자체 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticHelpers:
    """헬퍼가 올바른 shape/dtype/길이/값 범위를 내는지 확인."""

    def test_sine_shape_and_dtype(self):
        sig = _sine(1000.0, 1.0)
        assert sig.dtype == np.float32
        assert sig.shape == (SAMPLE_RATE,)
        # 진폭 0.5 사인파의 abs max는 0.5 근방
        assert sig.max() == pytest.approx(0.5, abs=1e-3)
        assert sig.min() == pytest.approx(-0.5, abs=1e-3)

    def test_sine_freq_via_zero_crossings(self):
        """1kHz 사인 1초에서 zero-crossing이 약 2000개 (양→음, 음→양)."""
        sig = _sine(1000.0, 1.0)
        # 부호가 바뀌는 위치 카운트
        signs = np.sign(sig)
        zc = int(np.sum(np.diff(signs) != 0))
        # 이론치 2000, 부동소수 오차 허용
        assert zc == pytest.approx(2000, abs=4)

    def test_white_noise_deterministic(self):
        """동일 seed에서 동일 값을 내야 한다."""
        a = _white_noise(0.5, seed=42)
        b = _white_noise(0.5, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_white_noise_different_seed(self):
        a = _white_noise(0.5, seed=1)
        b = _white_noise(0.5, seed=2)
        assert not np.array_equal(a, b)

    def test_impulse_single_nonzero(self):
        sig = _impulse(0.5)
        assert sig.dtype == np.float32
        nonzero = np.flatnonzero(sig)
        assert nonzero.size == 1
        assert sig[nonzero[0]] == pytest.approx(1.0)

    def test_silence_all_zero(self):
        sig = _silence(0.5)
        assert sig.dtype == np.float32
        assert sig.shape == (8000,)
        assert np.all(sig == 0.0)

    def test_chirp_shape(self):
        sig = _chirp(200.0, 4000.0, 1.0)
        assert sig.dtype == np.float32
        assert sig.shape == (SAMPLE_RATE,)
        # chirp는 시작/끝 모두 진폭 범위 내
        assert sig.max() <= 0.5 + 1e-3
        assert sig.min() >= -0.5 - 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# audio_io 프레이밍 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestSlidingWindowFraming:
    """sliding_window_frames의 윈도우 길이·hop·dtype·zero-pad 검증."""

    def test_window_and_hop_constants(self):
        """프로젝트 표준: 0.96s/15360, 0.48s/7680."""
        assert WINDOW_SAMPLES == 15360
        assert DEFAULT_HOP_SAMPLES == 7680
        assert SAMPLE_RATE == 16000

    def test_single_window_shape_dtype(self):
        """정확히 1개 윈도우 길이 입력 → 첫 프레임 shape (15360,) float32.

        구현은 `while start < total`이므로 start=0(15360s)과 start=7680(7680s+zero-pad)
        두 프레임을 yield한다. 첫 프레임만 shape/dtype을 검증.
        """
        sig = _sine(1000.0, 0.96)  # 정확히 15360 샘플
        assert sig.shape == (WINDOW_SAMPLES,)
        frames = list(sliding_window_frames(sig))
        assert len(frames) >= 1
        idx, frame = frames[0]
        assert idx == 0
        assert frame.shape == (WINDOW_SAMPLES,)
        assert frame.dtype == np.float32

    def test_two_full_windows_with_overlap(self):
        """1.44s(=15360+7680) 입력 → start=0,7680,15360 모두 < 23040 → 3 프레임."""
        sig = _sine(1000.0, 1.44)
        frames = list(sliding_window_frames(sig))
        assert len(frames) == 3
        for idx, frame in frames:
            assert frame.shape == (WINDOW_SAMPLES,)
            assert frame.dtype == np.float32

    def test_hop_overlap_content_matches(self):
        """첫 윈도우의 후반 7680 샘플 == 두 번째 윈도우의 전반 7680 샘플."""
        sig = _sine(500.0, 1.44)
        frames = list(sliding_window_frames(sig))
        f0 = frames[0][1]
        f1 = frames[1][1]
        np.testing.assert_array_equal(f0[DEFAULT_HOP_SAMPLES:], f1[:DEFAULT_HOP_SAMPLES])

    def test_last_frame_zero_padded(self):
        """짧은 입력(0.5s=8000샘플) → 첫 프레임은 원본+zero-pad, 두 번째는 전부 0."""
        sig = _sine(1000.0, 0.5)  # 8000 샘플
        frames = list(sliding_window_frames(sig))
        # start=0 (8000 < 15360, zero-pad), start=7680 (320샘플 < 15360, zero-pad) → 2 프레임
        assert len(frames) >= 1
        _, frame0 = frames[0]
        assert frame0.shape == (WINDOW_SAMPLES,)
        assert frame0.dtype == np.float32
        # 뒤쪽은 zero-padding
        assert np.all(frame0[8000:] == 0.0)
        # 앞쪽은 원본
        np.testing.assert_allclose(frame0[:8000], sig, atol=1e-7)

    def test_silence_input_all_zero_frames(self):
        """무음 입력 시 모든 프레임이 0이어야 한다."""
        sig = _silence(0.96)
        frames = list(sliding_window_frames(sig))
        assert len(frames) >= 1
        for _, frame in frames:
            assert frame.shape == (WINDOW_SAMPLES,)
            assert frame.dtype == np.float32
            assert np.all(frame == 0.0)

    def test_frame_indices_sequential(self):
        sig = _white_noise(2.0, seed=7)
        frames = list(sliding_window_frames(sig))
        indices = [idx for idx, _ in frames]
        assert indices == list(range(len(frames)))


# ─────────────────────────────────────────────────────────────────────────────
# danger_filter 매핑 검증 (13개 raw → 12개 이벤트)
# ─────────────────────────────────────────────────────────────────────────────

class TestDangerFilterMapping:
    """whitelist.yaml의 단일 출처 인덱스로부터 12종 이벤트 매핑 동작 검증.

    원칙: 테스트 안에서 인덱스를 재하드코딩하지 않고, DangerFilter 인스턴스에서
    실제 로드된 entry.indices만 사용해 검증.
    """

    @pytest.fixture
    def filter_(self):
        return DangerFilter()

    def test_twelve_event_keys(self, filter_):
        """이벤트 키는 12종이어야 한다 (glass+shatter 통합)."""
        assert len(filter_.classes) == 12
        # 키 집합은 CLAUDE.md 명세와 일치
        keys = {e.key for e in filter_.classes}
        expected = {
            "screaming", "baby_cry", "glass_shatter", "breaking",
            "gunshot", "explosion", "fire_alarm", "smoke_alarm",
            "siren", "civil_defense_siren", "car_alarm", "vehicle_horn",
        }
        assert keys == expected

    def test_thirteen_raw_indices_total(self, filter_):
        """raw YAMNet 인덱스 총합은 13개여야 한다 (glass_shatter가 2개)."""
        all_indices = []
        for entry in filter_.classes:
            all_indices.extend(entry.indices)
        assert len(all_indices) == 13
        # 중복 없음
        assert len(set(all_indices)) == 13

    def test_glass_shatter_has_two_indices(self, filter_):
        """glass_shatter만 인덱스 2개를 가져야 한다."""
        for entry in filter_.classes:
            if entry.key == "glass_shatter":
                assert len(entry.indices) == 2
            else:
                assert len(entry.indices) == 1

    def test_glass_shatter_uses_max(self, filter_):
        """glass_shatter는 두 인덱스 중 max 값을 사용 (해당 entry.indices에서 동적 추출)."""
        glass_entry = next(e for e in filter_.classes if e.key == "glass_shatter")
        idx_a, idx_b = glass_entry.indices

        scores = np.zeros(521, dtype=np.float32)
        scores[idx_a] = 0.2
        scores[idx_b] = 0.85
        result = filter_.extract(scores)
        assert result["glass_shatter"] == pytest.approx(0.85, abs=1e-6)

        # 반대 방향도 확인
        scores2 = np.zeros(521, dtype=np.float32)
        scores2[idx_a] = 0.9
        scores2[idx_b] = 0.1
        result2 = filter_.extract(scores2)
        assert result2["glass_shatter"] == pytest.approx(0.9, abs=1e-6)

    def test_each_class_hot_index_reflects_score(self, filter_):
        """각 단일 인덱스 클래스에 핫원 주입 → 해당 클래스만 점수 반영."""
        rng = np.random.default_rng(123)
        for entry in filter_.classes:
            scores = np.zeros(521, dtype=np.float32)
            target_score = float(rng.uniform(0.3, 0.95))
            for idx in entry.indices:
                scores[idx] = target_score
            result = filter_.extract(scores)
            # 해당 클래스는 target_score
            assert result[entry.key] == pytest.approx(target_score, abs=1e-6)
            # 다른 클래스는 0 (단, glass_shatter <-> breaking 등 인덱스 충돌은 whitelist 정의상 없음)
            for other in filter_.classes:
                if other.key == entry.key:
                    continue
                # 다른 클래스의 indices와 본 entry.indices가 disjoint해야 score 0이 보장됨
                if set(other.indices).isdisjoint(set(entry.indices)):
                    assert result[other.key] == pytest.approx(0.0, abs=1e-6)

    def test_synthetic_signal_score_vector_shape(self, filter_):
        """합성 신호처럼 만든 무작위 521 벡터도 valid input이어야 한다."""
        rng = np.random.default_rng(0)
        scores = rng.uniform(0.0, 1.0, 521).astype(np.float32)
        result = filter_.extract(scores)
        assert set(result.keys()) == {e.key for e in filter_.classes}
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_invalid_shape_raises(self, filter_):
        with pytest.raises(ValueError):
            filter_.extract(np.zeros(520, dtype=np.float32))
        with pytest.raises(ValueError):
            filter_.extract(np.zeros((2, 521), dtype=np.float32))
