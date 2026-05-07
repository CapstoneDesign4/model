"""YAMNet 로딩 및 출력 shape 단위 테스트.

네트워크가 없는 환경에서는 자동으로 skip된다.
실행: pytest tests/test_yamnet_load.py -v
"""

from __future__ import annotations

import numpy as np
import pytest


def _yamnet_available() -> bool:
    """TF-Hub YAMNet 접근 가능 여부를 확인한다."""
    try:
        import tensorflow_hub as hub  # noqa: F401
        import tensorflow as tf       # noqa: F401
        return True
    except ImportError:
        return False


requires_yamnet = pytest.mark.skipif(
    not _yamnet_available(),
    reason="tensorflow / tensorflow-hub 미설치 또는 네트워크 불가",
)


@requires_yamnet
class TestYAMNetLoad:
    """YAMNet 로딩 및 기본 추론 테스트."""

    @pytest.fixture(scope="class")
    def yamnet(self):
        """클래스 범위 YAMNet 픽스처 — 로딩은 1회만."""
        from src.model.yamnet_wrapper import YAMNetWrapper
        return YAMNetWrapper()

    def test_load_succeeds(self, yamnet):
        """YAMNet 인스턴스가 정상 생성되어야 한다."""
        assert yamnet is not None

    def test_scores_shape(self, yamnet):
        """0.96s 더미 입력에 대해 scores shape이 (N, 521)이어야 한다."""
        dummy = np.zeros(15360, dtype=np.float32)
        scores, embeddings, spectrogram = yamnet.infer(dummy)
        assert scores.ndim == 2
        assert scores.shape[1] == 521

    def test_embeddings_shape(self, yamnet):
        """embeddings shape이 (N, 1024)이어야 한다."""
        dummy = np.zeros(15360, dtype=np.float32)
        _, embeddings, _ = yamnet.infer(dummy)
        assert embeddings.ndim == 2
        assert embeddings.shape[1] == 1024

    def test_mean_scores_shape(self, yamnet):
        """infer_mean_scores 결과가 shape (521,)이어야 한다."""
        dummy = np.zeros(15360, dtype=np.float32)
        mean_scores = yamnet.infer_mean_scores(dummy)
        assert mean_scores.shape == (521,)

    def test_scores_range(self, yamnet):
        """scores 값이 [0, 1] 범위 내여야 한다."""
        dummy = np.random.uniform(-1.0, 1.0, 15360).astype(np.float32)
        scores, _, _ = yamnet.infer(dummy)
        assert scores.min() >= 0.0 - 1e-6
        assert scores.max() <= 1.0 + 1e-6


class TestDangerFilter:
    """DangerFilter — 네트워크 불필요, 항상 실행."""

    @pytest.fixture
    def filter_(self):
        from src.model.danger_filter import DangerFilter
        return DangerFilter()

    def test_extract_keys(self, filter_):
        """extract() 결과에 12종 클래스 키가 모두 포함되어야 한다."""
        dummy_scores = np.zeros(521, dtype=np.float32)
        result = filter_.extract(dummy_scores)
        expected_keys = {
            "screaming", "baby_cry", "glass_shatter", "breaking",
            "gunshot", "explosion", "fire_alarm", "smoke_alarm",
            "siren", "civil_defense_siren", "car_alarm", "vehicle_horn",
        }
        assert set(result.keys()) == expected_keys

    def test_glass_shatter_max(self, filter_):
        """glass_shatter는 인덱스 435, 437 중 max 값을 사용해야 한다."""
        dummy_scores = np.zeros(521, dtype=np.float32)
        dummy_scores[435] = 0.3
        dummy_scores[437] = 0.7
        result = filter_.extract(dummy_scores)
        assert abs(result["glass_shatter"] - 0.7) < 1e-6

    def test_extract_invalid_shape(self, filter_):
        """shape이 (521,)이 아니면 ValueError가 발생해야 한다."""
        with pytest.raises(ValueError):
            filter_.extract(np.zeros(100, dtype=np.float32))

    def test_override_threshold(self, filter_):
        """override_threshold() 후 모든 클래스 임계값이 변경되어야 한다."""
        filter_.override_threshold(0.3)
        for entry in filter_.classes:
            assert abs(entry.threshold - 0.3) < 1e-6


class TestTrigger:
    """Trigger — cooldown 로직 테스트."""

    @pytest.fixture
    def trigger(self):
        from src.model.danger_filter import DangerFilter
        from src.postprocess.trigger import Trigger
        f = DangerFilter()
        f.override_threshold(0.5)
        return Trigger(f)

    def test_trigger_fires_above_threshold(self, trigger):
        """임계값 초과 score는 TriggerEvent를 반환해야 한다."""
        scores = {
            "screaming": 0.9, "baby_cry": 0.1, "glass_shatter": 0.1,
            "breaking": 0.1, "gunshot": 0.1, "explosion": 0.1,
            "fire_alarm": 0.1, "smoke_alarm": 0.1, "siren": 0.1,
            "civil_defense_siren": 0.1, "car_alarm": 0.1, "vehicle_horn": 0.1,
        }
        events = trigger.evaluate(scores, now=1000.0)
        keys = [e.key for e in events]
        assert "screaming" in keys

    def test_cooldown_suppresses_repeat(self, trigger):
        """cooldown 내 동일 클래스 재트리거는 억제되어야 한다."""
        scores = {k: 0.9 for k in [
            "screaming", "baby_cry", "glass_shatter", "breaking",
            "gunshot", "explosion", "fire_alarm", "smoke_alarm",
            "siren", "civil_defense_siren", "car_alarm", "vehicle_horn",
        ]}
        trigger.evaluate(scores, now=1000.0)   # 1차 트리거
        events = trigger.evaluate(scores, now=1001.0)  # 1초 후 — cooldown 5초
        keys = [e.key for e in events]
        assert "screaming" not in keys

    def test_cooldown_allows_after_elapsed(self, trigger):
        """cooldown 경과 후에는 동일 클래스가 다시 트리거되어야 한다."""
        scores = {k: 0.9 for k in [
            "screaming", "baby_cry", "glass_shatter", "breaking",
            "gunshot", "explosion", "fire_alarm", "smoke_alarm",
            "siren", "civil_defense_siren", "car_alarm", "vehicle_horn",
        ]}
        trigger.evaluate(scores, now=1000.0)
        events = trigger.evaluate(scores, now=1006.0)  # 6초 후 — cooldown 해제
        keys = [e.key for e in events]
        assert "screaming" in keys
