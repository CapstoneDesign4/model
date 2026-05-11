"""YAMNet 합성 신호 sanity check (네트워크/모델 로딩 필요).

함수/클래스 이름에 'yamnet' 포함 → `pytest -k yamnet` 로 선택 실행.

검증 대상:
- 무음 입력 시 위험 클래스 max score < 0.5 (false positive 없음).
- 화이트노이즈 입력 시 위험 클래스 max score < 0.5 (조용한 환경 sanity).
- scores 벡터가 (521,) shape이고 값이 [0,1] 범위.

모델 로딩 실패(네트워크 없음, TF 미설치) 시 graceful skip.

실행: pytest tests/test_yamnet_synthetic.py -v -k yamnet
"""

from __future__ import annotations

import numpy as np
import pytest


def _load_yamnet_or_skip():
    """YAMNet 로딩을 시도하고 실패 시 pytest.skip."""
    try:
        from src.model.yamnet_wrapper import YAMNetWrapper
    except ImportError as e:
        pytest.skip(f"tensorflow/tensorflow_hub 미설치: {e}")
    try:
        return YAMNetWrapper()
    except Exception as e:  # 네트워크 실패, TF-Hub 캐시 문제 등
        pytest.skip(f"YAMNet 로딩 실패: {e}")


@pytest.fixture(scope="module")
def yamnet_synthetic_model():
    """모듈 범위 YAMNet 픽스처 — 로딩 1회만."""
    return _load_yamnet_or_skip()


@pytest.fixture(scope="module")
def yamnet_synthetic_filter():
    from src.model.danger_filter import DangerFilter
    return DangerFilter()


class TestYamnetSyntheticSilence:
    """무음 입력 — 위험 클래스가 절대 트리거되면 안 된다."""

    def test_yamnet_silence_scores_shape(self, yamnet_synthetic_model):
        """0.96s 무음 입력의 평균 score 벡터는 (521,) 이어야 한다."""
        sig = np.zeros(15360, dtype=np.float32)
        mean = yamnet_synthetic_model.infer_mean_scores(sig)
        assert mean.shape == (521,)
        assert mean.min() >= -1e-6
        assert mean.max() <= 1.0 + 1e-6

    def test_yamnet_silence_no_danger_above_threshold(
        self, yamnet_synthetic_model, yamnet_synthetic_filter
    ):
        """무음 → 모든 위험 클래스 score < 0.5."""
        sig = np.zeros(15360, dtype=np.float32)
        mean = yamnet_synthetic_model.infer_mean_scores(sig)
        scores = yamnet_synthetic_filter.extract(mean)
        max_key = max(scores, key=lambda k: scores[k])
        assert scores[max_key] < 0.5, (
            f"무음에서 위험 클래스 점수 과대: {max_key}={scores[max_key]:.4f}"
        )


class TestYamnetSyntheticWhiteNoise:
    """저진폭 화이트노이즈 — 조용한 환경 sanity check."""

    def test_yamnet_white_noise_no_danger(
        self, yamnet_synthetic_model, yamnet_synthetic_filter
    ):
        """화이트노이즈 amplitude=0.05 → 위험 클래스 max < 0.5."""
        rng = np.random.default_rng(2026)
        sig = (0.05 * rng.standard_normal(15360)).astype(np.float32)
        mean = yamnet_synthetic_model.infer_mean_scores(sig)
        scores = yamnet_synthetic_filter.extract(mean)
        max_key = max(scores, key=lambda k: scores[k])
        assert scores[max_key] < 0.5, (
            f"화이트노이즈에서 위험 클래스 점수 과대: {max_key}={scores[max_key]:.4f}"
        )

    def test_yamnet_white_noise_score_range(self, yamnet_synthetic_model):
        """화이트노이즈 추론도 score 범위 [0, 1] 유지."""
        rng = np.random.default_rng(42)
        sig = (0.05 * rng.standard_normal(15360)).astype(np.float32)
        mean = yamnet_synthetic_model.infer_mean_scores(sig)
        assert mean.min() >= -1e-6
        assert mean.max() <= 1.0 + 1e-6
