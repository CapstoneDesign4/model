"""Debounce K/N 후처리 단위 테스트 (M2).

YAMNet 모델 로딩 없이 합성 score 시퀀스를 직접 주입하여 검증한다.
모든 테스트에서 now= 인자를 명시적으로 주입하여 결정론적으로 동작한다.

실행: pytest tests/test_debounce_trigger.py -v

[동작 원칙]
스펙 §3.5: deque 크기가 N 미만이어도 평가를 실시한다.
예) N=3, K=2, deque=[1,1] → sum=2 >= K=2 → trigger 가능.
따라서 [1,1,0] 시퀀스에서 trigger는 두 번째 윈도우(deque=[1,1])에서 발생하고,
세 번째 윈도우(deque=[1,1,0], sum=2)는 cooldown에 의해 억제된다. 결과적으로 1회만 emit.
"""

from __future__ import annotations

from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from src.postprocess.trigger import DebounceState, Trigger, TriggerEvent


# ─────────────────────────────────────────────────────────────────────────────
# 공통 픽스처
# ─────────────────────────────────────────────────────────────────────────────

ALL_KEYS = [
    "screaming", "baby_cry", "glass_shatter", "breaking",
    "gunshot", "explosion", "fire_alarm", "smoke_alarm",
    "siren", "civil_defense_siren", "car_alarm", "vehicle_horn",
]

THRESHOLD = 0.5
SCORE_ABOVE = 0.8   # threshold 초과 → vote=1
SCORE_BELOW = 0.2   # threshold 미달 → vote=0
COOLDOWN_SEC = 5.0


def _make_scores(high_keys: List[str]) -> Dict[str, float]:
    """지정한 클래스만 threshold 초과 score를 갖는 합성 딕셔너리를 생성한다."""
    return {
        key: (SCORE_ABOVE if key in high_keys else SCORE_BELOW)
        for key in ALL_KEYS
    }


def _make_danger_filter_mock(threshold: float = THRESHOLD, cooldown_sec: float = COOLDOWN_SEC):
    """DangerFilter를 흉내내는 Mock 객체를 생성한다.

    실제 YAML 파일이나 YAMNet 없이 Trigger 생성에 필요한 인터페이스만 제공한다.
    """
    entries = []
    for key in ALL_KEYS:
        entry = MagicMock()
        entry.key = key
        entry.display_name = key
        entry.threshold = threshold
        entry.cooldown_sec = cooldown_sec
        entries.append(entry)

    mock_filter = MagicMock()
    mock_filter.classes = entries
    return mock_filter


@pytest.fixture
def danger_filter():
    """12종 클래스 Mock DangerFilter 픽스처."""
    return _make_danger_filter_mock()


@pytest.fixture
def trigger_23(danger_filter):
    """debounce window=3, k=2, cooldown=5s 기본 Trigger 픽스처."""
    return Trigger(danger_filter, debounce_window=3, debounce_k=2, debounce_enabled=True)


def _count_triggers(trigger: Trigger, vote_sequence: List[int], high_key: str = "screaming") -> int:
    """주어진 vote 시퀀스 전체에서 특정 클래스가 trigger된 총 횟수를 반환한다.

    각 윈도우는 1초 간격(ts=0.0, 1.0, 2.0, ...)으로 주입한다.
    cooldown=5초이므로 연속 소수 간격에서는 중복 trigger가 억제된다.
    """
    total = 0
    for i, vote in enumerate(vote_sequence):
        scores = _make_scores([high_key] if vote == 1 else [])
        events = trigger.evaluate(scores, now=float(i))
        total += sum(1 for e in events if e.key == high_key)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# DebounceState 단위 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestDebounceState:
    """DebounceState 내부 로직을 독립적으로 검증한다."""

    def test_initial_state_empty(self):
        """초기 생성 시 votes deque가 비어 있어야 한다."""
        state = DebounceState(window=3, k=2)
        assert len(state.votes) == 0

    def test_push_appends_vote(self):
        """push 후 votes 길이가 1 증가해야 한다."""
        state = DebounceState(window=3, k=2)
        state.push(1)
        assert list(state.votes) == [1]

    def test_maxlen_drops_oldest(self):
        """maxlen=3인 deque에 4번 push 시 가장 오래된 항목이 제거되어야 한다."""
        state = DebounceState(window=3, k=2)
        for v in [1, 0, 1, 0]:
            state.push(v)
        assert list(state.votes) == [0, 1, 0]

    def test_debounce_passed_when_sum_gte_k(self):
        """votes 합계 >= K 이면 is_debounce_passed()가 True를 반환해야 한다."""
        state = DebounceState(window=3, k=2)
        state.push(1)
        state.push(1)
        assert state.is_debounce_passed() is True

    def test_debounce_not_passed_when_sum_lt_k(self):
        """votes 합계 < K 이면 is_debounce_passed()가 False를 반환해야 한다."""
        state = DebounceState(window=3, k=2)
        state.push(1)
        state.push(0)
        assert state.is_debounce_passed() is False

    def test_cooldown_active_immediately_after_trigger(self):
        """trigger 직후 is_cooldown_active()가 True를 반환해야 한다."""
        state = DebounceState(window=3, k=2)
        state.record_trigger(ts=100.0)
        assert state.is_cooldown_active(now=101.0, cooldown_sec=5.0) is True

    def test_cooldown_inactive_after_elapsed(self):
        """cooldown_sec 경과 후 is_cooldown_active()가 False를 반환해야 한다."""
        state = DebounceState(window=3, k=2)
        state.record_trigger(ts=100.0)
        assert state.is_cooldown_active(now=106.0, cooldown_sec=5.0) is False

    def test_initial_cooldown_inactive(self):
        """초기 상태에서 is_cooldown_active()가 False를 반환해야 한다 (last_trigger_ts = -inf)."""
        state = DebounceState(window=3, k=2)
        assert state.is_cooldown_active(now=0.0, cooldown_sec=5.0) is False

    def test_snapshot_returns_list_copy(self):
        """snapshot()이 현재 votes를 리스트로 반환해야 한다."""
        state = DebounceState(window=3, k=2)
        state.push(1)
        state.push(0)
        snap = state.snapshot()
        assert snap == [1, 0]
        assert isinstance(snap, list)


# ─────────────────────────────────────────────────────────────────────────────
# TC-1 ~ TC-5: 스펙 §10 케이스 전체 구현
# ─────────────────────────────────────────────────────────────────────────────

class TestDebounceTC1:
    """TC-1: [1, 1, 0] 시퀀스 → trigger 정확히 1회 발생 (K=2/N=3).

    스펙 §3.5에 따라 deque=[1,1](2번째 윈도우)에서 sum=2 >= K=2 → 첫 trigger.
    3번째 윈도우(deque=[1,1,0], sum=2)는 cooldown에 의해 억제된다.
    전체 시퀀스에서 trigger는 1회만 발생한다.
    """

    def test_trigger_fires_exactly_once(self, trigger_23):
        """[1, 1, 0] 시퀀스 전체에서 screaming trigger가 정확히 1회 발생해야 한다."""
        count = _count_triggers(trigger_23, [1, 1, 0])
        assert count == 1

    def test_trigger_fires_on_second_window(self, trigger_23):
        """[1, 1] 처리 시점(두 번째 윈도우)에서 trigger가 발생해야 한다.

        N=3, K=2인데 deque=[1,1]로도 sum=2 >= K=2 조건을 충족하므로 즉시 trigger.
        """
        scores_high = _make_scores(["screaming"])

        result1 = trigger_23.evaluate(scores_high, now=0.0)   # deque=[1]
        assert "screaming" not in [e.key for e in result1]

        result2 = trigger_23.evaluate(scores_high, now=1.0)   # deque=[1,1], sum=2 >= K=2
        assert "screaming" in [e.key for e in result2]

    def test_trigger_suppressed_on_third_window_by_cooldown(self, trigger_23):
        """[1, 1, 0] 세 번째 윈도우는 cooldown에 의해 trigger가 억제되어야 한다."""
        scores_high = _make_scores(["screaming"])
        scores_low = _make_scores([])

        trigger_23.evaluate(scores_high, now=0.0)
        trigger_23.evaluate(scores_high, now=1.0)   # trigger 발생 (ts=1.0)

        # 3번째 윈도우: ts=2.0, elapsed = 2.0 - 1.0 = 1.0 < cooldown=5.0
        result3 = trigger_23.evaluate(scores_low, now=2.0)
        assert "screaming" not in [e.key for e in result3]

    def test_trigger_event_has_debounce_votes(self, trigger_23):
        """trigger 이벤트의 debounce_votes 필드가 스냅샷을 포함해야 한다."""
        scores_high = _make_scores(["screaming"])

        trigger_23.evaluate(scores_high, now=0.0)
        result = trigger_23.evaluate(scores_high, now=1.0)   # trigger 발생

        screaming_events = [e for e in result if e.key == "screaming"]
        assert len(screaming_events) == 1
        # deque=[1,1] 상태에서 trigger
        assert screaming_events[0].debounce_votes == [1, 1]


class TestDebounceTC2:
    """TC-2: 비연속 양성 [1, 0, 1] → trigger 1회 발생 (K=2/N=3).

    1번째 윈도우: deque=[1], sum=1 < K=2 → skip.
    2번째 윈도우: deque=[1,0], sum=1 < K=2 → skip.
    3번째 윈도우: deque=[1,0,1], sum=2 >= K=2 → trigger.
    """

    def test_trigger_fires_exactly_once(self, trigger_23):
        """[1, 0, 1] 시퀀스에서 trigger가 정확히 1회 발생해야 한다."""
        count = _count_triggers(trigger_23, [1, 0, 1])
        assert count == 1

    def test_trigger_fires_on_third_window(self, trigger_23):
        """[1, 0, 1] 시퀀스에서 세 번째 윈도우에서 trigger가 발생해야 한다."""
        scores_high = _make_scores(["screaming"])
        scores_low = _make_scores([])

        result1 = trigger_23.evaluate(scores_high, now=0.0)   # deque=[1], sum=1 < K
        assert "screaming" not in [e.key for e in result1]

        result2 = trigger_23.evaluate(scores_low, now=1.0)    # deque=[1,0], sum=1 < K
        assert "screaming" not in [e.key for e in result2]

        result3 = trigger_23.evaluate(scores_high, now=2.0)   # deque=[1,0,1], sum=2 >= K
        assert "screaming" in [e.key for e in result3]

    def test_votes_snapshot_is_1_0_1(self, trigger_23):
        """trigger 시점의 debounce_votes가 [1, 0, 1]이어야 한다."""
        scores_high = _make_scores(["screaming"])
        scores_low = _make_scores([])

        trigger_23.evaluate(scores_high, now=0.0)
        trigger_23.evaluate(scores_low, now=1.0)
        result = trigger_23.evaluate(scores_high, now=2.0)

        screaming_events = [e for e in result if e.key == "screaming"]
        assert screaming_events[0].debounce_votes == [1, 0, 1]


class TestDebounceTC3:
    """TC-3: 단발성 피크 [1, 0, 0] → trigger 없음 (K=2/N=3).

    어느 윈도우에서도 sum(deque) < K=2 이므로 debounce 미통과.
    """

    def test_no_trigger_on_single_vote(self, trigger_23):
        """[1, 0, 0] 시퀀스 전체에서 trigger가 발생하지 않아야 한다."""
        count = _count_triggers(trigger_23, [1, 0, 0])
        assert count == 0

    def test_no_trigger_each_window(self, trigger_23):
        """[1, 0, 0] 각 윈도우에서 개별 확인."""
        scores_high = _make_scores(["screaming"])
        scores_low = _make_scores([])

        result1 = trigger_23.evaluate(scores_high, now=0.0)   # deque=[1], sum=1 < K
        assert "screaming" not in [e.key for e in result1]

        result2 = trigger_23.evaluate(scores_low, now=1.0)    # deque=[1,0], sum=1 < K
        assert "screaming" not in [e.key for e in result2]

        result3 = trigger_23.evaluate(scores_low, now=2.0)    # deque=[1,0,0], sum=1 < K
        assert "screaming" not in [e.key for e in result3]

    def test_no_trigger_on_all_zeros(self, trigger_23):
        """[0, 0, 0] 시퀀스에서 trigger가 발생하지 않아야 한다."""
        scores_low = _make_scores([])
        for t in [0.0, 1.0, 2.0]:
            result = trigger_23.evaluate(scores_low, now=t)
            assert len(result) == 0


class TestDebounceTC4:
    """TC-4: 첫 trigger 후 cooldown 내 재시도 → 두 번째 trigger 없음.

    [1, 1] 시퀀스에서 ts=1.0에 trigger 발생(cooldown_sec=5).
    이후 ts=2.5, 3.0, 3.5에서 재시도 → elapsed < 5초이므로 cooldown 억제.
    스펙 §10.2 의사코드를 따른다.
    """

    def test_cooldown_suppresses_second_trigger(self, trigger_23):
        """첫 번째 trigger 발생 후 cooldown(5초) 내 동일 패턴 → trigger 없음이어야 한다."""
        scores_high = _make_scores(["screaming"])
        scores_low = _make_scores([])

        # 첫 번째 트리거 시퀀스: ts=0.0, 1.0
        # ts=1.0에서 deque=[1,1], sum=2 >= K=2 → trigger (last_trigger_ts=1.0)
        trigger_23.evaluate(scores_high, now=0.0)
        result1 = trigger_23.evaluate(scores_high, now=1.0)

        screaming_first = [e for e in result1 if e.key == "screaming"]
        assert len(screaming_first) == 1, "첫 번째 trigger가 발생해야 한다."

        # 두 번째 트리거 시도: cooldown_sec=5, elapsed = 2.5-1.0 = 1.5초 < 5초
        trigger_23.evaluate(scores_high, now=2.5)
        trigger_23.evaluate(scores_high, now=3.0)
        result2 = trigger_23.evaluate(scores_low, now=3.5)

        screaming_second = [e for e in result2 if e.key == "screaming"]
        assert len(screaming_second) == 0, "cooldown 중에는 trigger가 억제되어야 한다."

    def test_trigger_allowed_after_cooldown(self, trigger_23):
        """cooldown 경과 후 동일 패턴 → trigger 다시 발생해야 한다.

        흐름:
          ts=0.0: deque=[1]
          ts=1.0: deque=[1,1] → trigger (last_trigger_ts=1.0)
          ts=2.0: deque=[1,1,0], sum=2 → cooldown 중 억제
          ts=8.0: deque=[1,0,1], sum=2, elapsed=8.0-1.0=7.0 > 5.0 → cooldown 해제 → trigger 재발생
        """
        scores_high = _make_scores(["screaming"])
        scores_low = _make_scores([])

        # 첫 번째 트리거: ts=1.0에서 발생
        trigger_23.evaluate(scores_high, now=0.0)
        trigger_23.evaluate(scores_high, now=1.0)   # trigger (last_trigger_ts=1.0)
        trigger_23.evaluate(scores_low, now=2.0)    # cooldown 중

        # cooldown=5초, ts=1.0 + 5.0 = 6.0 이후 허용
        # ts=8.0: deque=[1,0,1], elapsed=7.0 > 5.0 → trigger 재발생
        result = trigger_23.evaluate(scores_high, now=8.0)

        screaming_events = [e for e in result if e.key == "screaming"]
        assert len(screaming_events) == 1, "cooldown 경과 후 trigger가 다시 발생해야 한다."


class TestDebounceTC5:
    """TC-5: debounce_enabled=False 시 단일 윈도우 즉시 trigger (M1 동작 재현)."""

    @pytest.fixture
    def trigger_no_debounce(self, danger_filter):
        """debounce 비활성화 Trigger 픽스처."""
        return Trigger(danger_filter, debounce_window=3, debounce_k=2, debounce_enabled=False)

    def test_trigger_fires_on_single_positive_vote(self, trigger_no_debounce):
        """debounce_enabled=False 시 vote=1 하나만으로 즉시 trigger가 발생해야 한다."""
        scores_high = _make_scores(["screaming"])

        result = trigger_no_debounce.evaluate(scores_high, now=0.0)
        keys = [e.key for e in result]
        assert "screaming" in keys

    def test_no_trigger_on_single_negative_vote(self, trigger_no_debounce):
        """debounce_enabled=False 시 score < threshold는 trigger가 없어야 한다."""
        scores_low = _make_scores([])

        result = trigger_no_debounce.evaluate(scores_low, now=0.0)
        assert len(result) == 0

    def test_no_debounce_respects_cooldown(self, trigger_no_debounce):
        """debounce_enabled=False 모드에서도 cooldown은 동작해야 한다."""
        scores_high = _make_scores(["screaming"])

        result1 = trigger_no_debounce.evaluate(scores_high, now=0.0)
        assert "screaming" in [e.key for e in result1]

        # cooldown 내 재시도 (elapsed=1.0 < cooldown=5.0)
        result2 = trigger_no_debounce.evaluate(scores_high, now=1.0)
        assert "screaming" not in [e.key for e in result2]

    def test_no_debounce_votes_empty(self, trigger_no_debounce):
        """debounce_enabled=False 시 TriggerEvent.debounce_votes가 빈 리스트여야 한다."""
        scores_high = _make_scores(["screaming"])

        result = trigger_no_debounce.evaluate(scores_high, now=0.0)
        screaming_events = [e for e in result if e.key == "screaming"]
        assert screaming_events[0].debounce_votes == []


# ─────────────────────────────────────────────────────────────────────────────
# 추가 검증: 다중 클래스 독립성, 초기 deque 미만 상태
# ─────────────────────────────────────────────────────────────────────────────

class TestDebounceAdditional:
    """스펙 §3.4(클래스 독립성)와 §3.5(윈도우 경계 처리) 검증."""

    def test_class_independence(self, trigger_23):
        """한 클래스의 투표가 다른 클래스에 영향을 주지 않아야 한다.

        screaming만 양성이면 screaming만 trigger되고 siren 등은 trigger 안 됨.
        """
        scores_screaming_only = _make_scores(["screaming"])

        # deque=[1,1] 시점(두 번째 윈도우)에서 screaming이 trigger됨
        trigger_23.evaluate(scores_screaming_only, now=0.0)
        result = trigger_23.evaluate(scores_screaming_only, now=1.0)

        triggered_keys = {e.key for e in result}
        assert "screaming" in triggered_keys
        assert "siren" not in triggered_keys
        assert "baby_cry" not in triggered_keys

    def test_trigger_with_partial_deque(self, trigger_23):
        """deque가 N 미만(초기 상태)에서도 K 충족 시 trigger가 발생해야 한다 (스펙 §3.5).

        N=3, K=2일 때 deque=[1,1](크기 2)이면 sum=2 >= K=2 → trigger 가능.
        """
        scores_high = _make_scores(["screaming"])

        trigger_23.evaluate(scores_high, now=0.0)   # deque=[1], sum=1 < K → skip
        result = trigger_23.evaluate(scores_high, now=1.0)  # deque=[1,1], sum=2 >= K → trigger

        keys = [e.key for e in result]
        assert "screaming" in keys, "deque가 N 미만이어도 K 충족 시 trigger 가능해야 한다."

    def test_get_debounce_snapshot_returns_all_classes(self, trigger_23):
        """get_debounce_snapshot()이 12종 클래스 전체 votes를 반환해야 한다."""
        scores = _make_scores([])
        trigger_23.evaluate(scores, now=0.0)

        snapshot = trigger_23.get_debounce_snapshot()
        assert set(snapshot.keys()) == set(ALL_KEYS)
        for key, votes in snapshot.items():
            assert isinstance(votes, list)

    def test_multiple_classes_trigger_simultaneously(self, trigger_23):
        """두 클래스가 동시에 K/N 조건을 충족하면 둘 다 trigger되어야 한다."""
        scores_multi = _make_scores(["screaming", "siren"])

        # deque=[1,1] 시점(두 번째 윈도우)에서 screaming, siren 모두 trigger됨
        trigger_23.evaluate(scores_multi, now=0.0)
        result = trigger_23.evaluate(scores_multi, now=1.0)

        triggered_keys = {e.key for e in result}
        assert "screaming" in triggered_keys
        assert "siren" in triggered_keys
