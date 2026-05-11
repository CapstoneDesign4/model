"""임계값 비교 + cooldown + Debounce K/N 후처리 기반 위험 이벤트 트리거 판정."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.model.danger_filter import DangerClassEntry, DangerFilter


@dataclass
class TriggerEvent:
    """트리거된 위험 이벤트 하나를 나타낸다."""

    key: str
    display_name: str
    score: float
    timestamp: float           # Unix epoch (초)
    debounce_votes: List[int] = field(default_factory=list)  # M2: 스냅샷 [오래된순 → 최신]


class DebounceState:
    """단일 위험 클래스의 슬라이딩 윈도우 투표 상태를 관리한다.

    deque(maxlen=N)에 이진 투표(0 or 1)를 순서대로 쌓는다.
    sum(votes) >= K 이면 debounce 통과로 판정한다.
    """

    def __init__(self, window: int, k: int) -> None:
        self.N: int = window  # 슬라이딩 윈도우 크기
        self.K: int = k       # 트리거 발생에 필요한 최소 양성 투표 수
        # deque(maxlen=N)이므로 push할 때 자동으로 가장 오래된 항목이 빠진다.
        self.votes: deque[int] = deque(maxlen=window)
        # -inf 대신 float('-inf')를 사용해 초기 cooldown 체크가 항상 통과하도록 한다.
        self.last_trigger_ts: float = float("-inf")

    def push(self, vote: int) -> None:
        """새 투표(0 or 1)를 deque 끝에 추가한다. maxlen 초과 시 가장 오래된 항목이 자동 제거된다."""
        self.votes.append(vote)

    def is_debounce_passed(self) -> bool:
        """슬라이딩 윈도우 내 양성 투표 합계가 K 이상인지 반환한다."""
        return sum(self.votes) >= self.K

    def is_cooldown_active(self, now: float, cooldown_sec: float) -> bool:
        """마지막 트리거로부터 cooldown_sec이 아직 경과하지 않았는지 반환한다."""
        return (now - self.last_trigger_ts) < cooldown_sec

    def record_trigger(self, ts: float) -> None:
        """트리거 발생 시각을 기록한다."""
        self.last_trigger_ts = ts

    def snapshot(self) -> List[int]:
        """현재 votes deque를 [오래된순 → 최신순] 리스트로 반환한다."""
        return list(self.votes)


class Trigger:
    """클래스별 임계값 비교, Debounce K/N 다수결, cooldown 관리를 수행한다.

    M1: debounce_enabled=False 또는 no_debounce=True 시 단순 threshold + cooldown.
    M2: debounce_enabled=True(기본) 시 슬라이딩 윈도우 K/N 다수결 후 cooldown.
    """

    def __init__(
        self,
        danger_filter: DangerFilter,
        debounce_window: int = 3,
        debounce_k: int = 2,
        debounce_enabled: bool = True,
    ) -> None:
        self._filter = danger_filter
        self._debounce_enabled = debounce_enabled

        # 클래스별 독립 DebounceState 초기화.
        # 한 클래스의 투표 이력이 다른 클래스에 영향을 주지 않도록 별도 상태를 유지한다.
        self._debounce_states: Dict[str, DebounceState] = {
            entry.key: DebounceState(window=debounce_window, k=debounce_k)
            for entry in danger_filter.classes
        }

    def evaluate(
        self,
        scores: Dict[str, float],
        now: Optional[float] = None,
    ) -> List[TriggerEvent]:
        """scores 딕셔너리를 받아 트리거 조건을 만족한 이벤트 목록을 반환한다.

        Args:
            scores: DangerFilter.extract() 결과 {key: score}.
            now:    현재 시각 (Unix epoch). None이면 time.time() 사용.

        Returns:
            트리거된 TriggerEvent 리스트 (없으면 빈 리스트).
        """
        if now is None:
            now = time.time()  # 테스트에서는 결정론적 시각을 명시 주입한다.

        events: List[TriggerEvent] = []
        # key → DangerClassEntry 빠른 조회용 맵.
        entry_map: Dict[str, DangerClassEntry] = {
            e.key: e for e in self._filter.classes
        }

        # 12종 클래스 각각에 대해 독립적으로 판정.
        for key, score in scores.items():
            entry = entry_map[key]
            state = self._debounce_states[key]

            # 임계값을 넘으면 1, 아니면 0인 이진 투표로 변환.
            vote = 1 if score >= entry.threshold else 0

            if self._debounce_enabled:
                # M2 경로: 투표 누적 후 K/N 다수결 평가
                state.push(vote)

                # 순서: debounce 통과 → cooldown 미활성 → emit
                if state.is_debounce_passed():
                    if not state.is_cooldown_active(now, entry.cooldown_sec):
                        state.record_trigger(now)  # cooldown 시작점 갱신
                        events.append(
                            TriggerEvent(
                                key=key,
                                display_name=entry.display_name,
                                score=score,
                                timestamp=now,
                                debounce_votes=state.snapshot(),  # 디버깅용 투표 스냅샷
                            )
                        )
            else:
                # M1 경로(--no-debounce): deque 무시, 단일 윈도우 즉시 판정
                # debounce_states의 last_trigger_ts를 cooldown 추적에 재활용
                if vote == 1 and not state.is_cooldown_active(now, entry.cooldown_sec):
                    state.record_trigger(now)
                    events.append(
                        TriggerEvent(
                            key=key,
                            display_name=entry.display_name,
                            score=score,
                            timestamp=now,
                            debounce_votes=[],  # no-debounce 모드는 votes 미사용
                        )
                    )

        return events

    def get_debounce_snapshot(self) -> Dict[str, List[int]]:
        """모든 클래스의 현재 debounce votes 스냅샷을 반환한다 (JSONL 로깅용)."""
        return {key: state.snapshot() for key, state in self._debounce_states.items()}

    def get_debounce_state(self, key: str) -> DebounceState:
        """특정 클래스의 DebounceState를 반환한다 (테스트·verbose 출력용)."""
        return self._debounce_states[key]
