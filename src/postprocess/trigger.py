"""임계값 비교 + cooldown 기반 위험 이벤트 트리거 판정."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.model.danger_filter import DangerClassEntry, DangerFilter


@dataclass
class TriggerEvent:
    """트리거된 위험 이벤트 하나를 나타낸다."""

    key: str
    display_name: str
    score: float
    timestamp: float  # Unix epoch (초)


class Trigger:
    """클래스별 임계값 비교 및 cooldown 관리를 수행한다.

    M1: 단순 threshold + cooldown.
    M2 이후: K/N debounce 추가 예정.
    """

    def __init__(self, danger_filter: DangerFilter) -> None:
        self._filter = danger_filter
        # 클래스별 마지막 트리거 시각 (Unix epoch)
        self._last_trigger: Dict[str, float] = {
            entry.key: 0.0 for entry in danger_filter.classes
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
            now = time.time()

        events: List[TriggerEvent] = []
        entry_map: Dict[str, DangerClassEntry] = {
            e.key: e for e in self._filter.classes
        }

        for key, score in scores.items():
            entry = entry_map[key]
            elapsed = now - self._last_trigger[key]

            if score >= entry.threshold and elapsed >= entry.cooldown_sec:
                self._last_trigger[key] = now
                events.append(
                    TriggerEvent(
                        key=key,
                        display_name=entry.display_name,
                        score=score,
                        timestamp=now,
                    )
                )

        return events
