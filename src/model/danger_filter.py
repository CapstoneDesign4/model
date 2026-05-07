"""화이트리스트 YAML을 로드하고 YAMNet scores에서 위험 클래스 점수를 추출한다."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config" / "whitelist.yaml"


class DangerClassEntry:
    """화이트리스트 클래스 하나의 설정을 담는 데이터 클래스."""

    def __init__(self, raw: dict) -> None:
        self.key: str = raw["key"]
        self.display_name: str = raw.get("display_name", self.key)
        # yamnet_indices 우선, 없으면 yamnet_index 단일 값을 리스트로 변환
        if "yamnet_indices" in raw:
            self.indices: List[int] = list(raw["yamnet_indices"])
        else:
            self.indices = [int(raw["yamnet_index"])]
        self.threshold: float = float(raw.get("threshold", 0.5))
        self.cooldown_sec: float = float(raw.get("cooldown_sec", 5.0))


class DangerFilter:
    """whitelist.yaml을 로드하여 521차원 scores 벡터에서 위험 클래스 점수를 추출한다."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG) -> None:
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self.classes: List[DangerClassEntry] = [
            DangerClassEntry(c) for c in raw["danger_classes"]
        ]

    def extract(self, scores_521: np.ndarray) -> Dict[str, float]:
        """shape (521,) scores 벡터에서 화이트리스트 클래스별 점수를 반환한다.

        복수 인덱스(glass_shatter: [435, 437])는 max()로 통합한다.

        Returns:
            {class_key: score, ...}  — 12종 딕셔너리
        """
        if scores_521.ndim != 1 or scores_521.shape[0] != 521:
            raise ValueError(
                f"scores_521 shape must be (521,), got {scores_521.shape}"
            )

        result: Dict[str, float] = {}
        for entry in self.classes:
            score = float(np.max(scores_521[entry.indices]))
            result[entry.key] = score
        return result

    def override_threshold(self, threshold: float) -> None:
        """모든 클래스의 임계값을 일괄 변경한다 (CLI --threshold 오버라이드용)."""
        for entry in self.classes:
            entry.threshold = threshold
