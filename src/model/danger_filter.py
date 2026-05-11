"""화이트리스트 YAML을 로드하고 YAMNet scores에서 위험 클래스 점수를 추출한다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config" / "whitelist.yaml"

# debounce 블록이 없을 때 사용하는 기본값 (M1 설정 파일 하위 호환)
_DEFAULT_DEBOUNCE_WINDOW = 3
_DEFAULT_DEBOUNCE_K = 2


@dataclass
class DebounceConfig:
    """YAML debounce 블록에서 읽어온 글로벌 debounce 설정."""

    window: int = _DEFAULT_DEBOUNCE_WINDOW
    k: int = _DEFAULT_DEBOUNCE_K


class DangerClassEntry:
    """화이트리스트 클래스 하나의 설정을 담는 데이터 클래스."""

    def __init__(self, raw: dict) -> None:
        # YAML 한 항목(딕셔너리)을 받아 필드를 채운다.
        self.key: str = raw["key"]                                # 시스템 내부 식별자 (예: "screaming")
        self.display_name: str = raw.get("display_name", self.key)  # 로그/UI 표시용 이름
        # yamnet_indices 우선, 없으면 yamnet_index 단일 값을 리스트로 변환
        # (glass_shatter처럼 여러 YAMNet 인덱스를 묶을 때 사용)
        if "yamnet_indices" in raw:
            self.indices: List[int] = list(raw["yamnet_indices"])
        else:
            self.indices = [int(raw["yamnet_index"])]
        self.threshold: float = float(raw.get("threshold", 0.5))   # 트리거 임계값 (기본 0.5)
        self.cooldown_sec: float = float(raw.get("cooldown_sec", 5.0))  # 동일 클래스 재트리거 억제 시간


class DangerFilter:
    """whitelist.yaml을 로드하여 521차원 scores 벡터에서 위험 클래스 점수를 추출한다."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG) -> None:
        # 1) whitelist.yaml 파싱.
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # 2) danger_classes 배열 → 12종 엔트리 리스트로 변환.
        self.classes: List[DangerClassEntry] = [
            DangerClassEntry(c) for c in raw["danger_classes"]
        ]

        # 3) 글로벌 debounce 블록 로드. 블록이 없는 M1 시점의 YAML도 그대로 로드되도록 기본값 fallback.
        debounce_raw = raw.get("debounce", {})
        self.debounce_config = DebounceConfig(
            window=int(debounce_raw.get("window", _DEFAULT_DEBOUNCE_WINDOW)),
            k=int(debounce_raw.get("k", _DEFAULT_DEBOUNCE_K)),
        )

    def extract(self, scores_521: np.ndarray) -> Dict[str, float]:
        """shape (521,) scores 벡터에서 화이트리스트 클래스별 점수를 반환한다.

        복수 인덱스(glass_shatter: [435, 437])는 max()로 통합한다.

        Returns:
            {class_key: score, ...}  — 12종 딕셔너리
        """
        # YAMNet은 521개 AudioSet 클래스를 출력하므로 shape 검증으로 입력 오류를 빠르게 잡는다.
        if scores_521.ndim != 1 or scores_521.shape[0] != 521:
            raise ValueError(
                f"scores_521 shape must be (521,), got {scores_521.shape}"
            )

        # 각 위험 클래스마다 등록된 YAMNet 인덱스들의 점수 중 최댓값을 채택.
        # (예: glass_shatter는 indices=[435, 437]을 묶어 max로 단일 score 산출)
        result: Dict[str, float] = {}
        for entry in self.classes:
            score = float(np.max(scores_521[entry.indices]))
            result[entry.key] = score
        return result

    def override_threshold(self, threshold: float) -> None:
        """모든 클래스의 임계값을 일괄 변경한다 (CLI --threshold 오버라이드용)."""
        for entry in self.classes:
            entry.threshold = threshold
