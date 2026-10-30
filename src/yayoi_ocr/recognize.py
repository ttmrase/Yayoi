"""認識エンジンのインターフェース(Phase 2 以降で実装)。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .models import ColumnKind


@dataclass
class Recognition:
    text: str
    char_confidences: list[float] = field(default_factory=list)
    alternatives: list[tuple[str, float]] = field(default_factory=list)


class Recognizer(Protocol):
    """セル画像 1 枚 → 認識結果。`column` は列の種類(語彙・文字集合の絞り込みに使う)。"""

    def recognize(self, cell_image: np.ndarray, column: ColumnKind) -> Recognition: ...
