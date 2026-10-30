"""PARSeq の初期重みの種類(素 / ドメイン適応済み)の置き場所と、確定データでの比較・選択。

- 素("base"): NDLOCR-Lite の ONNX 由来(`models/`)。ドメイン適応済み("adapted"): 合成データで追加学習したもの(`models/adapted/`)。
- 利用者には選ばせない。確定済みのページが `MIN_CONFIRMED_PAGES`(3 ページ)たまったら、その確定セルで両方を比べ、
  良いほうを自動で採用する(`choose_variant`)。それまでは `default_variant`(合成データの評価が良いほう。適応済みがあればそれ)。
- 比較の仕組み(`score_variant`)は Phase 4 の採用ゲートと同じものを使う(確定セルでの値の正解率に、見逃しのペナルティを加えた点数)。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import models_dir
from .parseq.engine import MODEL_FILES, ParseqEngine

VARIANTS = ("adapted", "base")  # 既定の優先順
MIN_CONFIRMED_PAGES = 3


def variant_dir(name: str, root: Path | None = None) -> Path:
    root = root or models_dir()
    return root if name == "base" else root / name


def available_variants(root: Path | None = None) -> list[str]:
    out = []
    for v in VARIANTS:
        d = variant_dir(v, root)
        if all((d / f).is_file() for f in MODEL_FILES.values()):
            out.append(v)
    return out


def default_variant(root: Path | None = None) -> str:
    """確定ページが足りないうちの既定。適応済みの重みがあればそれ(合成データでの評価が良い)、無ければ素。"""
    av = available_variants(root)
    return av[0] if av else "base"


def open_engine(name: str, root: Path | None = None, **kw) -> ParseqEngine:
    return ParseqEngine(models_dir=variant_dir(name, root), **kw)


@dataclass
class VariantScore:
    name: str
    accuracy: float  # 確定した text セルの値の正解率
    miss_penalty: float  # 見逃し(間違い ∧ 要確認なし)の割合 x 重み
    score: float


def score_variant(accuracy: float, miss_rate: float, miss_weight: float = 20.0) -> float:
    """採用ゲート共通の点数: 正解率 − 重み x 見逃し率(見逃しを強く嫌う)。"""
    return accuracy - miss_weight * miss_rate


def choose_variant(
    n_confirmed_pages: int, scores: Sequence[VariantScore] | None, root: Path | None = None
) -> str:
    """確定ページが足りなければ既定、そろっていれば点数の高いほう(同点は既定の優先順)を返す。"""
    if n_confirmed_pages < MIN_CONFIRMED_PAGES or not scores:
        return default_variant(root)
    order = {v: i for i, v in enumerate(VARIANTS)}
    return max(scores, key=lambda s: (s.score, -order.get(s.name, 99))).name
