"""確信度の統合(2.5): セルが正しい確率をロジスティック回帰で推定し、要確認のしきい値を決める。

- 特徴量: R1 の系列確率・文字ごとの確信度、文法・語彙・辞書への適合(top_p, mass, post_top, post_gap, vocab_fit …)、
  セル種別分類器の確率、レイアウトの状況(薄いインク・はみ出し・インクの高さ比)、同上の参照元の確信度。
- 群(列 x 種別)ごとにロジスティック回帰を学習する。確認が必須の理由コード(文法違反・年の不明など)が付いたセルは、
  最初から確認なので学習・しきい値の対象外。
- しきい値: 「見逃し率 ≦ 目標」を満たす範囲で要確認率が最小になるように、群ごとに決める
  (見逃し = 間違っているのに要確認が付かないセル。分母は列の全セル。目標は金額 0.1%・その他 0.5% 以下)。
  較正データが少ないほど危ういので、見逃しの件数に安全側の余裕(`margin`)を持たせる。
- 純 numpy(IRLS)で学習する。確定したセルが溜まったら同じコードで再較正できる(Phase 4 の世代管理で使う)。
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FEATURE_NAMES = (
    "cls_p",
    "r1_logp",
    "r1_min_char",
    "top_logp",
    "top_p",
    "mass",
    "post_top",
    "post_gap",
    "log_n_cands",
    "direct",
    "explicit_year",
    "vocab_fit",
    "free_agree",
    "from_dict",
    "ink_h_frac",
    "ink_w_frac",
    "faint",
    "overflow",
    "src_conf",
)
# 確認が必須の理由(学習・しきい値の対象外。常に要確認)
HARD_REASONS = frozenset(
    {
        "grammar_violation",
        "no_valid_candidate",
        "oov_account",
        "ditto_first_row",
        "arrow_candidate",
        "wave_candidate",
        "year_unknown",
        "month_decreased",
        "year_jump",
        "row_inconsistent",
        "cp932_unencodable",
        "too_long",
        "class_uncertain",
        "layout_uncertain",
        "page_needs_review",
        "ditto_gap",
        "ditto_source_uncertain",
        "annotation_present",
    }
)
ALWAYS_REVIEW_KINDS = ("arrow", "wave")


def group_of(column: str, kind: str) -> str:
    return f"{kind}:{column}"


def featurize(features: dict[str, float], cls_p: float) -> np.ndarray:
    """特徴量の辞書 → ベクトル(欠けている特徴は 0)。"""
    f = dict(features)
    f["cls_p"] = cls_p
    f["log_n_cands"] = math.log(max(f.get("n_cands", 1.0), 1.0))
    # 対数尤度は [-50, 0] に収め、スケールをそろえる(/10)
    for k in ("r1_logp", "top_logp"):
        f[k] = max(min(f.get(k, -50.0), 0.0), -50.0) / 10.0
    f["vocab_fit"] = max(min(f.get("vocab_fit", 0.0), 5.0), -20.0) / 5.0
    return np.array([float(f.get(n, 0.0)) for n in FEATURE_NAMES], dtype=np.float64)


class LogisticRegression:
    """L2 正則化つきロジスティック回帰(IRLS)。標準化して学習し、重みを元の尺度へ戻して保持する。"""

    def __init__(self, l2: float = 1.0) -> None:
        self.l2 = l2
        self.w = np.zeros(len(FEATURE_NAMES))
        self.b = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray, iters: int = 50) -> LogisticRegression:
        mu = x.mean(0)
        sd = x.std(0)
        sd[sd < 1e-8] = 1.0
        z = (x - mu) / sd
        n, d = z.shape
        beta = np.zeros(d + 1)
        zb = np.hstack([z, np.ones((n, 1))])
        reg = np.eye(d + 1) * self.l2
        reg[-1, -1] = 0.0
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(zb @ beta, -30, 30)))
            g = zb.T @ (p - y) + reg @ beta
            h = (zb * (p * (1 - p))[:, None]).T @ zb + reg + 1e-6 * np.eye(d + 1)
            step = np.linalg.solve(h, g)
            beta -= step
            if np.abs(step).max() < 1e-7:
                break
        self.w = beta[:-1] / sd
        self.b = float(beta[-1] - (beta[:-1] * mu / sd).sum())
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x @ self.w + self.b, -30, 30)))


@dataclass
class GroupModel:
    lr: LogisticRegression | None  # None = 定数(学習データが無い・全部正解/全部不正解)
    const: float = 0.5
    threshold: float = 1.1  # p < threshold を要確認。1.1 = 常に確認
    n_train: int = 0

    def prob(self, x: np.ndarray) -> float:
        if self.lr is None:
            return self.const
        return float(self.lr.predict_proba(x[None, :])[0])


@dataclass
class ConfidenceModel:
    """群ごとの P(正しい) としきい値。PageRecognizer の `calibrator` として使う(呼び出し可能)。"""

    groups: dict[str, GroupModel] = field(default_factory=dict)
    default_threshold: float = 0.9

    def __call__(self, r) -> float:  # r: page.CellResult
        g = self.groups.get(group_of(r.column, r.kind))
        if r.kind in ALWAYS_REVIEW_KINDS:
            return 0.0
        if g is None:
            return r.cls_probs.get(r.kind, 1.0) * r.features.get("post_top", 1.0)
        return g.prob(featurize(r.features, r.cls_probs.get(r.kind, 1.0)))

    def threshold(self, r) -> float:
        if r.kind in ALWAYS_REVIEW_KINDS:
            return 1.1
        g = self.groups.get(group_of(r.column, r.kind))
        return self.default_threshold if g is None else g.threshold

    # ---- 保存・読み込み(JSON。重みは小さい)
    def to_json(self) -> str:
        out = {}
        for k, g in self.groups.items():
            out[k] = {
                "w": None if g.lr is None else g.lr.w.tolist(),
                "b": None if g.lr is None else g.lr.b,
                "const": g.const,
                "threshold": g.threshold,
                "n_train": g.n_train,
            }
        return json.dumps(
            {"features": list(FEATURE_NAMES), "groups": out, "default": self.default_threshold},
            ensure_ascii=False,
            indent=1,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ConfidenceModel:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d["features"] != list(FEATURE_NAMES):
            raise ValueError("特徴量の構成が違う確信度モデルです(再較正が必要)")
        m = cls(default_threshold=d.get("default", 0.9))
        for k, g in d["groups"].items():
            lr = None
            if g["w"] is not None:
                lr = LogisticRegression()
                lr.w = np.array(g["w"])
                lr.b = float(g["b"])
            m.groups[k] = GroupModel(lr, g["const"], g["threshold"], g["n_train"])
        return m


@dataclass
class Sample:
    """較正用の 1 セル: 特徴量・正誤・確認必須か。"""

    column: str
    kind: str
    x: np.ndarray
    correct: bool
    hard: bool  # 確認が必須の理由が付いている(しきい値の対象外)


def samples_from_dump(rows: Iterable[dict]) -> list[Sample]:
    """eval_recognize.py の --dump(JSON Lines の各行)→ Sample。"""
    out = []
    for r in rows:
        kind = r["pred_kind"]
        hard = bool(HARD_REASONS & set(r["reasons"])) or bool(r.get("page_review"))
        cls_p = r["cls_probs"].get(kind, 1.0)
        out.append(
            Sample(
                r["column"],
                kind,
                featurize(r["features"], cls_p),
                bool(r["correct"]),
                hard or kind in ALWAYS_REVIEW_KINDS,
            )
        )
    return out


def choose_threshold(
    p: np.ndarray,
    correct: np.ndarray,
    n_column_cells: int,
    max_miss_rate: float,
    margin: float = 1.0,
) -> float:
    """p < τ を要確認としたとき、(間違い ∧ p ≥ τ)の件数 ≦ 目標 x 列の全セル数 / margin を満たす最小の τ。"""
    allowed = max_miss_rate * n_column_cells / margin
    order = np.argsort(-p)  # 確信度の高い順に「確認なし」にしていく
    wrong_cum = np.cumsum(~correct[order])
    # k 件を確認なしにしたときの見逃し = wrong_cum[k-1]。許される最大の k を探す
    ok = np.nonzero(wrong_cum <= allowed)[0]
    k = 0 if len(ok) == 0 else int(ok.max()) + 1
    if k == 0:
        return 1.1
    if k >= len(p):
        return 0.0
    # k 件目と k+1 件目の確信度の間にしきい値を置く(同点は確認側に倒す)
    return (
        float(0.5 * (p[order][k - 1] + p[order][k]))
        if p[order][k - 1] > p[order][k]
        else float(p[order][k - 1] + 1e-9)
    )


def fit_confidence(
    samples: Sequence[Sample],
    n_column_cells: dict[str, int],
    max_miss: dict[str, float] | None = None,
    default_max_miss: float = 0.005,
    margin: float = 1.0,
    l2: float = 3.0,
) -> ConfidenceModel:
    """群ごとにロジスティック回帰を学習し、列ごとの見逃し目標から群のしきい値を決める。

    n_column_cells: 列 → 較正データでの全セル数(空欄を含む)。見逃し率の分母。
    max_miss: 列 → 見逃し率の上限(既定 0.5%、金額は 0.1%)。同じ列の群には目標を群の寄与に応じて配分する。
    """
    max_miss = {"amount": 0.001, **(max_miss or {})}
    by_group: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_group[group_of(s.column, s.kind)].append(s)
    model = ConfidenceModel()
    for gk, ss in by_group.items():
        kind = gk.split(":")[0]
        free = [s for s in ss if not s.hard]
        if kind in ALWAYS_REVIEW_KINDS or not free:
            model.groups[gk] = GroupModel(None, 0.0, 1.1, len(ss))
            continue
        x = np.stack([s.x for s in free])
        y = np.array([s.correct for s in free], dtype=np.float64)
        if y.min() == y.max() or len(y) < 20:
            const = float(y.mean())
            model.groups[gk] = GroupModel(None, const, 0.5 if const > 0.5 else 1.1, len(free))
            continue
        lr = LogisticRegression(l2).fit(x, y)
        model.groups[gk] = GroupModel(lr, 0.5, 0.9, len(free))
    # しきい値: 列ごとに、その列の全群(text・ditto・blank)をまとめて、目標の見逃し率を満たす最小の τ を決める。
    # 各群のロジスティック回帰は P(正しい) を表すので、列の中で共通のしきい値でよい。
    for col in {gk.split(":")[1] for gk in model.groups}:
        pooled = [
            (model.groups[gk].prob(s.x), s.correct)
            for gk in model.groups
            if gk.endswith(f":{col}") and model.groups[gk].lr is not None
            for s in by_group[gk]
            if not s.hard
        ]
        if not pooled:
            continue
        p = np.array([t[0] for t in pooled])
        c = np.array([t[1] for t in pooled])
        target = max_miss.get(col, default_max_miss)
        tau = choose_threshold(p, c, max(1, n_column_cells.get(col, 1)), target, margin)
        for gk in model.groups:
            if gk.endswith(f":{col}") and model.groups[gk].lr is not None:
                model.groups[gk].threshold = tau
    return model


Calibrator = Callable[[object], float]
