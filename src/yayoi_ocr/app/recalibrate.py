"""追加学習した世代の確信度の再較正(純 numpy。torch なしで試験できる)。設計は `docs/phase4_design.md`。

出発点は**全体の較正**(出荷時の `confidence.json`。新しい重みでも同じ特徴量・同じロジスティック回帰を使う)。評価用ページ(学習に使わなかった確定ページ)での結果を使って、
列ごとの要確認のしきい値 τ(p < τ を要確認)を次のように調整する:

- **厳しくする(τ を上げる)**: 観測した見逃し(間違っているのに要確認が付かないセル)が、その列の目標(日付・科目・摘要 0.5%・金額 0.1%)の許容数を超えたときだけ。
  許容数以下になる最小の τ に上げる(経験的な見逃しで決める。CP は使わない)。許容数まで下げられないほど見逃しが多いときは、要確認だらけにしない(据え置き。採否のゲートが見逃しの悪化を見る)。
- **緩める(τ を下げる)**: Clopper-Pearson の 95% 上側限界が目標以下であると示せたときだけ。評価ページが少ないと(金額の 0.1% は約 3,000 セル、0.5% は約 600 セル)示せないので、
  事実上、評価ページが十分たまるまでは緩めない。
- **床**: `floor`(親の世代の確信度)を与えると、どの群のしきい値も親のものより緩めない(確定ページが 20 未満の間、`train_worker` がこれを与える)。
- **ウォームアップ**: 採用した直後の数ページには、通常のしきい値より厳しい `warm_threshold`(確信度から予測される見逃し率が目標 × 0.8 以下になる最小のしきい値。観測した誤りではなく
  確信度 p から見積もるので、少数の評価ページでも安定する)を使う。
- 見覚えメモリ・履歴(金額・摘要)の特徴量は、全体の較正では重みを 0 にしてある(`confidence.LEARNED_FEATURES`)。この再較正でも、緩める判断と同じ証明ができない標本では重みを持たせない(0 のまま)。

評価用ページを採否にも再較正にも使うので、**クロスフィッティング**(`cross_fit`)で採否の見逃しを数える: 各評価ページの判定に使うしきい値は、そのページを除いた評価ページから決める。
採用する世代が実際に使うしきい値は、全評価ページで決めたもの。
"""

from __future__ import annotations

import copy
import math
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache

from ..recognize.confidence import ConfidenceModel, cp_upper, group_of
from .evalstats import CELLS_PER_PAGE_COLUMN, CellObs, apply_floor, review_flags

# 列ごとの見逃し率の目標(`confidence.fit_confidence` の既定と同じ。金額は 0.1%、その他は 0.5%)
TARGETS = {"date": 0.005, "account": 0.005, "description": 0.005, "amount": 0.001}
ALPHA = 0.05
EPS = 1e-6
WARM_FACTOR = 0.8  # 採用直後の数ページのしきい値: 予測される見逃し率が目標のこの倍率以下
WARM_PAGES = 5  # 採用してから、この数のページを認識するまでがウォームアップ
FLOOR_PAGES = 20  # 確定ページがこれ未満の間は、しきい値を親の世代より緩めない


@dataclass
class ColumnCalib:
    column: str
    n_cells: int  # 列の全セル数(見逃し率の分母。空欄を含む)
    target: float
    allowed: int  # 目標から決まる見逃しの許容数
    tau_before: float
    tau_after: float
    k_before: int  # 全体の較正での観測した見逃し
    k_after: int
    action: str  # keep / tighten / loosen / unreachable / floor
    cp_upper: float | None = None  # 緩める判断で使った上側限界(緩めたとき)

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=4096)
def _cp_upper(k: int, n: int, alpha: float) -> float:
    return cp_upper(k, n, alpha)


def _tunable(base: ConfidenceModel, column: str) -> list[str]:
    return [gk for gk, g in base.groups.items() if gk.endswith(f":{column}") and g.lr is not None]


def _floor_tau(floor: ConfidenceModel | None, tun: Sequence[str]) -> float | None:
    if floor is None:
        return None
    ts = [floor.groups[gk].threshold for gk in tun if gk in floor.groups]
    return max(ts) if ts else None


def recalibrate(
    base: ConfidenceModel,
    obs: Sequence[CellObs],
    n_cells: dict[str, int],
    targets: dict[str, float] | None = None,
    alpha: float = ALPHA,
    floor: ConfidenceModel | None = None,
) -> tuple[ConfidenceModel, list[ColumnCalib]]:
    """`base`(全体の較正)を評価用ページの観測で調整した新しいモデルと、列ごとの結果を返す。`base` は変更しない。
    見逃しの数え方は、確信度とそのしきい値だけで決める近似(同上の参照元による要確認は無視 = 見逃しをやや多めに数える = 安全側)。"""
    targets = {**TARGETS, **(targets or {})}
    model = copy.deepcopy(base)
    out: list[ColumnCalib] = []
    for col in sorted({gk.split(":")[1] for gk in base.groups}):
        tun = _tunable(base, col)
        if not tun:
            continue
        tun_set = set(tun)
        n_col = max(1, n_cells.get(col, 0))
        target = targets.get(col, 0.005)
        allowed = math.floor(target * n_col + 1e-9)
        tau0 = max(base.groups[gk].threshold for gk in tun)
        wrong_t: list[float] = []  # 調整できる群の見逃し候補(間違い ∧ 確認必須でない)の確信度
        free_t: list[float] = []  # 調整できる群の確認必須でないセルの確信度(緩める候補)
        k_fixed = 0  # 固定の群の見逃し
        for o in obs:
            if o.column != col or o.hard:
                continue
            gk = group_of(o.column, o.kind)
            if gk in tun_set:
                free_t.append(o.p)
                if not o.correct:
                    wrong_t.append(o.p)
            elif not o.correct:
                g = base.groups.get(gk)
                thr = base.default_threshold if g is None else g.threshold
                k_fixed += int(o.p >= thr)
        wrong_t.sort()
        free_t.sort()

        def k_of(tau: float, wrong_t=wrong_t, k_fixed=k_fixed) -> int:
            return k_fixed + len(wrong_t) - bisect_left(wrong_t, tau)

        k0 = k_of(tau0)
        tau1, action, cpu = tau0, "keep", None
        if k0 > allowed:
            need = k0 - allowed  # p の低い見逃しから順に要確認に回す = 要確認の増加が最小
            cand = wrong_t[bisect_left(wrong_t, tau0) :]
            if need <= len(cand):
                tau1, action = min(1.1, cand[need - 1] + EPS), "tighten"
            else:
                action = "unreachable"  # 固定の群の見逃しが多すぎて、しきい値では許容数に届かない
        else:
            for tau in free_t[: bisect_left(free_t, tau0)]:  # 低い τ から試す(k は τ について単調)
                ub = _cp_upper(k_of(tau), n_col, alpha)
                if ub <= target:
                    tau1, action, cpu = tau, "loosen", ub
                    break
        ft = _floor_tau(floor, tun)
        if ft is not None and tau1 < ft:
            tau1, action, cpu = ft, "floor", None
        if tau1 != tau0:
            for gk in tun:
                model.groups[gk].threshold = tau1
        out.append(
            ColumnCalib(col, n_col, target, allowed, tau0, tau1, k0, k_of(tau1), action, cpu)
        )
    return model, out


def warm_thresholds(
    model: ConfidenceModel,
    obs: Sequence[CellObs],
    n_cells: dict[str, int],
    targets: dict[str, float] | None = None,
    factor: float = WARM_FACTOR,
) -> ConfidenceModel:
    """採用直後の数ページ用のしきい値(`warm_threshold`)を付けた新しいモデル。列ごとに、確信度から予測される見逃し率(要確認でないセルの (1 - p) の合計 ÷ 列の全セル数)が
    目標 × `factor` 以下になる最小のしきい値(通常のしきい値以上)。固定の群だけで予測が超えるときは付けない。"""
    targets = {**TARGETS, **(targets or {})}
    out = copy.deepcopy(model)
    for col in sorted({gk.split(":")[1] for gk in model.groups}):
        tun = _tunable(model, col)
        if not tun:
            continue
        tun_set = set(tun)
        n_col = max(1, n_cells.get(col, 0))
        budget = factor * targets.get(col, 0.005) * n_col
        tau_now = max(model.groups[gk].threshold for gk in tun)
        fixed_e = 0.0
        ps: list[float] = []
        for o in obs:
            if o.column != col or o.hard:
                continue
            gk = group_of(o.column, o.kind)
            if gk in tun_set:
                ps.append(min(1.0, max(0.0, o.p)))
            else:
                g = model.groups.get(gk)
                thr = model.default_threshold if g is None else g.threshold
                if o.p >= thr:
                    fixed_e += 1.0 - min(1.0, max(0.0, o.p))
        ps.sort()
        suffix = [0.0] * (len(ps) + 1)  # suffix[i] = Σ_{j ≥ i} (1 - p_j)
        for i in range(len(ps) - 1, -1, -1):
            suffix[i] = suffix[i + 1] + (1.0 - ps[i])

        def expected(tau: float, ps=ps, suffix=suffix, fixed_e=fixed_e) -> float:
            return fixed_e + suffix[bisect_left(ps, tau)]

        warm: float | None = None
        for tau in [tau_now, *[p for p in ps if p > tau_now], 1.1]:
            if expected(tau) <= budget:
                warm = tau if tau == tau_now else min(1.1, tau + EPS)
                break
        if warm is not None:
            for gk in tun:
                out.groups[gk].warm_threshold = warm
    return out


def cross_fit(
    base: ConfidenceModel,
    pages: Sequence[Sequence[CellObs]],
    *,
    floor: ConfidenceModel | None = None,
    targets: dict[str, float] | None = None,
    cells_per_column: int = CELLS_PER_PAGE_COLUMN,
) -> tuple[ConfidenceModel, list[ColumnCalib], list[list[bool]]]:
    """(採用した場合に使う確信度モデル, 列ごとの結果, 評価ページごとの要確認の判定)。
    判定は、そのページを**除いた**評価ページ(と全体の較正 `base`、`floor`)から決めたしきい値で作る(クロスフィッティング)。
    採用した場合に使うモデルは、全評価ページで決めたもの。評価ページが 1 ページしかなければ、除くと観測が無いので `base`(と `floor`)のしきい値を使う。"""
    n_pages = len(pages)
    all_obs = [o for pg in pages for o in pg]
    final, calibs = recalibrate(
        base, all_obs, {c: n_pages * cells_per_column for c in TARGETS}, targets, floor=floor
    )
    flags: list[list[bool]] = []
    for i, pg in enumerate(pages):
        if n_pages == 1:
            model_i = apply_floor(base, floor)
        else:
            others = [o for j, p in enumerate(pages) if j != i for o in p]
            model_i, _ = recalibrate(
                base,
                others,
                {c: (n_pages - 1) * cells_per_column for c in TARGETS},
                targets,
                floor=floor,
            )
        flags.append(review_flags(pg, model_i))
    return final, calibs, flags


def summarize(calibs: Sequence[ColumnCalib]) -> dict:
    """世代のメタ・精度レポート用。列ごとの件数としきい値の変化だけ(値は含まない)。"""
    by: dict[str, dict] = defaultdict(dict)
    for c in calibs:
        by[c.column] = c.to_dict()
    return {
        "columns": dict(by),
        "changed": [c.column for c in calibs if c.tau_after != c.tau_before],
    }
