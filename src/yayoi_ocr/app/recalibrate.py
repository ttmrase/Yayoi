"""追加学習した世代の確信度の再較正(純 numpy。torch なしで試験できる)。設計は `docs/phase4_design.md`。

出発点は**全体の較正**(出荷時の `confidence.json`。新しい重みでも同じ特徴量・同じロジスティック回帰を使う)。評価用ページ(学習に使わなかった確定ページ)での結果を使って、
列ごとの要確認のしきい値 τ(p < τ を要確認)を次のように調整する:

- **厳しくする(τ を上げる)**: 観測した見逃し(間違っているのに要確認が付かないセル)が、その列の目標(日付・科目・摘要 0.5%・金額 0.1% に、出荷時の余裕係数)の許容数を超えたときだけ。
  許容数以下になる最小の τ に上げる(経験的な見逃しで決める。CP は使わない)。許容数まで下げられないほど見逃しが多いときは、要確認だらけにしない(据え置き。採否のゲートが見逃しの悪化を見る)。
- **緩める(τ を下げる)**: Clopper-Pearson の 95% 上側限界が目標以下であると示せたときだけ。評価ページが少ないと(金額の 0.1% は約 3,000 セル、0.5% は約 600 セル)示せないので、
  事実上、評価ページが十分たまるまでは緩めない。
- 見覚えメモリ・履歴(金額・摘要)の特徴量は、全体の較正では重みを 0 にしてある(`confidence.LEARNED_FEATURES`)。この再較正でも、緩める判断と同じ証明ができない標本では
  重みを持たせない(0 のまま)。

評価用ページを採否にも再較正にも使うので、較正は**厳しくする方向を主**にして、採否の評価が楽観に偏らないようにしている(限界は設計文書に記す)。
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from ..recognize.confidence import ConfidenceModel, cp_upper, group_of

# 列ごとの見逃し率の目標(`confidence.fit_confidence` の既定と同じ。金額は 0.1%、その他は 0.5%)
TARGETS = {"date": 0.005, "account": 0.005, "description": 0.005, "amount": 0.001}
ALPHA = 0.05
EPS = 1e-6


@dataclass(frozen=True)
class CellObs:
    """評価用ページの 1 セルの観測(確定値は含まない)。"""

    column: str
    kind: str
    p: float  # 全体の較正での確信度
    correct: bool
    hard: bool  # 確認が必須の理由が付いている(しきい値に関係なく要確認)


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
    action: str  # keep / tighten / loosen / unreachable
    cp_upper: float | None = None  # 緩める判断で使った上側限界(緩めたとき)

    def to_dict(self) -> dict:
        return asdict(self)


def _tunable(base: ConfidenceModel, column: str) -> list[str]:
    return [gk for gk, g in base.groups.items() if gk.endswith(f":{column}") and g.lr is not None]


def _misses(
    obs: Sequence[CellObs], base: ConfidenceModel, column: str, tau: float
) -> tuple[int, int]:
    """(固定の群での見逃し, 調整できる群での見逃し)。見逃し = 間違い ∧ 要確認の理由なし ∧ p ≥ しきい値。"""
    tun = set(_tunable(base, column))
    fixed = tuned = 0
    for o in obs:
        if o.column != column or o.hard or o.correct:
            continue
        gk = group_of(o.column, o.kind)
        if gk in tun:
            tuned += int(o.p >= tau)
        else:
            g = base.groups.get(gk)
            thr = base.default_threshold if g is None else g.threshold
            fixed += int(o.p >= thr)
    return fixed, tuned


def recalibrate(
    base: ConfidenceModel,
    obs: Sequence[CellObs],
    n_cells: dict[str, int],
    targets: dict[str, float] | None = None,
    alpha: float = ALPHA,
) -> tuple[ConfidenceModel, list[ColumnCalib]]:
    """`base`(全体の較正)を評価用ページの観測で調整した新しいモデルと、列ごとの結果を返す。`base` は変更しない。"""
    targets = {**TARGETS, **(targets or {})}
    model = copy.deepcopy(base)
    out: list[ColumnCalib] = []
    for col in sorted({gk.split(":")[1] for gk in base.groups}):
        tun = _tunable(base, col)
        if not tun:
            continue
        n_col = max(1, n_cells.get(col, 0))
        target = targets.get(col, 0.005)
        allowed = math.floor(target * n_col + 1e-9)
        tau0 = max(base.groups[gk].threshold for gk in tun)
        fixed0, tuned0 = _misses(obs, base, col, tau0)
        k0 = fixed0 + tuned0
        tau1, action, cpu = tau0, "keep", None
        if k0 > allowed:
            # 厳しくする: 許容数まで減らすのに必要な件数ぶん、p の低い見逃しから順に要確認に回す(τ を p の低い見逃しの直上まで上げる = 要確認の増加が最小)
            need = k0 - allowed
            wrong_p = sorted(
                o.p
                for o in obs
                if o.column == col
                and not o.hard
                and not o.correct
                and group_of(o.column, o.kind) in tun
                and o.p >= tau0
            )
            if need <= len(wrong_p):
                tau1, action = min(1.1, wrong_p[need - 1] + EPS), "tighten"
            else:
                action = "unreachable"  # 固定の群の見逃しが多すぎて、しきい値では許容数に届かない
        else:
            # 緩める: 上側限界が目標以下と示せる最も低い τ(示せなければ据え置き)
            cand = sorted(
                {
                    o.p
                    for o in obs
                    if o.column == col and group_of(o.column, o.kind) in tun and o.p < tau0
                }
            )
            for tau in cand:  # 低い τ から試す(k は τ について単調)
                f, t = _misses(obs, base, col, tau)
                ub = cp_upper(f + t, n_col, alpha)
                if ub <= target:
                    tau1, action, cpu = tau, "loosen", ub
                    break
        f1, t1 = _misses(obs, base, col, tau1)
        if tau1 != tau0:
            for gk in tun:
                model.groups[gk].threshold = tau1
        out.append(ColumnCalib(col, n_col, target, allowed, tau0, tau1, k0, f1 + t1, action, cpu))
    return model, out


def summarize(calibs: Sequence[ColumnCalib]) -> dict:
    """世代のメタ・精度レポート用。列ごとの件数としきい値の変化だけ(値は含まない)。"""
    by: dict[str, dict] = defaultdict(dict)
    for c in calibs:
        by[c.column] = c.to_dict()
    return {
        "columns": dict(by),
        "changed": [c.column for c in calibs if c.tau_after != c.tau_before],
    }
