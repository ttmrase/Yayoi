"""合成帳票の内容: 架空の取引データ(論理データ)→ 書かれ方(〃・矢印・波線による省略)→ 正解 JSON。

内容はすべて架空(一般的な勘定科目名・費目の語彙と、乱数で作った日付・金額)。実データの値は使わない。
論理データ(resolved)を先に決め、そこから「どう書くか」(raw)を決めるので、正解 JSON が必ず整合する。

正解 JSON(docs/ground_truth_schema.json):
- raw: 書かれた通りの文字列。同上記号は「〃」、矢印は「→」、波線による省略は「頭〜尾」。
  日付・金額は描いた文字列そのまま(例: 「R7 11.19」「3,456-」)。
- resolved: 展開後の値。日付は R<年>.<月>.<日>、金額は数字のみ、〃・→・〜は元の記入に展開。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..groundtruth import GTField, GTPage, GTRow

DITTO = "〃"
ARROW = "→"
ELIDE = "〜"

TITLES = [
    "消耗品費",
    "旅費交通費",
    "水道光熱費",
    "地代家賃",
    "租税公課",
    "保険料",
    "会議費",
    "事務用品費",
    "雑費",
    "広告宣伝費",
    "車両費",
    "荷造運賃",
    "修繕費",
    "福利厚生費",
    "接待交際費",
]  # fmt: skip  (一般的な勘定科目名)
ACCOUNTS = [
    "現金", "テスト銀行", "サンプル信金", "サンプルカード", "ダミー銀行",
    "仮払金", "普通預金", "売掛金", "未払金",
]  # fmt: skip
# 費目の語彙(一般語)。{m} は月に置き換える。
DESCRIPTIONS = [
    "切手代", "電話代 {m}月分", "(株)サンプル 手数料", "ネット回線 利用料", "消耗品代", "事務用品",
    "配送料", "会議費", "サンプル電力 {m}月分", "ダミー運送 配送料", "テスト保険料",
    "試験用 利用料",
    "ABC)SAMPLE", "ダミー郵便 切手代", "サンプルガス {m}月分", "水道料金 {m}月分", "コピー用紙",
    "駐車場代", "交通費", "(株)テスト商会 請求分", "テスト通信 電話代 {m}月分", "サンプル便 送料",
    "試験用品 購入", "デモ書店 図書代",
]  # fmt: skip
# 波線で省略して書く長い記入(頭, 全体, 尾)。実データの語や構造を写さない、架空の語。
LONG_DESCRIPTIONS = [
    ("サンプル協会", "サンプル協会 定期点検 立会い 作業 手数料", "手数料"),
    ("テスト機構", "テスト機構 検査記録 閲覧 取扱手数料", "取扱手数料"),
    ("ダミー連合会", "ダミー連合会 登録 照会 代行 手数料", "手数料"),
]
ANNOTATIONS = ["(新)", "3月", "分", "サンプル", "(旧)", "再"]


@dataclass
class CellPlan:
    column: str  # date / account / description / amount
    kind: str  # blank / text / ditto / arrow / elided
    raw: str
    resolved: str
    head: str = ""  # elided: 頭
    tail: str = ""  # elided: 尾
    annotation: str = ""  # セル上部の小さな注記
    overflow: bool = False  # セルからはみ出して書く
    stray: bool = False  # 余計な点・線
    note: str = ""


@dataclass
class PagePlan:
    title: str
    number: int
    rows: list[dict[str, CellPlan]] = field(default_factory=list)

    def raw_text(self, r: int, col: str) -> str:
        return self.rows[r][col].raw

    def to_ground_truth(self, image_name: str = "synthetic.png") -> GTPage:
        rows = []
        for r in self.rows:
            fields = {}
            for col in ("date", "account", "description", "amount"):
                c = r[col]
                note = c.note
                if c.annotation:
                    note = (note + " " if note else "") + f"上部注記あり({c.annotation})"
                fields[col] = GTField(raw=c.raw, resolved=c.resolved, uncertain=False, note=note)
            rows.append(GTRow(**fields))
        return GTPage(
            schema_version=1,
            image=image_name,
            title=GTField(raw=self.title, resolved=self.title),
            number=GTField(raw=f"No.{self.number}", resolved=str(self.number)),
            rows=rows,
        )


def _random_amount(rng: np.random.Generator) -> int:
    """100〜99,999 の対数一様分布から取り、1・10・100 円単位にランダムに丸める(固定リストなし)。"""
    v = 10 ** float(rng.uniform(2.0, 5.0))
    unit = int(rng.choice([1, 10, 100], p=[0.3, 0.4, 0.3]))
    return int(min(99_999, max(100, round(v / unit) * unit)))


def _fmt_amount(rng: np.random.Generator, v: int) -> str:
    s = f"{v:,}" if rng.random() < 0.88 else str(v)
    return s + ("-" if rng.random() < 0.95 else "")


def plan_page(rng: np.random.Generator, n_rows: int = 29, fill_rows: int | None = None) -> PagePlan:
    """1 ページ分の内容を作る。`fill_rows` 行が埋まり、残りは空行(途中の空行も少し混ぜる)。"""
    if fill_rows is None:
        fill_rows = int(rng.integers(10, n_rows + 1))
    plan = PagePlan(title=str(rng.choice(TITLES)), number=int(rng.integers(1, 30)))
    era = int(rng.integers(6, 9))
    month, day = int(rng.integers(1, 13)), int(rng.integers(1, 29))
    prev: dict[str, str] = {"date": "", "account": "", "description": "", "amount": ""}
    desc_history: list[str] = []
    acct = str(rng.choice(ACCOUNTS))
    desc = ""
    amount = 0
    for i in range(n_rows):
        if i >= fill_rows or (i > 0 and rng.random() < 0.02):
            blank = {
                c: CellPlan(c, "blank", "", "")
                for c in ("date", "account", "description", "amount")
            }
            plan.rows.append(blank)
            continue
        row: dict[str, CellPlan] = {}
        # ---- 日付
        year_changed = False
        same_date = i > 0 and prev["date"] != "" and rng.random() < 0.28
        if i > 0 and not same_date:
            if rng.random() < 0.08:  # 少し前後する
                day = int(np.clip(day + int(rng.integers(-12, 5)), 1, 28))
            else:
                day += int(rng.integers(1, 21))
            while day > 28:
                day -= 28
                month += 1
            if month > 12:
                month -= 12
                era += 1
                year_changed = True
        resolved_date = f"R{era}.{month}.{day}"
        explicit_era = i == 0 or year_changed or rng.random() < 0.04
        written = f"R{era} {month}.{day}" if explicit_era else f"{month}.{day}"
        if same_date and prev["date"] == resolved_date and rng.random() < 0.8:
            row["date"] = CellPlan("date", "ditto", DITTO, resolved_date)
        else:
            row["date"] = CellPlan("date", "text", written, resolved_date)
        prev["date"] = resolved_date
        # ---- 科目(同じ科目が続きやすい)
        if i == 0 or rng.random() > 0.7:
            acct = str(rng.choice(ACCOUNTS))
        if i > 0 and acct == prev["account"] and rng.random() < 0.88:
            row["account"] = CellPlan("account", "ditto", DITTO, acct)
        else:
            ann = str(rng.choice(ANNOTATIONS)) if rng.random() < 0.05 else ""
            row["account"] = CellPlan("account", "text", acct, acct, annotation=ann)
        prev["account"] = acct
        # ---- 摘要(同上・矢印・波線による省略・空欄)
        u = rng.random()
        if i > 0 and desc and u < 0.30:
            new_desc = desc
        else:
            new_desc = str(rng.choice(DESCRIPTIONS)).replace("{m}", str(int(rng.integers(1, 13))))
        if i > 0 and desc and new_desc == desc and rng.random() < 0.9:
            row["description"] = CellPlan("description", "ditto", DITTO, desc)
        elif desc_history and u > 0.95:
            target = desc_history[-int(rng.integers(1, min(12, len(desc_history)) + 1))]
            row["description"] = CellPlan(
                "description", "arrow", ARROW, target, note="矢印は上の記入を指す"
            )
            new_desc = target
        elif u > 0.90:
            head, full, tail = LONG_DESCRIPTIONS[int(rng.integers(len(LONG_DESCRIPTIONS)))]
            row["description"] = CellPlan(
                "description", "elided", f"{head}{ELIDE}{tail}", full, head=head, tail=tail,
                note="波線で省略",
            )  # fmt: skip
            new_desc = full
        elif u > 0.87:
            row["description"] = CellPlan("description", "blank", "", "")
            new_desc = ""
        else:
            row["description"] = CellPlan(
                "description", "text", new_desc, new_desc, overflow=bool(rng.random() < 0.05),
            )  # fmt: skip
        if new_desc:
            desc = new_desc
            desc_history.append(new_desc)
        # ---- 金額
        if i > 0 and amount and rng.random() < 0.3:
            pass  # 前の行と同じ金額
        else:
            amount = _random_amount(rng)
        raw_amount = _fmt_amount(rng, amount)
        row["amount"] = CellPlan(
            "amount", "text", raw_amount, str(amount), stray=bool(rng.random() < 0.03),
        )  # fmt: skip
        if row["amount"].stray:
            row["amount"].note = "余計な点・線あり"
        plan.rows.append(row)
    return plan
