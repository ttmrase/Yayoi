"""一般的な日本語の語彙(合成の摘要用)。ドメイン適応で「摘要(漢字・かな)の汎用的な読み取り能力を落とさない」ための
リハーサル用と、その忘却を測る評価用(学習に使わない語)を分ける。

- 取引語(勘定科目・費目・一般的な取引の語): 学習用と評価用で語を分ける(評価用 = 学習に出ない語を含む)。
- 地名・社名風の語: 実在の語と、カタカナ音節から作る架空語。学習用/評価用で名前の集合を分ける。
- 摘要は「名前 + 取引語」「取引語 + n月分」などの型で組み立てる(日本語として自然な長さ・並び)。
内容はすべて架空・一般語で、実データの値は使わない。
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

# 取引語・費目(一般語)。末尾 6 割を学習用、先頭 4 割から評価用の語を作る(下の split で決める)
TERMS = [
    "手数料", "利用料", "購入", "代金", "送料", "配送料", "使用料", "保守料", "点検料", "更新料",
    "年会費", "会費", "登録料", "検査料", "加入料", "分担金", "負担金", "賃借料", "借室料", "駐車料金",
    "電気料金", "ガス料金", "水道料金", "電話料金", "回線料", "郵送料", "印刷代", "コピー代", "製本代",
    "事務用品", "消耗品代", "備品購入", "図書代", "新聞代", "雑誌代", "資料代", "切手代", "収入印紙", "交通費",
    "出張旅費", "宿泊費", "タクシー代", "ガソリン代", "高速料金", "車検費用", "修繕費", "清掃費", "廃棄物処理", "損害保険料",
    "生命保険料", "広告掲載料", "広告制作費", "看板制作費", "会議費", "打合せ費用", "接待飲食代", "慶弔費", "祝金", "香典",
    "研修費", "講習会費", "受講料", "試験手数料", "認証手数料", "審査手数料", "翻訳料", "委託費", "外注加工費", "業務委託料",
    "ソフト使用料", "ライセンス料", "サーバー利用料", "ドメイン更新料", "クラウド利用料", "保管料", "運搬費", "梱包資材", "包装用品", "作業用品",
]  # fmt: skip

# 実在の地名(一般語)。都道府県・主要都市の名前だけ
PLACES = [
    "北海道", "札幌", "仙台", "東京", "横浜", "川崎", "千葉", "埼玉", "新潟", "金沢",
    "長野", "静岡", "浜松", "名古屋", "岐阜", "京都", "大阪", "堺", "神戸", "奈良",
    "和歌山", "岡山", "広島", "山口", "高松", "松山", "高知", "福岡", "北九州", "熊本",
    "長崎", "鹿児島", "那覇", "青森", "盛岡", "秋田", "山形", "福島", "水戸", "宇都宮",
]  # fmt: skip

ORG_SUFFIX = [
    "商事", "工業", "運輸", "通信", "電力", "ガス", "書店", "薬局", "食堂", "文具",
    "印刷", "設計", "建設", "不動産", "保険", "サービス", "システム", "センター", "協会", "組合",
]  # fmt: skip

# 架空語を作るカタカナ音節
SYLLABLES = [
    *"アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワ",
    "ガ", "ギ", "グ", "ゲ", "ゴ", "ザ", "ジ", "ズ", "ゼ", "ゾ", "ダ", "デ", "ド", "バ", "ビ", "ブ", "ベ", "ボ",
    "パ", "ピ", "プ", "ペ", "ポ", "ン", "ー",
]  # fmt: skip


def _bucket(s: str, mod: int = 10) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % mod


def in_split(word: str, split: str) -> bool:
    """語が学習用("train")か評価用("eval")か(ハッシュで決定。学習用 8 割・評価用 2 割)。"""
    return (_bucket(word, 10) < 8) == (split == "train")


def _fictional_name(rng: np.random.Generator, split: str) -> str:
    for _ in range(100):
        n = int(rng.integers(3, 6))
        w = (
            "".join(str(rng.choice(SYLLABLES)) for _ in range(n))
            .replace("ーー", "ー")
            .lstrip("ンー")
        )
        if len(w) >= 3 and in_split(w, split):
            return w
    return "サンプル"


def general_description(rng: np.random.Generator, split: str) -> str:
    """一般的な摘要を 1 つ作る。split は "train" か "eval"(名前・取引語を分ける)。"""
    terms = [t for t in TERMS if in_split(t, split)] or TERMS
    places = [p for p in PLACES if in_split(p, split)] or PLACES
    term = str(rng.choice(terms))
    kind = int(rng.integers(0, 6))
    if kind == 0:
        return term
    if kind == 1:
        return f"{_fictional_name(rng, split)} {term}"
    if kind == 2:
        return f"{rng.choice(places)}{rng.choice(ORG_SUFFIX)} {term}"
    if kind == 3:
        return f"{term} {int(rng.integers(1, 13))}月分"
    if kind == 4:
        return f"{_fictional_name(rng, split)}{rng.choice(ORG_SUFFIX)} 請求分"
    return f"({rng.choice(['株', '有'])}){_fictional_name(rng, split)} {term}"


def general_pool(split: str, n: int = 400, seed: int = 0) -> list[str]:
    """摘要の候補集合(重複なし)。月は {m} で置き換えられる形("n月分" は固定の数字を避けて {m} にする)。"""
    rng = np.random.default_rng(seed + (0 if split == "train" else 1))
    out: list[str] = []
    seen: set[str] = set()
    for _ in range(n * 6):
        d = general_description(rng, split)
        d = re.sub(r"\d{1,2}月分", "{m}月分", d)
        if d not in seen:
            seen.add(d)
            out.append(d)
        if len(out) >= n:
            break
    return out
