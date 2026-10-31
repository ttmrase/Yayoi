"""「ライセンス」画面のデータ。主な利用物(出典・ライセンス・変更点)はここに固定で持ち、依存ライブラリの一覧は配布物の作成時に lock から生成する(`scripts/gen_licenses.py` → `data/licenses.json`)。

ネットワークは使わない。ライセンス本文(各ライブラリのもの)は配布物の `licenses/` に入っている。
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).parent / "data" / "licenses.json"

# 主な利用物。帰属表示・変更点は、docs/model_provenance.md・docs/dependencies.md と同じ内容。
MAIN = [
    {
        "name": "NDLOCR-Lite の文字列認識モデル(PARSeq tiny 24x256・24x384)と文字集合 NDLmoji.yaml",
        "license": "CC BY 4.0",
        "owner": "国立国会図書館(National Diet Library, Japan)",
        "source": "https://github.com/ndl-lab/ndlocr-lite(コミット e57f7c6bd9cfb3de5f12f008e71ad2122bd44bfd)",
        "changes": "ONNX 形式の重みを PyTorch のテンソルに形式変換した(値は変更していない)。"
        "適応済みの重み(models/adapted)と、あなたのデータから作る世代は、この重みに追加学習をした派生物。",
        "note": "ライセンス: https://creativecommons.org/licenses/by/4.0/",
    },
    {
        "name": "PARSeq(モデルのコード)",
        "license": "Apache License 2.0",
        "owner": "Copyright 2022 Darwin Bautista (baudm/parseq)",
        "source": "https://github.com/baudm/parseq(コミット 1902db043c029a7e03a3818c616c06600af574be)",
        "changes": "推論に必要な部分だけを取り込み、学習用コードの削除、timm 依存の除去、系列の対数尤度・ビームサーチ・列ごとの文字集合マスクの追加を行った。"
        "ライセンス全文は配布物の licenses/LICENSE-parseq-Apache-2.0.txt。",
        "note": "",
    },
    {
        "name": "Python(CPython 3.11)と python-build-standalone",
        "license": "PSF License(CPython)。同梱の第三者ライブラリはそれぞれのライセンス",
        "owner": "Python Software Foundation / astral-sh",
        "source": "https://github.com/astral-sh/python-build-standalone",
        "changes": "なし(配布物をそのまま同梱。SHA-256 を固定して検証している)。",
        "note": "ライセンス全文は配布物の licenses/python-LICENSE.txt(OpenSSL・SQLite・zlib などの表記を含む)。",
    },
    {
        "name": "合成データの生成に使った手書き風フォント(Yomogi・Zen Kurenaido・Klee One など)",
        "license": "SIL Open Font License 1.1",
        "owner": "各フォントの作者(docs/fonts.md)",
        "source": "https://github.com/google/fonts(ofl/ 配下)",
        "changes": "フォントそのものは配布しない。架空の内容を描いた合成画像から作った学習用の束(models/rehearsal.npz)と、モデルの重みだけを同梱する。",
        "note": "",
    },
    {
        "name": "Microsoft Edge WebView2 ランタイム",
        "license": "Microsoft のライセンス条項(配布物には含まない。Windows 11 には標準で入っている)",
        "owner": "Microsoft",
        "source": "https://developer.microsoft.com/microsoft-edge/webview2/",
        "changes": "なし。",
        "note": "画面の表示にだけ使う。アプリは実行時にネットワークを使わない。",
    },
]


def load() -> dict:
    """主な利用物 + 配布物の作成時に生成した依存ライブラリの一覧(無ければ空。開発中は生成されていない)。"""
    components: list[dict] = []
    generated = False
    try:
        data = json.loads(DATA.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("components"), list):
            components = [c for c in data["components"] if isinstance(c, dict)]
            generated = bool(data.get("generated"))
    except (OSError, ValueError):
        pass
    return {"main": MAIN, "components": components, "generated": generated}
