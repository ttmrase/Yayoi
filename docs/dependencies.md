# 依存ライブラリの記録(出所・必要性・ライセンス)

GPL / AGPL 系は使わない方針。追加・変更時はここに追記する。出所はすべて PyPI。

| ライブラリ | 用途 | ライセンス |
|---|---|---|
| numpy | 配列演算 | BSD-3-Clause |
| opencv-python-headless | 画像処理(射影変換・罫線検出・二値化) | Apache-2.0(同梱ライブラリは LGPL 等) |
| Pillow | 画像の読み込み(EXIF 回転・TIFF)・合成帳票の文字描画 | HPND |
| pypdfium2 | PDF のページ画像化(PDFium: BSD / Apache-2.0)。PyMuPDF(AGPL)は使用禁止 | Apache-2.0 / BSD-3-Clause |
| pydantic | データモデル・テンプレートの検証 | MIT |
| pytest(開発) | テスト | MIT |
| ruff(開発) | リント・整形 | MIT |

後の Phase で導入予定(未導入): FastAPI + uvicorn、pywebview、PyTorch、SQLite(標準ライブラリ)。

## フォント(合成帳票用)

フォントファイルはリポジトリに含めない(`.gitignore` で除外)。手書き風フォントは SIL OFL 1.1 のものだけを
`scripts/fetch_fonts.py` で google/fonts リポジトリから取得する(取得先・ライセンスの一覧は `docs/fonts.md`)。
環境変数 `YAYOI_FONT_DIR` を指定するとそのディレクトリのフォントを優先する。見出し帯の印刷体と、手書き風フォントが
無いときの代替には、システムの日本語フォント(IPA フォントライセンス等)を使う。

### 取得した手書き風フォント(すべて SIL Open Font License 1.1・google/fonts の `ofl/` 配下から取得)

| フォント | 作者 | 取得ファイル | 出所 |
|---|---|---|---|
| Yomogi | Satsuyako | Yomogi-Regular | https://github.com/google/fonts/tree/main/ofl/yomogi |
| Zen Kurenaido | Yoshimichi Ohira | ZenKurenaido-Regular | https://github.com/google/fonts/tree/main/ofl/zenkurenaido |
| Klee One | Fontworks Inc. | KleeOne-Regular, KleeOne-SemiBold | https://github.com/google/fonts/tree/main/ofl/kleeone |
| Hachi Maru Pop | Nonty | HachiMaruPop-Regular | https://github.com/google/fonts/tree/main/ofl/hachimarupop |
| Yusei Magic | Tanukizamurai | YuseiMagic-Regular | https://github.com/google/fonts/tree/main/ofl/yuseimagic |
| Kiwi Maru | Hiroki-Chan | KiwiMaru-Light/Regular/Medium | https://github.com/google/fonts/tree/main/ofl/kiwimaru |
| Zen Maru Gothic | Yoshimichi Ohira | ZenMaruGothic-Light/Regular/Medium/Bold/Black | https://github.com/google/fonts/tree/main/ofl/zenmarugothic |

取得は `scripts/fetch_fonts.py`(取得時に METADATA.pb のライセンス表記と OFL.txt の本文を確認)。フォントファイルは
コミットしない(`$YAYOI_PRIVATE_DATA/fonts`)。IPA ゴシック等の印刷体は、見出し帯の文字と手書き風フォントが無いときの代替だけに使う。
詳細は `docs/fonts.md`。

