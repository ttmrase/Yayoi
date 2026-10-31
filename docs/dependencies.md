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

認識用(オプション依存 `[ml]` / `[ml-dev]`。出所はすべて PyPI):

| ライブラリ | 用途 | ライセンス |
|---|---|---|
| torch | 文字列認識(PARSeq)・分類器・学習 | BSD-3-Clause |
| PyYAML | 文字集合(NDLmoji.yaml)の読み込み(開発・学習用) | MIT |
| onnx(開発) | NDLOCR-Lite の ONNX から重みを復元(`scripts/fetch_models.py`) | Apache-2.0 |
| onnxruntime(開発) | 同梱 ONNX との数値一致テスト | MIT |

後の Phase で導入予定(未導入): FastAPI + uvicorn、pywebview、SQLite(標準ライブラリ)。

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


## 文字列認識モデル(R1): NDLOCR-Lite の PARSeq tiny(ver1.2, 高さ 24)

### 出典とライセンス

- **学習済みモデル**: 国立国会図書館 NDLOCR-Lite ver1.2 の文字列認識モデル
  `parseq-ndl-24x256-30-tiny-189epoch-tegaki3-r8data-202604`(最大 30 字)と
  `parseq-ndl-24x384-50-tiny-300epoch-tegaki3-r8data-202604`(最大 50 字)。
  https://github.com/ndl-lab/ndlocr-lite のコミット `e57f7c6bd9cfb3de5f12f008e71ad2122bd44bfd`、
  ライセンスは **CC BY 4.0**(リポジトリの `LICENCE`)。文字集合は同リポジトリの `src/config/NDLmoji.yaml`(7,141 字)。
  - **帰属表示**: 国立国会図書館「NDLOCR-Lite」の学習済みモデル(CC BY 4.0)を利用。
    重みは ONNX 形式の配布物から PyTorch のテンソルに変換した(形式変換のみ。値は変更していない)。
    この変換物をそのまま、または **ファインチューニング(Phase 4)した派生モデル** を、インストーラに同梱する場合は、
    上記の出典・ライセンス・変更の有無を、アプリの「ライセンス情報」画面と配布物に記載する(Phase 6)。
- **モデルのコード**: baudm/parseq(Scene Text Recognition Model Hub, Copyright 2022 Darwin Bautista)
  commit `1902db043c029a7e03a3818c616c06600af574be`、**Apache License 2.0**。
  `src/yayoi_ocr/recognize/parseq/model.py` に推論に必要な部分だけを取り込んだ(冒頭に出典・ライセンス・変更点を記載。
  ライセンス全文は `docs/LICENSE-parseq-Apache-2.0.txt`)。変更点: 学習用コードの削除、timm 依存の除去
  (同じパラメータ名で ViT を自前実装)、系列の対数尤度・ビームサーチ・列ごとの文字集合マスク・エンコーダ出力の追加、
  反復修正マスクの ONNX 互換モード。
- 前処理は NDLOCR-Lite の `src/parseq.py`(CC BY 4.0)と同じ手順(縦横比を無視して幅 x 高さ 24 にリサイズ、BGR→RGB、
  /127.5 - 1)を `preprocess.py` に再実装した。

### 取得と配置(開発用。アプリは実行時にダウンロードしない)

`python scripts/fetch_models.py`(`pip install -e ".[ml,ml-dev]"`)が、上記コミットから ONNX と `NDLmoji.yaml` を取得し、
SHA256 を検証して、`$YAYOI_PRIVATE_DATA/models/` に置く。ONNX から PyTorch の重みを復元した
`parseq-ndl-24x{256,384}-….pt`(テンソルと基本型のみ。`torch.load(weights_only=True)` で読める)も同じ場所に作る。
**重みはリポジトリにコミットしない**(`.gitignore` で `*.onnx` `*.pt` `*.ckpt` を除外)。

| ファイル | SHA256 |
|---|---|
| parseq-ndl-24x256-30-tiny-189epoch-tegaki3-r8data-202604.onnx | `9e651bae4c1a4d5254da1127e86e82e21ef62d5339b37e62d4a3d3d30831772d` |
| parseq-ndl-24x384-50-tiny-300epoch-tegaki3-r8data-202604.onnx | `49cea9db4552f19eb05c8ee202fcf74714977749b2f4c9376b127fde41b07a99` |
| NDLmoji.yaml | `f6ad5a2de444b495155866af811cf1a98309dcae3225db802767ea531a2dc529` |
| (参考)…24x256….ckpt(NDL 提供・ファインチューニング用) | `7bd3b5d552e24500a0f94d702904a3beb82368c9612df2f0751f4aa137a8c7ea` |
| (参考)…24x384….ckpt | `fbdd5a165d9505caea264f07f56309429fdd6a638d1bc51d74341f20a9dc3bfb` |

### 数値一致の検証(tests/test_parseq_equivalence.py)

- ONNX から復元した重みで動かした自作モデルの logits は、ONNX Runtime の出力と一致する
  (幅 256: 最大差 2.4e-4、幅 384: 7.0e-4。argmax・認識文字列の不一致 0)。テスト入力は合成文字列と乱数画像のみ。
- **ckpt と同梱 ONNX の関係**: 幅 384 は全 175 テンソルがビット同一。幅 256 は **別スナップショット**
  (ckpt の内部 epoch は 253、ONNX のファイル名は 189epoch。相対 L2 差は中央値 4.5%、最大 20.5%)。
  そのため **基礎重みには幅 256/384 とも ONNX 由来の重みを使う**(同梱配布物で出所が一意)。ckpt は参照用。
- ckpt は pickle 形式なので `torch.load(weights_only=False)` は使わない。`weights_only=True` と、ckpt に含まれる学習用
  スケジューラ状態が参照する `getattr` / `SWALR` のみの許可で読む(それ以外の pickle グローバルを含むファイルは拒否)。
- **ONNX の癖**: torch.onnx のトレースにより、attention マスクが -inf ではなく「値 1.0 の float 行列(加算バイアス)」として
  書き出されている。AR デコードは未来トークンをスライスで除くので影響しないが、反復修正(cloze マスク)では
  自分自身のトークンが遮蔽されない。`PARSeq.refine_mask = "onnx"`(既定。NDLOCR-Lite が実際に配布・実行する挙動)と
  `"cloze"`(baudm/parseq の PyTorch 実装)を切り替えられる。

## 確認 UI(Phase 3)

### Python(サーバ。extras `app` / `app-dev` / `desktop`)

| パッケージ | 用途 | ライセンス |
|---|---|---|
| fastapi / starlette / uvicorn | ローカルサーバ(127.0.0.1 のみ) | MIT / BSD-3-Clause / BSD-3-Clause |
| httpx(`app-dev`) | サーバのテスト | BSD-3-Clause |
| pywebview(`desktop`) | Windows のウィンドウ(WebView2 を使う) | BSD-3-Clause |

### フロントエンド(`web/`。ビルド成果物だけをアプリに同梱し、Node は利用者の環境に要らない)

アプリに入る(本番のバンドル): react 18.3.1(MIT)、react-dom 18.3.1(MIT)、scheduler 0.23.2(MIT)、loose-envify 1.4.0(MIT)、js-tokens 4.0.0(MIT)。
ビルド・テストだけに使う(同梱しない): vite 5.4.11(MIT)、@vitejs/plugin-react 4.3.3(MIT)、typescript 5.6.3(Apache-2.0)、vitest 2.1.5(MIT)、
jsdom 25.0.1(MIT)、@playwright/test 1.49.0(Apache-2.0)、@axe-core/playwright 4.10.1 と axe-core 4.10.3(アクセシビリティ検査。MPL-2.0。テスト専用で同梱しない)、@types/react・@types/react-dom(MIT)。ロックファイル `web/package-lock.json` で固定する(`npm ci`)。

- **外部リソースを読み込まない**: CDN・Web フォント・外部スクリプトは使わない。フォントは OS のシステムフォント(`Yu Gothic UI` など)を名前で指定するだけ。
  `tests/test_app_static.py` がビルド成果物に外部 URL が無いことを(名前空間の識別子と React のエラー文言を除いて)検査し、Playwright の E2E が実行時の通信先を検査する。
