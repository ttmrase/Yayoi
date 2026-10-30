# Yayoi OCR — 手書き経理帳票 → 弥生会計 CSV

手書きの経理帳票(A4・1 様式)のスキャン画像・スマホ写真を文字認識し、弥生会計へ取り込める CSV に変換する
Windows 向け・完全オフラインのアプリです。書き手本人の字形を、確認フェーズで確定した結果から逐次学習して
精度を上げていきます。

> **このリポジトリは public です。** サンプル画像・切り出し画像・正解データ・学習データ・モデルは
> **絶対にコミットしません**(後述)。

## 開発状況

| Phase | 内容 | 状態 |
|---|---|---|
| 0 | 骨組み・データモデル・様式テンプレート・合成帳票生成器(書き手シミュレーション) | 完了 |
| 1 | 前処理・表の位置合わせ・セル切り出し | 完了 |
| 2 | 認識(セル種別・PARSeq・後処理・見覚えメモリ・確信度の較正) | 進行中(`docs/recognition.md`) |
| 3 | 確認フェーズ(Web UI) | 未着手 |
| 4 | バックグラウンド学習(世代管理・巻き戻し) | 未着手 |
| 5 | 弥生 CSV 出力 | 未着手 |
| 6 | インストーラ・ログ・精度レポート | 未着手 |

## セットアップ

```sh
python -m venv .venv && . .venv/bin/activate     # Python 3.11 以上
pip install -e ".[dev]"
sh scripts/install_hooks.sh                       # 実データ混入チェックの git フックを有効化
```

## 実データの置き場所

実データは **リポジトリの外** の `YAYOI_PRIVATE_DATA`(既定 `~/yayoi_private`)に置きます。
開発中は実データの内容を一切扱いません(サンプル画像は様式のレイアウト実測・セル境界の目視検証にだけ使い、
文字の読み取り・学習・評価には使わない)。実際の帳票での学習は、納品後にユーザーの PC 内で行います。

```
$YAYOI_PRIVATE_DATA/
  samples/        # サンプル画像
  fonts/          # 合成帳票用の手書き風フォント(scripts/fetch_fonts.py で取得。コミットしない)
  debug/          # 重ね描き画像・切り出し画像などのデバッグ出力
```

コード・テスト・ドキュメント・コミットメッセージに実データの値(実際の摘要・取引先・金額など)を書かないでください。
テスト・評価はすべて、実行時に合成した架空の帳票(`yayoi_ocr.synth`)で行います。

## 使い方(開発用 CLI)

```sh
python -m yayoi_ocr.cli layout <画像> --out <出力先>
```

画像の向きと傾きを補正し、表の罫線を 1 本ずつ追跡してセルを切り出します。出力先には
重ね描き画像(`overlay.png`)、正準化したページ画像、セルごとの切り出し画像、`layout.json` が書かれます。
**出力先はリポジトリの外**(`YAYOI_PRIVATE_DATA` 配下など)にしてください(リポジトリ内は拒否されます)。

## 合成帳票(テスト・評価・学習用)

開発中の評価はすべて合成帳票で行います。

```sh
python scripts/fetch_fonts.py --write-docs     # 手書き風フォント(SIL OFL)を取得(コミットしない。docs/fonts.md に一覧)
python -m yayoi_ocr.cli synth --out ~/yayoi_private/synth -n 20 [--writer 3]
python scripts/eval_layout.py --n 120 --jobs 4 # セル境界 IoU と黙った失敗の評価
```

「書き手」= フォント 1 つ + 書き手ごとに固定した変形パラメータ(傾き・線の太さ・大きさ・字間・ベースラインの
ゆらぎ・文字ごとの弾性変形・〃の形など)。同じ書き手には一貫した癖が出るので、後で学習ループ(使うほど精度が
上がるか)の検証に使えます。内容は架空の取引データで、日付(`R7 11.19` と省略形、〃)・科目・摘要(〃・矢印・
波線による省略・セル上部の小さな注記・はみ出し記入)・金額(`3,456-`)・ページタイトル・No. を含み、
正解 JSON(raw / resolved。`docs/ground_truth_schema.json`)も一緒に出力します。

## 認識(Phase 2)の開発用スクリプト

認識モデルの重み・合成データセットはすべて `$YAYOI_PRIVATE_DATA`(既定 `~/yayoi_private`)に置きます(コミットしない)。

```sh
pip install -e ".[dev,ml,ml-dev]"                      # PyTorch・onnx・onnxruntime(開発用)
python scripts/fetch_models.py                         # NDLOCR-Lite の PARSeq(CC BY 4.0)を固定コミットから取得・変換
python scripts/build_synth_cells.py --out ~/yayoi_private/datasets/train --n 300 --preset typical --jobs 4
python scripts/train_celltype.py --train DIR --calib DIR --out ~/yayoi_private/models/celltype.pt
python scripts/eval_recognize.py --data DIR --classifier ~/yayoi_private/models/celltype.pt --dump OUT.jsonl
python scripts/fit_confidence.py --dump OUT.jsonl --out ~/yayoi_private/models/confidence.json
```

出典・ライセンス・SHA256 は `docs/dependencies.md`、処理の流れと評価指標は `docs/recognition.md`。

## テスト

```sh
python -m pytest                                # 合成画像による自動テスト(実サンプルがあれば追加で実行)
ruff check . && ruff format --check .
python scripts/check_no_private_data.py --all
```

## ディレクトリ構成

```
src/yayoi_ocr/
  ingest.py       取り込み(PDF / PNG / JPEG / TIFF / EXIF 回転)
  preprocess.py   前処理(照明補正・コントラスト正規化・ノイズ除去)
  layout/         表の位置合わせ・セル切り出し(見出し帯 → 外枠 → 罫線 1 本ずつ)
  recognize/      認識(セル種別・PARSeq・列ごとの解決・ページ単位の解決・確信度)
  review.py       確認フェーズ(Phase 3)
  learn.py        インクリメンタル学習(Phase 3)
  export.py       CSV 出力(Phase 2)
  storage.py      データ保存(Phase 3)
  models.py       Document → Page → Row → Cell
  template.py     様式テンプレート(templates/form_v1.json)
  synth/          合成帳票の生成器(テスト・学習用)
docs/             設計メモ・依存ライブラリの記録・正解データのスキーマ
scripts/          check_no_private_data.py と git フック
```
