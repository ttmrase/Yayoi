# Yayoi OCR — 手書き経理帳票 → 弥生会計 CSV

手書きの経理帳票(A4・1 様式)のスキャン画像・スマホ写真を文字認識し、弥生会計へ取り込める CSV に変換する
Windows 向け・完全オフラインのアプリです。書き手本人の字形を、確認フェーズで確定した結果から逐次学習して
精度を上げていきます。

> **このリポジトリは public です。** サンプル画像・切り出し画像・正解データ・学習データ・モデルは
> **絶対にコミットしません**(後述)。

## 開発状況

| Phase | 内容 | 状態 |
|---|---|---|
| 0 | 骨組み・データモデル・様式テンプレート・合成帳票生成器 | 進行中 |
| 1 | 前処理・表の位置合わせ・セル切り出し | 進行中 |
| 2 | 認識エンジン・後処理(同上記号の展開など)・弥生 CSV | 未着手 |
| 3 | 確認フェーズ(Web UI)・インクリメンタル学習 | 未着手 |

## セットアップ

```sh
python -m venv .venv && . .venv/bin/activate     # Python 3.11 以上
pip install -e ".[dev]"
sh scripts/install_hooks.sh                       # 実データ混入チェックの git フックを有効化
```

## 実データの置き場所

実データは **リポジトリの外** の `YAYOI_PRIVATE_DATA`(既定 `~/yayoi_private`)に置きます。

```
$YAYOI_PRIVATE_DATA/
  samples/        # サンプル画像
  ground_truth/   # 正解データ(スキーマは docs/ground_truth_schema.json)
  debug/          # 重ね描き画像・切り出し画像などのデバッグ出力
```

コード・テスト・ドキュメント・コミットメッセージに実データの値(実際の摘要・取引先・金額など)を書かないでください。
テストは実行時に合成した架空の帳票(`yayoi_ocr.synth`)を使います。

## 使い方(開発用 CLI)

```sh
python -m yayoi_ocr.cli layout <画像> --out <出力先>
```

画像の向きと傾きを補正し、表の罫線を 1 本ずつ追跡してセルを切り出します。出力先には
重ね描き画像(`overlay.png`)、正準化したページ画像、セルごとの切り出し画像、`layout.json` が書かれます。
**出力先はリポジトリの外**(`YAYOI_PRIVATE_DATA` 配下など)にしてください(リポジトリ内は拒否されます)。

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
  recognize.py    認識エンジンのインターフェース
  postprocess.py  後処理(Phase 2)
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
