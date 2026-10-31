# 同梱する重みの出所(試用版の配布前の記録)

試用版に同梱するすべてのファイルについて、由来・学習/較正に使ったデータ・実行したスクリプト・SHA256 を記録する。
最大の論点は **「実サンプル(利用者の帳票画像)に由来するデータが重みに入っていないこと」**。結論と根拠、**開示すべき事実**(§5)を分けて書く。
検査は `scripts/model_provenance.py`(§6)で再実行できる。

## 1. 結論

- 認識の重み(PARSeq の基本/適応済み、セル種別分類器)と確信度モデルは、**NDLOCR-Lite の学習済みモデル(変換のみ)と、合成データ(架空の帳票)だけ**から作った。
  実サンプルの画像・切り出し・文字起こし・統計・しきい値は、これらの**学習・較正・しきい値の決定のどこにも入っていない**(コードの経路と検査は §4)。
- ただし **重みとは別に**、レイアウト解析(罫線の検出などの古典的な画像処理)の定数と様式テンプレートの寸法は、実サンプルを見て決めた。重みファイルには含まれないが、
  「実サンプルを使った処理」として §5 に全部開示する。**また、開発の最初の約 11 分間だけ、実サンプルの文字起こしを private に作って削除した**(§5-c)。
- 同梱するのは §2 の 9 ファイルだけ(許可リスト。`scripts/make_trial_bundle.py` が強制する)。評価用の画像・正解・学習データ・実験用の重みは入らない。

## 2. 同梱するファイル

| ファイル | 由来 | 作り方 | 学習/較正データ | SHA256 |
|---|---|---|---|---|
| `models/NDLmoji.yaml` | NDLOCR-Lite(CC BY 4.0)の文字集合 | `scripts/fetch_models.py` が固定コミットから取得し SHA256 を検証 | なし(配布物そのもの) | `f6ad5a2de444b495155866af811cf1a98309dcae3225db802767ea531a2dc529` |
| `models/parseq-ndl-24x256-30-tiny.pt` | NDLOCR-Lite の PARSeq tiny(幅 256)の **ONNX を PyTorch のテンソルに変換**したもの | `fetch_models.py`(重みの値は変更しない。形式変換のみ) | なし(NDL が学習済み) | `fdb63f461191878f4fdd5b164b99f7db36648517f078d6334083b5e5a1489fc1` |
| `models/parseq-ndl-24x384-50-tiny.pt` | 同上(幅 384) | 同上 | なし | `42ec07c48e9eacbb02047c257269576cb46c9d4e535c250a5abac065ba8ef97a` |
| `models/celltype.pt` | **学習**(自作の小型 CNN。セル種別: 空欄/文字/〃/→/〜) | `scripts/train_celltype.py --train datasets/train --calib datasets/calib --epochs 8 --threads 4` | 合成 `train`(500 ページ)+ `calib`(100 ページ) | `5cbd43b5d018c566031b48f04ce8380194bc711e188340432fca387b4f65d095` |
| `models/confidence.json` | **較正**(基本の重み用の確信度モデル) | 合成 `calib500`・`calib_gen` を基本の重みで認識したダンプから `scripts/fit_confidence.py` で当てはめ | 合成 `calib500`(500 ページ)+ `calib_gen`(200 ページ) | `5204f5d5f8d379017cee41a5397129ae3090240128151b587b425804811e6fcb` |
| `models/adapted/parseq-ndl-24x256-30-tiny.pt` | 基本の重み(上)を **合成データで追加学習**(L2-SP) | `scripts/train_adapt.py --train datasets/train datasets/train_gen --out-dir models/adapted --l2sp 0.1 --epochs 3 --threads 3` | 合成 `train`(500)+ `train_gen`(300) | `1f3f5b4861c54d01b71e24742cdbe8c4fee02ce2872b33ae45c303a862445d8a` |
| `models/adapted/parseq-ndl-24x384-50-tiny.pt` | 同上(幅 384) | 同上 | 同上 | `ab6d17ac0f0970451d5d6c05a9f39e39ad88f88f90c23dcea181bfc41a0507a9` |
| `models/rehearsal.npz` | **合成セルの束**(追加学習のリハーサル用。利用者のセルだけで学習すると他の書き手・一般語彙を忘れるので、学習のたびに利用者のセル数の 50% を混ぜる) | `scripts/build_rehearsal.py --data datasets/train datasets/train_gen --out models/rehearsal.npz`(seed 0。再実行でビット単位で同じものができる。作成時に、入力の全ページが生成器の署名[整数の seed・writer_id と OFL フォント名]を持つことを検査し、入力にしたデータセットの識別子[名前・使ったページ数・seed の範囲]を束の `meta` に記録する) | 合成 `train`(500)+ `train_gen`(300)から幅 256 の 1,200 件(日付・科目・金額)と幅 384 の 1,200 件(摘要)を無作為抽出。**学習の対象ではなく、配布する合成データ**(手書き風 OFL フォントで描いた架空の内容) | `bd10074a7ae599369b84b1f4738704f1780f5a11c005238d0267ea4ba674a9da` |
| `models/adapted/confidence.json` | **較正**(適応済みの重み用) | `calib500`・`calib_gen` を適応済みの重みで認識したダンプから `fit_confidence.py` | 合成 `calib500` + `calib_gen` | `9741f7166825713e08105cb3b0605eb7cd39fd55bac185e36ea6be17716a6c4a` |

- 基本の重みの再現: `fetch_models.py` で ONNX から作り直した `.pt` の SHA256 は上の値と**ビット単位で一致する**(`model_provenance.py --reconvert`)。
- ONNX の出所: <https://github.com/ndl-lab/ndlocr-lite> のコミット `e57f7c6bd9cfb3de5f12f008e71ad2122bd44bfd`。ONNX の SHA256 は `scripts/fetch_models.py` に固定してある。ライセンス・帰属は `docs/dependencies.md`。
- 追加学習の枠組み: 先頭 8 ブロックを凍結し、初期重みへの L2 アンカー(L2-SP、係数 0.1)をかけて摘要の忘却を抑える。学習率 1e-4、エポック 3、幅 256 は日付・科目・金額の各列 2,500 件、幅 384 は摘要 6,000 件(既定値)。
- 同じディレクトリにある `confidence_*.json`・`adapted/confidence_mem.json`・`adapted/confidence_seq.json`・`models_ft1/` は実験用で、**同梱しない**(許可リストにない)。

### 実行したスクリプトとコミット

作業記録(実行時刻 UTC 2026-10-30)から、実行時点の HEAD を引いた。作業ツリーに未コミットの変更があった可能性は排除できないが、スクリプトの該当部分は後のコミットで変わっていない。

| 作業 | 実行時刻 | その時点の HEAD |
|---|---|---|
| 基本の重みの変換(`fetch_models.py`) | 08:28 | `f023270`(スクリプトは `fdd74fb` で追加。作成時は未コミット) |
| 合成データ `train`・`calib` の生成 | 08:42〜 | `fdd74fb` |
| `celltype.pt` の学習 | 09:40〜10:16 | `1fca083` |
| 合成データ `calib500` の生成 | 11:17〜 | `0689e17` |
| 合成データ `train_gen` の生成 | 11:23〜 | `0b16993` |
| `adapted/*.pt` の学習(`train_adapt.py`) | 11:53〜13:19 | `eba959c` |
| 合成データ `calib_gen` の生成 | 14:44〜 | `0412a91` |
| 確信度モデルの最終当てはめ(`fit_confidence.py`、2 つとも) | 18:45 | `0ac53a4` |

## 3. 学習・較正に使った合成データ

生成器: `scripts/build_synth_cells.py`(`src/yayoi_ocr/synth/`)。**合成帳票を描いてレイアウト解析を通し、セル単位の画像と正解(生成した論理データ)を保存する**。
内容はすべて架空: 一般的な勘定科目名・費目の語彙(`synth/content.py`・`synth/vocab.py`)と、乱数で作った日付・金額。書き手は「書き手シミュレーション」(`synth/writer.py`)で、フォントは SIL OFL の手書き風(`docs/fonts.md`)。
`datasets/` は**リポジトリの外**(`YAYOI_PRIVATE_DATA`)にあり、コミットしない。下の表は `model_provenance.py` が各ページのメタデータ(`seed`・`writer_id`・`font`)から集計した。

| データセット | 役割 | ページ | セル | seed の範囲 | 書き手 ID(人数) | フォント |
|---|---|---:|---:|---|---|---|
| `train` | 学習(`celltype.pt`・`adapted/*.pt`) | 500 | 58,000 | 1000000–1000299, 1100000–1100099, 1200000–1200099 | 0–499(500) | OFL 11 書体(下) |
| `train_gen` | 学習(`adapted/*.pt` の一般語彙) | 300 | 34,800 | 8000000–8000299 | 500000–500299(300) | 同上 |
| `calib` | 較正(`celltype.pt` の確率の較正) | 100 | 11,600 | 2000000–2000099 | 100000–100099(100) | 同上 |
| `calib500` | 較正(`confidence.json` ×2) | 500 | 58,000 | 7000000–7000499 | 400000–400499(500) | 同上 |
| `calib_gen` | 較正(`confidence.json` ×2。一般語彙) | 200 | 23,200 | 9000000–9000199 | 700000–700199(200) | 同上 |

フォント(OFL 1.1): HachiMaruPop-Regular、KiwiMaru-Light/Medium/Regular、Yomogi-Regular、ZenKurenaido-Regular、ZenMaruGothic-Black/Bold/Light/Medium/Regular(学習・較正では YuseiMagic・KleeOne を除外 = `--font-exclude yusei klee`。評価でだけ使う)。

### 生成コマンド(作業記録のとおり)

```sh
B="python3 -I scripts/build_synth_cells.py --jobs 3"
# 学習用(書き手 ID 0〜499)
$B --out $D/train --n 300 --preset typical --seed0 1000000 --writer0 0   --font-exclude yusei klee
$B --out $D/train --n 100 --preset clean   --seed0 1100000 --writer0 300 --font-exclude yusei klee
$B --out $D/train --n 100 --preset default --seed0 1200000 --writer0 400 --font-exclude yusei klee
# 較正用(学習・評価と別の書き手)
$B --out $D/calib    --n 100 --preset typical --seed0 2000000 --writer0 100000 --font-exclude yusei klee
$B --out $D/calib500 --n 500 --preset typical --seed0 7000000 --writer0 400000 --font-exclude yusei klee
# 一般語彙(摘要の汎用的な読み取りのリハーサル用と、較正用)
$B --out $D/train_gen --n 300 --preset typical --vocab general-train --strength-range 0.5 1.0 --seed0 8000000 --writer0 500000 --font-exclude yusei klee
$B --out $D/calib_gen --n 200 --preset typical --vocab general-eval  --seed0 9000000 --writer0 700000 --font-exclude yusei klee
```

較正用のダンプ(認識結果と特徴量): `scripts/eval_recognize.py --data $D/calib500 --classifier models/celltype.pt [--models-dir models/adapted] --dump E/calib500_{base,adapted}.jsonl`、
`--data $D/calib_gen … --dump E/calib_gen_{base,adapted}.jsonl`。確信度の当てはめの最終コマンド(基本の重みと適応済みの重みで 2 回):

```sh
python3 -I scripts/fit_confidence.py --dump $E/calib500_base.jsonl    $E/calib_gen_base.jsonl    --out $M/confidence.json         --margin 0.6 --margins amount=0.4 --column-dump description=$E/calib_gen_base.jsonl
python3 -I scripts/fit_confidence.py --dump $E/calib500_adapted.jsonl $E/calib_gen_adapted.jsonl --out $M/adapted/confidence.json --margin 0.6 --margins amount=0.4 --column-dump description=$E/calib_gen_adapted.jsonl
```

**学習・較正・評価の分離**: 上の 5 つのデータセットは、seed の範囲も書き手 ID の範囲も互いに重ならない(検査 C3)。評価用(`test_*`・`f_*`・`lc_*`・`final_*` など)は別の seed・書き手 ID を使い、重みの作成には使っていない。

## 4. 実サンプルが重みに入っていないことの根拠

| 検査 | 内容 | 結果 |
|---|---|---|
| C1 コードの経路 | 実サンプルの置き場(`samples_dir()` = `YAYOI_PRIVATE_DATA/samples`)を参照するのは、`src/yayoi_ocr/config.py` の定義と `tests/test_samples.py`(レイアウトの構造チェック。行数・列数と空欄の数だけで、内容は読まない)**だけ**。`scripts/` の学習・較正・評価の全スクリプトは参照しない。`tests/test_provenance_paths.py` が CI で、この状態が崩れたら失敗する | なし(許可リストのみ) |
| C2 合成の署名 | 学習・較正に使った 1,600 ページの**全部**が、生成器のメタデータ(整数の `seed`・`writer_id`、OFL フォント名)を持つ | 署名のないページ 0 |
| C3 分離 | 5 つのデータセットで seed・書き手 ID が重ならない | 重なりなし |
| C4 バイト列の走査 | 同梱 9 ファイルのバイト列を(`.npz` は展開して。画像は生データ扱い、`meta` の文字列は全部の語で)、拒否リスト(実データの語 35 項目)で走査(UTF-8・UTF-16LE・cp932。テンソルの生データは 6 バイト以上の語のみ=短い語は偶然一致するため。NDL 配布の重みから変換しただけのファイルにも短い語は一致することを確認済み)。json・yaml は数値も検査 | 検出 0 |
| C5 変換の再現 | ONNX から基本の重みを作り直すと、同梱のものと SHA256 が一致する | 一致(2/2) |
| C6 学習データの文字列 | 学習・較正データの正解の文字列(タイトル・raw・resolved)に拒否リストの語が含まれない。数値は金額が乱数なので偶然の一致があり得る → 新しい seed(学習に使っていない)で作った内容との**一致率を比較**した | 語 0 件。数値の一致率は学習・較正データ 0.0788/ページ、新しい seed の内容 0.0800/ページ(同程度 = 偶然) |

加えて、リポジトリ全体(履歴の未 push 分を含む)は `scripts/check_no_private_data.py`(拒否リスト・画像/PDF/整数列の検査)を毎コミットで通している。

## 5. 開示事項: 実サンプルを使った処理の全部

standing rule(開発中は実データの内容を扱わない。サンプルは様式のレイアウト実測とセル境界の目視検証にだけ使う)の以前または範囲に関わる事実を、省かずに記す(UTC 2026-10-30)。

**(a) 様式テンプレート `src/yayoi_ocr/templates/form_v1.json`**(06:21)。白紙の様式を含む実サンプル 3 枚の写真から、縦罫線・行の高さ・見出し帯・タイトル欄などの**寸法**を平均して決めた。文字内容は含まない。リポジトリ(public)にあり、合成帳票の生成器も同じ寸法を使う。

**(b) レイアウト解析(罫線の検出・セル切り出し・空欄判定)の定数**(06:18〜06:51、07:06)。古典的な画像処理(学習する重みなし)で、
実サンプルの重ね描き画像・切り出し画像を**目視**し、薄い記入を拾う感度(`layout/cells.py` の `FAINT_*`)や罫線除去の幅などの定数を、画素強度・面積の統計を見て決めた。
これらは**コード内の定数**(public リポジトリ)で、重みファイルではない。以後のレイアウトの評価(`eval_layout.py`)は合成ページだけで行っている。
画素の統計は文字内容を表さないが、これらの定数が実サンプルに合わせて調整されている事実は変わらない。

**(c) 開発の最初に、記入済みサンプル 2 枚の文字起こし(正解データ)を作った**(06:42〜06:44 に切り出し画像を目視して `ground_truth/` に作成)。
このデータは **06:55 に削除**し(`rm -rf ground_truth debug/gt_*`、約 11 分間)、**どのスクリプトからも読まれていない**(作業記録で、作成と削除以外の参照がない)。
その過程で、実値がリポジトリの履歴に混入していたことが分かり(be781d6)、director の許可を得て履歴を書き換えて除去した。現在、`ground_truth` に相当するファイルはどこにもない。
合成帳票の内容(`synth/content.py`)は、この削除の**後に**書き直した(06:59)。削除前の最初の版の合成器は履歴ごと除去されており、学習データの生成には一度も使っていない(最初の学習データ生成は 08:42 で、書き直し後の版)。書き直し後の内容は一般的な科目名・費目と乱数で、拒否リストの語を含まない(C6、リポジトリ全体の検査)。
目視した内容が語彙の選び方に無意識に影響した可能性は、完全には排除できない(語彙は一般的な科目名・費目語で、固有の値は含まない)。

**(d) 実サンプルの構造チェック**: `tests/test_samples.py` と `python -m yayoi_ocr.cli layout <画像>` は、レイアウト解析を実サンプルに実行して、行数・列数と空欄の数を検査する(`samples` マーカー。無ければスキップ)。内容は読まない。

**(e) 認識・学習・較正・評価には使っていない**: 認識器(PARSeq・セル種別分類器)・確信度・見覚えメモリ・しきい値(Clopper-Pearson の見逃し率)は、すべて合成データだけで学習・較正・評価した。
実サンプルの切り出しは、これらの入力に一度も使っていない(§4 C1: 経路が存在しない)。

→ **配布の判断材料**: 重みファイルには実サンプル由来のものは含まれない。ただし (a)(b) のレイアウトの寸法・定数と、(c) の経緯は開示事項である。
より厳格にするなら、(b) の定数を合成ページだけで再調整する作業が選べる(必要なら指示してほしい)。

## 6. 再検査の手順

```sh
python scripts/model_provenance.py --reconvert    # C1〜C6 と SHA256 の一覧。失敗があれば終了コード 1(要 YAYOI_PRIVATE_DATA と拒否リスト)
pytest tests/test_provenance_paths.py             # C1 だけ(CI でも走る)
```

配布物(zip)を作るときは `scripts/make_trial_bundle.py` が許可リストの 9 ファイルだけを入れ、`MANIFEST.sha256` と zip の SHA256 を表示・記録する(`docs/windows_setup.md`)。
同梱前に、この文書の SHA256 の表と実ファイルが一致することを確認する(`model_provenance.py` の先頭の表)。
