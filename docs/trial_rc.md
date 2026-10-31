# 試用版の RC(リリース候補): 記録

> **配布するのは、この RC(開発のコンテナで作った束)ではなく、GitHub Actions でビルドした束**です(`docs/trial_ci_build.md`)。この RC は、参考として残してある(コンテナの重みとの比較用)。

**この束は公開していない。** GitHub の Release・リポジトリ・その他どこにも置いていない。開発のコンテナの中(`$YAYOI_PRIVATE_DATA/trial/`、リポジトリの外)にだけある。
公開・配布は、利用者の承認(director の Q10 / Q10-b)のあとに director / planning が指示する。それまで、この束を外に出さない。

## 1. 束

| 項目 | 値 |
|---|---|
| ファイル | `yayoi-ocr-trial-0.1.0.zip`(オフライン版。`--smoke` でない完全版) |
| 大きさ | 385,422,030 バイト(385.4 MB)、114 ファイル |
| **zip の SHA256** | `5d69f66680faea2bd6b9846063236d85cbd82dc1f058a3316349146568526d89` |
| 作ったコミット | `4651b73d6e2b6b9c8cd96749ded501f624df5dfc`(作業ツリーはクリーン。ブランチ `claude/handwritten-document-ocr-sgf76h`。このあとのコミットは、束に入らない文書・E2E の試験だけの変更) |
| 作った日時 | 2026-10-31 07:36 UTC |
| 作った環境 | Linux・Python 3.13.17・Node 22(フロントエンドのビルド)。同梱するものは Windows 64 ビット・CPython 3.11 向け |
| 作り方 | `cd web && npm run build` → `python scripts/make_trial_bundle.py`(出力先 `$YAYOI_PRIVATE_DATA/trial/`) |
| 同梱の Python | python-build-standalone `cpython-3.11.18+20261026`(install_only_stripped、Windows x86_64)。SHA256 `67215a7bf1643e33d01b17ee3f1e91a8b0b39163746855ee02594bf0608f68b4`(固定値と公式 SHA256SUMS の両方で検証) |

- **zip はビット単位では再現しない**(zip の項目の更新時刻が入る)。同じ SHA256 の束を作り直すことはできない。束の同一性は、上の zip の SHA256 と、§8 の各ファイルの SHA256(`MANIFEST.sha256` と同じ)で確かめる。
- zip の SHA256 は、zip の配布経路とは別の経路で利用者に伝える(`docs/windows_setup.md`)。`<zip>.sha256` に同じ値を書いてある。
- 試用版の束(`--smoke` あり)との違い: 完全版は、セル種別分類器 `celltype.pt`・確信度モデル `confidence.json`・リハーサルの束 `rehearsal.npz`・**適応済みの重み `models/adapted/`** を含む。smoke 版(CI 用)は基本の重みだけで、配布しない。

## 2. 重みの出所との突き合わせ(`docs/model_provenance.md` の SHA256)

束の `models/` の 9 ファイルを zip から直接読んで SHA256 を求め、出所の文書 §2 の表(9 ファイル)と突き合わせた(`scripts/verify_trial_bundle.py` の B4)。**過不足なく 9 ファイルが一致**(不一致 0・表に無い 0・束に無い 0)。

| ファイル | SHA256(束 = 出所の表) |
|---|---|
| `models/NDLmoji.yaml` | `f6ad5a2de444b495155866af811cf1a98309dcae3225db802767ea531a2dc529` |
| `models/parseq-ndl-24x256-30-tiny.pt` | `fdb63f461191878f4fdd5b164b99f7db36648517f078d6334083b5e5a1489fc1` |
| `models/parseq-ndl-24x384-50-tiny.pt` | `42ec07c48e9eacbb02047c257269576cb46c9d4e535c250a5abac065ba8ef97a` |
| `models/celltype.pt` | `5cbd43b5d018c566031b48f04ce8380194bc711e188340432fca387b4f65d095` |
| `models/confidence.json` | `5204f5d5f8d379017cee41a5397129ae3090240128151b587b425804811e6fcb` |
| `models/rehearsal.npz` | `bd10074a7ae599369b84b1f4738704f1780f5a11c005238d0267ea4ba674a9da` |
| `models/adapted/parseq-ndl-24x256-30-tiny.pt` | `1f3f5b4861c54d01b71e24742cdbe8c4fee02ce2872b33ae45c303a862445d8a` |
| `models/adapted/parseq-ndl-24x384-50-tiny.pt` | `ab6d17ac0f0970451d5d6c05a9f39e39ad88f88f90c23dcea181bfc41a0507a9` |
| `models/adapted/confidence.json` | `9741f7166825713e08105cb3b0605eb7cd39fd55bac185e36ea6be17716a6c4a` |

出所の再検査(`python scripts/model_provenance.py --reconvert`、同日に実行)も **OK**: C1(実サンプルの置き場を参照するコードは許可リストのみ)・C2(学習/較正の合成データは全ページが生成器の署名と OFL フォントを持つ)・
C3(データセット間の seed・書き手 ID の重なりなし)・C4(同梱 9 ファイルの拒否リスト走査、検出 0 件)・C5(NDL の ONNX から基本の重みを作り直すと SHA256 がビット単位で一致、2/2)・C6(合成データの文字列と拒否リスト、語の一致 0)。

## 3. 束の検査(`scripts/verify_trial_bundle.py`、zip を展開せずに読む)

| 検査 | 結果 |
|---|---|
| B1 zip の SHA256 = 控え(`<zip>.sha256`) | OK |
| B2 zip の構造(壊れなし・危険な名前 0・重複 0) | OK(114 ファイル) |
| B3 `MANIFEST.sha256` の 113 行すべてが zip の中身と一致、MANIFEST にないファイルなし | OK |
| B4 `models/` ≡ 出所の表(§2) | OK(9/9) |
| B5 wheel 36 個: アプリの wheel は 1 個で版 `0.1.0` 一致。ほかの 35 個は lock の `名前==版` にあり、lock にハッシュがある 34 個は wheel の SHA256 と一致。同梱の `requirements-lock.txt` = リポジトリの lock | OK |
| B6 同梱の Python の SHA256 = `python-standalone.pin` の固定値 | OK |
| B7 画像・PDF・CSV・DB・zip の類なし。拒否リストの語が、wheel の外のファイルとアプリの wheel の中身に 0 件 | OK |

- 依存ライブラリのライセンス一覧(アプリの「ライセンス」画面): 40 件、**「要確認」0 件**。`clr_loader`(METADATA にライセンス表記がなく、LICENSE ファイルが MIT)は、定型の MIT の本文から判定した(`scripts/gen_licenses.py`)。
- 重みのテンソルの中身は、B7 では走査しない(語は浮動小数点の列に偶然一致するため)。B4 のハッシュで、出所の検査(C4)が走査したファイルと同一であることを担保する。

## 4. 実行して確かめたこと

| 確認 | 対象 | 結果 |
|---|---|---|
| Windows の CI `trial-windows`(windows-latest・Windows PowerShell 5.1): 束を展開 → `setup.ps1` → 開発用でないモードで起動 → HTTP の煙試験(取り込み〜認識〜確定〜cp932 の出力〜バックアップ〜**復元(再起動して入れ替え)**)→ ウィンドウ起動 → アンインストール(データを残す/消す・世代の保持・別フォルダと印のないフォルダの拒否) | **smoke 版の束**(基本の重みだけ。CI に適応済みの重みを置けない)。コミット `0b9e09d`(run 64)・`4651b73`(run 65。RC のビルド元)・`618a91c`(run 66。最終) | **成功**(run 64・66 は全ジョブ成功。run 65 は trial-windows を含む 5 ジョブが成功し、E2E の 2 件が共有ランナーの遅さで揺れて落ちた → `618a91c` で是正)。経緯: run 62・63 は、アンインストールの確認(`models\` が空のまま残った)で落ち、`0b9e09d` で修正 |
| 同じ煙試験(`scripts/trial_smoke.py`)を、**この RC の `models/`(zip から取り出した適応済みの重みを含む 9 ファイル)** で実行(Linux、データ領域 = インストール先の配置) | 最終の RC の束(zip の SHA256 `5d69f666…`)から取り出した重み。コミット `4651b73` の煙試験 | すべて成功(認識 8 秒・116 セル、確定、cp932 の出力、バックアップ → 検査 → 復元の準備 → 再起動して入れ替え → 出荷時の重みが残る・世代の組の照合・新しいページの認識、標準出力・標準エラーに入力した値なし) |
| CI(Python 3.11 の試験・`ruff`・フロントエンドの型検査/Vitest/ビルド・Playwright E2E(ubuntu と windows)・trial-windows) | コミット `7dc5b3a`(run 69。最終) | **全ジョブ成功**。`4651b73`(RC のビルド元)からここまでの変更は、E2E の試験の揺れの是正(共有ランナーの遅さで落ちた perf・⑳・㉜・a11y の 4 件。いずれもアプリの不具合ではない)と文書だけで、**束に入るファイルは変わらない**(束の `docs/`・`scripts/windows/`・wheel の中身は、HEAD と一致を確認済み) |

- **RC そのものを Windows の CI に通すことはできない**(適応済みの重みは公開リポジトリに置けない)。smoke 版の束の Windows CI の結果で代える。smoke 版と RC は、`make_trial_bundle.py` の同じ経路で作る(違いは `models/` の中身と `SMOKE-ONLY.txt` の有無だけ)。
- **Windows の実機の確認は未実施**: `docs/windows_checklist.md` の F'(T1〜T12)・H(バックアップ)・I(ライセンス・ログ・アンインストール)・GPU は、利用者の PC で行う。
  この RC の重みでの認識の速度・実機の IME・高 DPI・WebView2 は、そこで初めて分かる。

## 5. 既知の限界(RC に含まれるもの)

- 重みの適応は合成データだけで行った。実際の手書きでの精度は未測定(実機で確定したページから、追加学習で合わせていく設計。提案モードは安全基準に届かない世代を値の提案にだけ使う)。
- 署名なし(SmartScreen の警告が出る場合がある。`docs/windows_setup.md`)。暗号化はしない(BitLocker を案内)。
- CI の外部アクションは、コミットの SHA で固定した(`ci.yml` と新しい 2 つのワークフロー)。`fetch_fonts` は固定コミット + SHA256 の照合にした(`docs/trial_ci_build.md`)。この RC のビルド(コンテナ)は、固定前のフォントで作った(SHA256 は、固定後のものと一致を確認済み)。
- 試用版は CPU 版(GPU は `docs/phase6_gpu_design.md`)。

## 6. 検証の手順(受け取った側が、同じ束であることを確かめる)

```sh
sha256sum yayoi-ocr-trial-0.1.0.zip                      # 上の SHA256 と比べる(別経路で受け取った値と)
python scripts/verify_trial_bundle.py yayoi-ocr-trial-0.1.0.zip --sha256 <別経路で受け取った SHA256>
```

## 7. 重みの作り直し(コンテナが回収されたとき)

束・重み・合成データ・拒否リストは `$YAYOI_PRIVATE_DATA`(開発のコンテナの中。リポジトリの外)にだけあり、コンテナが回収されると失われる。リポジトリ(公開)に残るのは、コード・文書・lock だけ。
ここに、**残るものだけから重みを作り直す手順・所要時間・再現性の確認結果**を記す。公開はしない(作り直した束も、director の指示があるまで出さない)。

### 7.1 必要なもの

| 必要なもの | 入手 | 備考 |
|---|---|---|
| リポジトリのコード | git(ブランチ `claude/handwritten-document-ocr-sgf76h`) | 重みに関わる `src/yayoi_ocr/synth/`・`src/yayoi_ocr/recognize/`・`scripts/` を、RC のビルド元(`4651b73`)から変えていないこと(`git diff 4651b73 HEAD -- src scripts` で確認)。変えた場合、ビット一致は保証されない |
| Python と依存(作成時: Python 3.13.17・torch 2.15.0(CPU 版)・numpy 2.5.3・Pillow 12.3.0・opencv-python-headless 5.0.0.93・onnx 1.23.2・onnxruntime 1.29.0・scipy 1.18.1) | `pip install -e ".[dev,ml,ml-dev]"` | **数値の再現は、版・CPU・スレッド数に依存しうる**(§7.3)。作成時の環境は Linux・Intel Xeon 2.1 GHz・4 コア・RAM 15 GB |
| 合成帳票用の手書き風フォント(OFL・14 ファイル) | `python scripts/fetch_fonts.py --ref <fonts-pin.txt の ref> --verify requirements/fonts.sha256`(ネットワーク。google/fonts を固定コミットで取得し、全ファイルの SHA256 を照合する) | 取得したファイルの SHA256 を `docs/fonts.md` の表と突き合わせる(2026-10-31 に再取得して **14/14 一致**を確認。一致しないと、データはビット単位では再生成できない) |
| NDLOCR-Lite の ONNX(固定コミット + SHA256 検証) | `python scripts/fetch_models.py` | 基本の重みへの変換は、ビット単位で一致する(§7.3) |
| 拒否リスト `$YAYOI_PRIVATE_DATA/denylist.txt` | **リポジトリにない**(実データの語を含むため)。控えは作らない(下の方針) | `scripts/check_no_private_data.py` と `scripts/model_provenance.py`(C4・C6)が使う。**重みの作成そのものには要らない**が、無いと、混入の検査とコミット前のフックが働かない |
| 実サンプル | 不要 | 重みの作成には使っていない(`docs/model_provenance.md` §4) |

- **拒否リストの方針(director の判断)**: 拒否リストは、実データの語を外に出さないため、別の経路に控えを作らない。コンテナが回収されて失われた場合は、それを受け入れ、`scripts/check_no_private_data.py` の拡張子・バイナリ(画像・PDF・整数列)の検査と、`scripts/model_provenance.py` の残りの検査(C1〜C3・C5)で運用する(今後の開発では実データを一切読まないので、混入のリスクは低いと判断)。

### 7.2 手順(順序に依存関係がある。所要時間は 4 コア・RAM 15 GB の CPU 機)

`D=$YAYOI_PRIVATE_DATA/datasets`・`M=$YAYOI_PRIVATE_DATA/models`・`E=$YAYOI_PRIVATE_DATA/evals` とする。コマンドの全文は `docs/model_provenance.md` §2・§3。

| # | 作業 | コマンド(要点) | 入力 | 出力 | 所要時間(目安) |
|---|---|---|---|---|---|
| 1 | フォントと基本の重み | `scripts/fetch_fonts.py`、`scripts/fetch_models.py` | ネットワーク | `fonts/`・`models/*.pt`(基本の重み)・`NDLmoji.yaml` | 1 分 |
| 2 | 合成データ(5 組・7 コマンド) | `python3 -I scripts/build_synth_cells.py --jobs 3 --out $D/<名前> --n … --preset … --seed0 … --writer0 … --font-exclude yusei klee`(`model_provenance.md` §3) | フォント | `train` 500・`train_gen` 300・`calib` 100・`calib500` 500・`calib_gen` 200 ページ | 約 1.6 時間(実測 3.6 秒/ページ × 1,600 ページ。`--jobs 3`) |
| 3 | セル種別分類器 | `python3 -I scripts/train_celltype.py --train $D/train --calib $D/calib --out $M/celltype.pt --epochs 8 --threads 4` | 2 の `train`・`calib` | `celltype.pt` | 約 17 分(実測) |
| 4 | 適応済みの重み | `python3 -I scripts/train_adapt.py --train $D/train $D/train_gen --out-dir $M/adapted --l2sp 0.1 --epochs 3 --threads 3` | 1 の基本の重み・2 | `adapted/parseq-ndl-24x256-30-tiny.pt`・`adapted/parseq-ndl-24x384-50-tiny.pt` | 約 27 分(実測。幅 256 が約 11 分、幅 384 が約 16 分) |
| 5 | リハーサルの束 | `python3 -I scripts/build_rehearsal.py --data $D/train $D/train_gen --out $M/rehearsal.npz` | 2 | `rehearsal.npz` | 5 秒 |
| 6 | 較正のダンプ(4 つ) | `python3 -I scripts/eval_recognize.py --data $D/calib500 --classifier $M/celltype.pt [--models-dir $M/adapted] --threads 3 --dump $E/calib500_{base,adapted}.jsonl`、`$D/calib_gen` も同様 → `calib_gen_{base,adapted}.jsonl` | 3・4・2 | ダンプ 4 つ | 約 1.5 時間(実測 約 4 秒/ページ × 700 ページ × 2 組(基本・適応済み)) |
| 7 | 確信度モデル(2 つ) | `python3 -I scripts/fit_confidence.py --dump …`(`model_provenance.md` §3 の 2 コマンド) | 6 | `confidence.json`・`adapted/confidence.json` | 25 秒 |

- 4 と 5 は 2 だけに依存し、3 と並行できる(CPU が 4 コアなので、並行すると互いに遅くなる)。6 は 3 と(適応済みの方は)4 が要る。
- 所要時間: 手順 2 は、アイドルの CPU で 30 ページを 109 秒で作った実測(3.6 秒/ページ)から 1,600 ページ分を、3・4 は再学習の実測、6 は 10 ページの実測からの外挿。逐次で合計 **4 時間弱**(約 1.6 時間 + 17 分 + 27 分 + 1.5 時間)。作業記録(`model_provenance.md` §2 の実行時刻。ほかの作業と並行していた)の実績より速い。
- 学習・ダンプのスレッド数(`--threads`)は、**上の値に揃える**(§7.3)。
- **確信度の較正の余裕(`--margins`)は、CI のビルドと、ここ(コンテナでの作り直し)で違う**: `docs/model_provenance.md` §3 の記録は、コンテナで作った RC の値(`--margin 0.6 --margins amount=0.4`)。`scripts/trial_ci.py` が実行する CI のビルドは、科目も 0.3(`--margins account=0.3,amount=0.4`)。コンテナで作り直す重みも配るなら、`trial_ci.py run fit` と同じ値にすること(`docs/trial_ci_build.md` §4)。
- `docs/model_provenance.md` §3 の較正のダンプのコマンドには `--threads` が書かれていない。**§7.2 の `--threads 3` を使う**(`model_provenance.md` は束に入る文書なので、次に束を作り直すときに追記する)。

### 7.3 再現性の確認(2026-10-31、RC のビルド元と同じコードで実施。結果は SHA256 / 行の一致)

| 成果物 | 確認の内容 | 結果 |
|---|---|---|
| 基本の重み(`*.pt`) | ONNX から再変換(`model_provenance.py --reconvert` の C5) | **ビット一致**(2/2) |
| 合成データ | 7 つのレシピ(`train` の 3 つ・`calib`・`calib500`・`train_gen`・`calib_gen`)から、各 3〜6 ページ(計 24 ページ)を再生成して、元のファイルと SHA256 を比較 | **ビット一致**(24/24)。全ページの再生成は未実施(生成器は seed ごとに決定的で、ページは互いに独立) |
| `rehearsal.npz` | 手順 5 で作り直し | **ビット一致**(`bd10074a…`)。5 秒 |
| `celltype.pt` | 手順 3 で再学習(`--threads 4`) | **ビット一致**(`5cbd43b5…`)。実測 16 分 43 秒 |
| 適応済みの重み 2 つ | 手順 4 で再学習(`--threads 3`) | **ビット一致**(2/2: `1f3f5b48…`・`ab6d17ac…`)。実測 27 分(幅 256 が約 11 分、幅 384 が約 16 分) |
| 較正のダンプ | 手順 6 を `calib500` の先頭 3 ページ(348 セル)と 10 ページ(1,160 セル。基本・適応済みの両方)で再実行して、既存のダンプの先頭と比較 | `--threads 3` で**全行一致**(10 ページ × 2)。`--threads` が 2・4・既定(0)だと、セル種別の確率(`cls_probs`)が 1e-9 程度ずれて、348 行中 313 行が一致しない(判定の値・要確認・正誤は同じ)。**ダンプは `--threads 3` で作る**。全ページの再実行は未実施 |
| 確信度モデル 2 つ | 既存のダンプに手順 7 を再実行 | **ビット一致**(`5204f5d5…`・`9741f716…`)。25 秒 |

→ 結論: **同じ CPU(上の確認は Intel Xeon 2.10 GHz)・同じコード・同じフォント・同じ依存・同じスレッド数なら**、合成データ・セル種別分類器・リハーサルの束・確信度の当てはめは、ビット単位で作り直せる。
全ページ規模の再生成(データ 1,600 ページ・ダンプ 700 ページ × 2)は未実施だが、ページ単位で決定的な処理の抜き取りが一致している。

**CPU が違うと一致しない**(2026-10-31 に確認。コンテナの再起動で、CPU が Xeon 2.10 GHz から Xeon 2.80 GHz(AVX-512 あり)に変わったあと、同じコード・同じフォントで再生成した合成データが、元のデータと**画素レベルで一致しなかった**:
`train` の先頭 4 ページのうち 3 ページ。ページの内容〔seed・書き手・フォント・セル数〕は同じで、画像の数値だけが違う)。画像処理(OpenCV)の演算が CPU の命令セットで変わるためと見られる。
同じ CPU の中では、並列数(`--jobs`)を変えても、シャードに分けて(seed と書き手 ID を同じオフセットでずらして)作っても、同じページができる。
したがって、別の CPU(GitHub Actions のランナーを含む)で作り直した重みは、開発のコンテナの重みと **SHA256 が一致しない**。その場合は、次の §7.4 の「一致しなかった場合」に進む。

### 7.4 作り直したあとにすること

1. 重みを `$M`(と `$M/adapted/`)に置き、`python scripts/model_provenance.py --reconvert` を実行する(C1〜C6・SHA256 の一覧。拒否リストが要る)。
2. 一覧の SHA256 が `docs/model_provenance.md` §2 の表と**一致した**なら、重みは同じ(再ビルドした zip は、zip の項目の時刻が入るので SHA256 が変わる)。
3. **一致しなかった**なら(版・CPU の違い等):
   - `docs/model_provenance.md` §2 の SHA256 の表と、「実行したスクリプトとコミット」の表(日時・コミット)を更新する(束に入る文書なので、束も作り直す)。
   - 重みの質を確かめ直す: `docs/recognition.md` の最終評価(合成の評価用データでの列別の文字・金額の正解率と、要確認の判定の見逃し率・要確認率)を再実行し、以前の値(`docs/recognition.md`・`docs/phase4_method_validation.md`)と同程度であること(摘要の文字一致率が下がっていないこと、見逃し率が列別の目標以下であること)を確認する。
4. 束を作る: `cd web && npm ci && npm run build` → `python scripts/make_trial_bundle.py` → `python scripts/verify_trial_bundle.py <zip> --sha256 <別経路で伝える値>`(B1〜B7)。
5. 煙試験: 束の `models/` を取り出し、`python scripts/trial_smoke.py --python <python> --models <取り出した models> --data-dir <その親>`(§4 と同じ)。
6. **新しい zip の SHA256 を director に報告する**。`docs/trial_rc.md` §1・§8 を更新してコミットする(コミットは、束に入るファイルを変えない)。

## 8. 束に含まれる全ファイル(パス・大きさ・SHA256。`MANIFEST.sha256` 自身の行を除き、`MANIFEST.sha256` と同じ値)

| パス | バイト数 | SHA256 |
|---|---:|---|
| `MANIFEST.sha256` | 11,612 | `83e8563c0fb73dc424889b4f8595c8adb0647d2f562c08597f793552d16788d4` |
| `NOTICE.txt` | 1,346 | `ccf8ff8dbeab15462b2a36952501ff3c8a52403bf01ba1e047c0b98c1557a6ec` |
| `README.txt` | 2,237 | `0900f002b31532e0b5e7f9d22b3353767002bf28b3f38d5534a04265b277888a` |
| `licenses/LICENSE-parseq-Apache-2.0.txt` | 11,358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `licenses/npm/js-tokens-4.0.0/LICENSE` | 1,103 | `2213d91c606205c71eb051a199478cdc2adde945893404d7f1421436dd6d5cc1` |
| `licenses/npm/loose-envify-1.4.0/LICENSE` | 1,100 | `4eb7543b08d955a6d23fcc224601d43ff566e775be918805e26210d7f6eb4893` |
| `licenses/npm/react-18.3.1/LICENSE` | 1,086 | `52412d7bc7ce4157ea628bbaacb8829e0a9cb3c58f57f99176126bc8cf2bfc85` |
| `licenses/npm/react-dom-18.3.1/LICENSE` | 1,086 | `52412d7bc7ce4157ea628bbaacb8829e0a9cb3c58f57f99176126bc8cf2bfc85` |
| `licenses/npm/scheduler-0.23.2/LICENSE` | 1,086 | `52412d7bc7ce4157ea628bbaacb8829e0a9cb3c58f57f99176126bc8cf2bfc85` |
| `licenses/pip/annotated-doc-0.0.5/LICENSE` | 1,086 | `fff170779a6acbf65abdb405c087f1cee1786691e4a96a4034517e4a504a0cdf` |
| `licenses/pip/annotated-types-0.8.0/LICENSE` | 1,083 | `fe1049884b1a0d9342901e88e07f32925d24b3121d9972b6a6805fb9824b095d` |
| `licenses/pip/anyio-4.15.1/LICENSE` | 1,081 | `5361ac9dc58f2ef5fd2e9b09c68297c17f04950909bbc8023bdb82eacf22c2b0` |
| `licenses/pip/bottle-0.13.4/LICENSE` | 1,066 | `af150257b1f7cce92b6207854f2e8f90ea9ae52d7cac758d13a47bd12e2a2443` |
| `licenses/pip/cffi-2.1.1/LICENSE` | 1,123 | `5ba24ddc57067f9249add644c3afc41a5d6dc37e23433ef759d95df370b0af63` |
| `licenses/pip/click-8.5.0/LICENSE.txt` | 1,475 | `9a8ad106a394e853bfe21f42f4e72d592819a22805d991b5f3275029292b658d` |
| `licenses/pip/clr-loader-0.3.1/LICENSE` | 1,079 | `dd6bdca10cdc3b24c6b0b6cef5fd611b7feb6917f127833283ddcdb310d3b453` |
| `licenses/pip/fastapi-0.162.0/LICENSE` | 1,086 | `4ec89ffc81485b97fec584b2d4a961032eeffe834453894fd9c1274906cc744e` |
| `licenses/pip/filelock-4.2.5/LICENSE` | 1,088 | `608c89d5060ae9921adccf3236695bc654a9946e12323ef6c021dfa04e294d48` |
| `licenses/pip/fsspec-2026.9.0/LICENSE` | 1,513 | `2dc35496ce53a7307915c008a844aad53e772b49b34cdd00445067691407ec94` |
| `licenses/pip/h11-0.16.0/LICENSE.txt` | 1,124 | `37db5bb85926db28a427a25867f10b1232003aea1be69ccb851138adb8e6f361` |
| `licenses/pip/idna-3.20/LICENSE.md` | 1,541 | `1a9a4f0e3d479a27240ddd59a9137a66ab4a0f9dfdc8ca6188cc0bfd85187f04` |
| `licenses/pip/jinja2-3.1.6/LICENSE.txt` | 1,475 | `3b49dcee4105eb37bac10faf1be260408fe85d252b8e9df2e0979fc1e094437b` |
| `licenses/pip/markupsafe-3.0.4/LICENSE.txt` | 1,503 | `4631ec0db5fd90a547e336817264c6798214338146f8ac94b4a57f96ee8c9ec4` |
| `licenses/pip/mpmath-1.3.0/LICENSE` | 1,537 | `c26cae81da4508e5e249985777a33821f183223ebb74d7f8cfbf90fe7eef2fb7` |
| `licenses/pip/networkx-3.6.1/LICENSE.txt` | 1,763 | `3cf7c3a179d817b23b8b8568211769fefdf93797d207d984b8b2fa24d0220b33` |
| `licenses/pip/numpy-2.4.6/COPYING` | 704 | `aea41976c720f91bd19725c16916a1ace40eb1ea42620d340e587a5de4bbcea0` |
| `licenses/pip/numpy-2.4.6/LICENSE` | 1,573 | `585483905b13d4b42f800c55ba99e89d6e2e33ccd5f503e201c07531d148ff41` |
| `licenses/pip/numpy-2.4.6/LICENSE.md` | 1,542 | `a732977853530066790c42817087f4829b518dd98d340c916acbfd227e39d3cb` |
| `licenses/pip/numpy-2.4.6/LICENSE.txt` | 1,039 | `d544761558b510866c21f7e8a2d5716fca76e0b98117e3b6b86314db7244d150` |
| `licenses/pip/opencv-python-headless-5.0.0.93/LICENSE-3RD-PARTY.txt` | 183,251 | `c1d60169b55cee56452b227c2fd2a7b9ddda3dd3741065f1c6201072be73fadb` |
| `licenses/pip/opencv-python-headless-5.0.0.93/LICENSE.txt` | 1,090 | `edef0fac1eb08d29d34563f724742e078da2513e196133a12c6ad9ff01e26107` |
| `licenses/pip/opentelemetry-api-1.45.2/LICENSE` | 11,357 | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| `licenses/pip/pillow-13.0.0/LICENSE` | 78,481 | `988ffe8ab3480b0f4e9a0ab2702bffd73d9fb6b47198043fb3ba101a62fac236` |
| `licenses/pip/pycparser-3.12/LICENSE` | 1,543 | `0a9e78156ae8b96caf1b4f11ac2df424b83de29736f9e757630548a6272fdf90` |
| `licenses/pip/pydantic-2.14.2/LICENSE` | 1,129 | `a9e186f3ca16b5eef84318e7a701721351a00cb7b8ae3a4394b67b49e3529ef3` |
| `licenses/pip/pydantic-core-2.50.2/LICENSE` | 1,101 | `fbe7f615f18d135c07000e5a86325db27f42c7729699ef166ac1f7a3f4586bf4` |
| `licenses/pip/pythonnet-3.2.1/AUTHORS.md` | 5,096 | `65a13631f078bed51e9a4033dea257e57a7f23097f3003b1ce1ba1c8a9136c93` |
| `licenses/pip/pythonnet-3.2.1/LICENSE` | 1,104 | `e1dc3a6ff017a6800dc27a7220500f512b95fbcd996d38ebbbff85c693245369` |
| `licenses/pip/pywebview-6.2.1/LICENSE` | 1,518 | `4a988dd3598832cd3653de20dc33cb677d0fb53ab5551c879ca31280ae653675` |
| `licenses/pip/pyyaml-6.0.3/LICENSE` | 1,101 | `8d3928f9dc4490fd635707cb88eb26bd764102a7282954307d3e5167a577e8a4` |
| `licenses/pip/setuptools-84.2.1/LICENSE` | 1,023 | `86da0f01aeae46348a3c3d465195dc1ceccde79f79e87769a64b8da04b2a4741` |
| `licenses/pip/starlette-1.10.0/LICENSE.md` | 1,518 | `dcb95677a02240243187e964f941847d19b17821cf99e5afae684fab328c19bf` |
| `licenses/pip/sympy-1.14.0/AUTHORS` | 55,779 | `82b1038521b0696c63c90b31ed4a0a2c9c4c16871c648b157b07ca44b6f18f8d` |
| `licenses/pip/sympy-1.14.0/LICENSE` | 7,885 | `07a5e9819f727b4986ad2829c7a29a6320d42575f720eb24d71b7fef573a0286` |
| `licenses/pip/torch-2.15.0/LICENSE` | 1,098 | `7aab26dd7e0e6661accc1fd9fc19410e01b466aa0ac36ea668b2013152a8ca23` |
| `licenses/pip/torch-2.15.0/LICENSE-MIT.txt` | 1,137 | `7eacf2f44274c7a788870af113765e7504a9aec1d14401a96bcce4a0695c6686` |
| `licenses/pip/torch-2.15.0/LICENSE-extra` | 1,639 | `ffb82250d2784f2e1a1590dc96243429bc3254ca345238450cf9cb8245e2a760` |
| `licenses/pip/torch-2.15.0/LICENSE.Apache-2.0` | 11,558 | `1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6` |
| `licenses/pip/torch-2.15.0/LICENSE.BSD` | 1,530 | `853617931515e363f5277679c5f9d18b191d63a313e628e63961d535478fc6e0` |
| `licenses/pip/torch-2.15.0/LICENSE.MIT` | 1,097 | `b9715282b2f1db1359a919bb3a542dd4362fd9e38ae72d421367eb11abe62eec` |
| `licenses/pip/torch-2.15.0/LICENSE.md` | 1,523 | `8db000e158c162e7249787852d8c69764706502ccee725c06641b31c1e95c8af` |
| `licenses/pip/torch-2.15.0/LICENSE.txt` | 1,565 | `a51531fad20e44cae492c601bc0997d961c17a74f25f9473ff62eb70e108277d` |
| `licenses/pip/torch-2.15.0/LICENSE.uvw` | 1,102 | `de585e354ff6b8dfb25196f7b9f0f8282e2f1334a92e2f4eb65edee3581c940d` |
| `licenses/pip/typing-extensions-4.16.0/LICENSE` | 13,936 | `3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf` |
| `licenses/pip/typing-inspection-0.4.4/LICENSE` | 1,090 | `804b59b25f2c31bd278f9202a19ae49a3945aa2664387e2d0a128c7cacc61ec3` |
| `licenses/pip/uvicorn-0.54.0/LICENSE.md` | 1,526 | `efe1acf3e62fb99c288b0ec73e5a773b7268ef4320fe757ea994214e4b63c371` |
| `licenses/python-LICENSE.txt` | 25,066 | `886a0ead2d89030ee62dbff52b04e47ab91998341295bb9c56fb952b4e081c7a` |
| `manual.html` | 16,626 | `44fdfdcc56cddced4208f8a500700d43731c23c13fedecffb272c94b296792b1` |
| `model_provenance.md` | 17,822 | `f1d598a704ffa1d144d0963ba73795382c060fb35a3d3162dd6ec60461c0b2f6` |
| `models/NDLmoji.yaml` | 42,434 | `f6ad5a2de444b495155866af811cf1a98309dcae3225db802767ea531a2dc529` |
| `models/adapted/confidence.json` | 5,677 | `9741f7166825713e08105cb3b0605eb7cd39fd55bac185e36ea6be17716a6c4a` |
| `models/adapted/parseq-ndl-24x256-30-tiny.pt` | 35,056,731 | `1f3f5b4861c54d01b71e24742cdbe8c4fee02ce2872b33ae45c303a862445d8a` |
| `models/adapted/parseq-ndl-24x384-50-tiny.pt` | 35,145,819 | `ab6d17ac0f0970451d5d6c05a9f39e39ad88f88f90c23dcea181bfc41a0507a9` |
| `models/celltype.pt` | 742,253 | `5cbd43b5d018c566031b48f04ce8380194bc711e188340432fca387b4f65d095` |
| `models/confidence.json` | 5,672 | `5204f5d5f8d379017cee41a5397129ae3090240128151b587b425804811e6fcb` |
| `models/parseq-ndl-24x256-30-tiny.pt` | 35,056,539 | `fdb63f461191878f4fdd5b164b99f7db36648517f078d6334083b5e5a1489fc1` |
| `models/parseq-ndl-24x384-50-tiny.pt` | 35,145,627 | `42ec07c48e9eacbb02047c257269576cb46c9d4e535c250a5abac065ba8ef97a` |
| `models/rehearsal.npz` | 22,770,692 | `bd10074a7ae599369b84b1f4738704f1780f5a11c005238d0267ea4ba674a9da` |
| `python-standalone.pin` | 569 | `89836c6f8c462cca8c6a628382adb776fc9b50b07d5fb3eac4ff187f039986f7` |
| `python-standalone.tar.gz` | 25,128,375 | `67215a7bf1643e33d01b17ee3f1e91a8b0b39163746855ee02594bf0608f68b4` |
| `requirements-lock.txt` | 62,999 | `5490428a3c75a2f740bf36bb10460464d3498c2ee237da0842bc491685a4610f` |
| `setup.ps1` | 11,347 | `20fd4df1db7cfe693ddac1e471758cec0bc2df165e553de31abceef6af25a49c` |
| `start-yayoi-ocr.cmd` | 457 | `c3dd86787feb525c05b700cbbdb2706d241e48a8b2469f1cc52e3d268b9431e9` |
| `uninstall.ps1` | 7,569 | `7a35f46357868c0bfae5b42dd39c919244e5225d1d20556fc458cdc05c2c9b03` |
| `user_manual.md` | 13,898 | `54708e4418c6b73e378d1653bed61822fc797c478e3d6e9f1d9951abceb980ca` |
| `wheels/annotated_doc-0.0.5-py3-none-any.whl` | 5,302 | `117bac03a25ede5df5440e855b32d556049ca169ead221505badf432fed4b101` |
| `wheels/annotated_types-0.8.0-py3-none-any.whl` | 13,427 | `f072f4d804ea359e4eaf198b1af7a8b0943881a87f31bb764f8bf219bb9419e0` |
| `wheels/anyio-4.15.1-py3-none-any.whl` | 132,079 | `6152fdbbf9a77fdec97731721bebf7c4c44f7c29b424b0065826173efc7ed101` |
| `wheels/bottle-0.13.4-py2.py3-none-any.whl` | 103,807 | `045684fbd2764eac9cdeb824861d1551d113e8b683d8d26e296898d3dd99a12e` |
| `wheels/cffi-2.1.1-cp311-cp311-win_amd64.whl` | 185,096 | `42f6930c31dc7f50732c9ae793c2786c7b6b044195967bbdde40bb9be81c4cc0` |
| `wheels/click-8.5.0-py3-none-any.whl` | 125,251 | `255bc9599cf7748b4b1a446ccc735421bd08a2ae529a8b88597d3de5664ee360` |
| `wheels/clr_loader-0.3.1-py3-none-any.whl` | 55,730 | `cbad189de20d202a7d621956b0fc38049e13c9bf7ca2923441eff725cd121aa1` |
| `wheels/fastapi-0.162.0-py3-none-any.whl` | 148,912 | `3785963de305dd09912426cb81f2d230fce98ab3521cdfe97265c8d14983128c` |
| `wheels/filelock-4.2.5-py3-none-any.whl` | 163,877 | `46ce42d27e5cdde09ee7197349663e97cb106679bd9e75db3a09679fbc7aa1a8` |
| `wheels/fsspec-2026.9.0-py3-none-any.whl` | 221,738 | `8dd6e646e99ea382bd85f97a45e6b526a442d79423a7dc673f1e2756d05fcb5f` |
| `wheels/h11-0.16.0-py3-none-any.whl` | 37,515 | `63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86` |
| `wheels/idna-3.20-py3-none-any.whl` | 69,583 | `ab7ae7122974553370f0bdb919e1a960b2cd1bc1ef0276416d896db81c14582c` |
| `wheels/jinja2-3.1.6-py3-none-any.whl` | 134,899 | `85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67` |
| `wheels/markupsafe-3.0.4-cp311-cp311-win_amd64.whl` | 14,328 | `fdb4ca07ab75ffadab4a8b135ad59cdbb3156b99310f3d565370da74a15d6bd3` |
| `wheels/mpmath-1.3.0-py3-none-any.whl` | 536,198 | `a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c` |
| `wheels/networkx-3.6.1-py3-none-any.whl` | 2,068,504 | `d47fbf302e7d9cbbb9e2555a0d267983d2aa476bac30e90dfbe5669bd57f3762` |
| `wheels/numpy-2.4.6-cp311-cp311-win_amd64.whl` | 12,608,406 | `1e254a00cdf42b1e4d5b3d68d33af63268d41340d8885df2ab6470f2e1500147` |
| `wheels/opencv_python_headless-5.0.0.93-cp37-abi3-win_amd64.whl` | 43,825,962 | `829717b6a95554f273e49e357cee3b3a2a26b6f4842fbc1bed2b45bdd8f87e0e` |
| `wheels/opentelemetry_api-1.45.2-py3-none-any.whl` | 60,335 | `2cbd737377e710ee740d261c94e93fa0f08e6d037c18f17513cc0a2d94479a34` |
| `wheels/pillow-13.0.0-cp311-cp311-win_amd64.whl` | 7,278,471 | `d7710eb676e9bb68965cb045be0d333b4c0d7f9dfbaa7fbf121385b1588683d3` |
| `wheels/proxy_tools-0.1.0-py3-none-any.whl` | 2,935 | `dad7e5b352f42c5833f5b1dbb3b50d43c70f1c6054144d1dfd2f957d060f710c` |
| `wheels/pycparser-3.12-py3-none-any.whl` | 62,600 | `a076f3f489b39b43ed13563a1dafe42447445bec2e294154ec5c57fcdda4c85e` |
| `wheels/pydantic-2.14.2-py3-none-any.whl` | 490,403 | `7efbf7507b937931cd32a5755acb00ea1cf5ccb1d4e8067a5850d5069e382118` |
| `wheels/pydantic_core-2.50.2-cp311-cp311-win_amd64.whl` | 2,238,388 | `b958c217c6e56bef261f516e4caf5d490b079ad6a428752afada5a764cca6c71` |
| `wheels/pypdfium2-5.14.0-py3-none-win_amd64.whl` | 3,947,280 | `149fd5c6397b8df8bf7911a93506eff0be874f877afe7ac936cf5d37d21a6a06` |
| `wheels/pythonnet-3.2.1-cp311.cp312.cp313.cp314.cp315-none-any.whl` | 222,572 | `dd5668b74b0c903d9a24e1ba86f728fc490f8e919d56b3c24bf06739b9ee2b7f` |
| `wheels/pywebview-6.2.1-py3-none-any.whl` | 525,463 | `9d07275f53894ab4d5e2e0e996227193e7187dec276d9b624dccbce029216b46` |
| `wheels/pyyaml-6.0.3-cp311-cp311-win_amd64.whl` | 158,763 | `9f3bfb4965eb874431221a3ff3fdcddc7e74e3b07799e0e84ca4a0f867d449bf` |
| `wheels/setuptools-84.2.1-py3-none-any.whl` | 876,263 | `ca140b45d5325a9478f880e351484f62fef15f336462e449ca0c03c873d589c2` |
| `wheels/starlette-1.10.0-py3-none-any.whl` | 80,571 | `478c0ab3b4ced6e82719849c118fa406f0278a3499c1a898cd15cd31bdc38ce4` |
| `wheels/sympy-1.14.0-py3-none-any.whl` | 6,299,353 | `e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5` |
| `wheels/torch-2.15.0-cp311-cp311-win_amd64.whl` | 126,720,349 | `d2c5938f593a47cf6d151a16d130a3cc3f0c105df1b905a571efc00b01e38c1f` |
| `wheels/typing_extensions-4.16.0-py3-none-any.whl` | 45,571 | `481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8` |
| `wheels/typing_inspection-0.4.4-py3-none-any.whl` | 14,750 | `65b8397ba37ccbce054456aaccddfc91e6e3083c92824df348d96ca832f3f147` |
| `wheels/uvicorn-0.54.0-py3-none-any.whl` | 87,427 | `505bdb0f318731d45f1f712071fc781a8981f6847a31c902c9f5e652d4f67faf` |
| `wheels/yayoi_ocr-0.1.0-py3-none-any.whl` | 382,023 | `1fa0b5d11e24df275a40d9058a5c131c1fb1e93642ba83ffe68bb222d7172536` |
| `windows_checklist.md` | 22,950 | `b2a6dbae71e6ea6c851b4d6313b64b24cd7854794adc9bbf604a147885985c48` |
| `windows_setup.md` | 11,426 | `8a12650df260e147999b4f8aa151bedb23a3f0cb50aba98b15a422e6ac293151` |
| `yayoi_export.md` | 10,757 | `0514534a3d2f167fd27f26587f3f617dc4a9af158d329005841df80b2d985713` |
