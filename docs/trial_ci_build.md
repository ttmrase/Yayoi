# 試用版の配布物を GitHub Actions でビルドして、ユーザーのクリックで公開する

配布するのは、**GitHub Actions でビルドした束**(開発のコンテナの束は参考として残してあるだけ)。トークンをチャットやコンテナで扱わず、**公開の最終操作を、ユーザー本人が GitHub の画面で
「Run workflow」を押すこと**にする(押すことが、公開の承認になる)。副産物として、**配布する束そのもの(適応済みの重みを含む)を Windows のランナーで煙試験できる**
(コンテナの束は、重みを CI に置けないため、できなかった)。

```
.github/trial-build-request を更新して push ─▶ [trial-build.yml]  ビルドと検査だけ(公開しない)
                                                  plan → shards(合成データ 16 並列)→ base-weights / celltype / adapt → dumps(4 並列)→ confidence
                                                       → bundle(出所の文書を実際の重みで作る・束を作る・B1〜B7)→ smoke-windows(この束を Windows で)→ finalize
                                                  成果物: artifact「trial-bundle」(zip・.sha256・出所の文書・ビルドの記録・検査の結果。90 日)
.github/trial-eval-request を更新して push      ─▶ [trial-eval.yml]  重みの質の評価(公開の前の必須の確認。公開しない)
                                                  束の重み(基本・適応済み)で、最終ホールドアウト 200 ページ(CI で作って捨てる)を認識 → 合格の基準で判定
                                                  成果物: artifact「trial-eval-result」(zip の SHA256 に結び付けた評価の結果)
docs/release/trial-v0.1.0.{json,md} をコミット   ─▶ (公開の内容が、リポジトリに残る。押す前に中身を確かめられる)
ユーザーが GitHub の画面で「Trial publish」を Run workflow ─▶ [trial-publish.yml]  verify(読み取り)→ publish(contents: write はここだけ)
                                                  pre-release を作成 → アセットをダウンロードし直して SHA256 を照合 → 一致しなければ Release とタグを削除して失敗
```

> **注意: public リポジトリの Actions の artifact は、GitHub のアカウントがあれば誰でもダウンロードできる**(「ログインした人だけ」ではなく、アカウントがあれば誰でも)。
> ビルドの途中の artifact(合成データ・途中の重み)は、その段階が終わった時点で、full の最終の artifact(約 400 MB。適応済みの重みを含む)は、ビルドが終わった時点で、
> ユーザーの最終のクリック(trial-publish)より前に、取得できる状態になる。これは、ユーザーが公開を承認した配布物と同じ内容で、承認の範囲内。
> **承認を要する公開の操作(タグ・Release・アセットの公開、利用者への案内)は、引き続きユーザーのクリックまで行わない。**
> ユーザーが望めば、Actions の画面から artifact を削除できる。**full のビルドの実行・再実行は、director の指示があるときだけ**(mini は、重みが使い物にならない)。

## 1. trial-build.yml(公開しない)

- **起動**: `.github/trial-build-request` を変更して push する(`mode: full` = 配布物 / `mode: mini` = 通し試験)。GitHub の画面の「Run workflow」からも起動できる(`mode` を選ぶ)。
- **mini**: 同じ経路を、ごく小さなデータ(合成データ 84 ページ)・1 エポックで通す。**段階のつなぎ・artifact の受け渡し・束の作成・検査・Windows の煙試験を、短い時間で確かめる用**で、重みは使い物にならない。
  artifact の名前は `trial-bundle-mini`、zip の名前は `…-MINI.zip`。trial-publish は mode が full でない成果物を拒否する。full の前に、必ず mini を通す。
- **full**(配布物): `docs/model_provenance.md` §3 と `docs/trial_rc.md` §7.2 の手順・seed・スレッド数のとおり(`scripts/trial_ci.py` が実行する)。
  合成データ 1,600 ページを 100 ページずつのシャード 16 個に分けて並列に作り(ページごとに seed と書き手 ID が決まるので、同じオフセットでずらせば、1 回で作ったときと同じページになる。同じ環境で確認済み)、
  セル種別分類器(`--threads 4`)・適応済みの重みとリハーサルの束(`--threads 3`)・較正のダンプ 4 つ(`--threads 3`。4 並列)・確信度モデル 2 つを作り、束にする。
  ジョブを分けてあるので、1 つのジョブの上限(6 時間)に当たらない(各ジョブの目安: シャード 約 8 分・celltype 約 20 分・adapt 約 30 分・ダンプ 約 35 分)。
- **入力の固定**: フォントは google/fonts の**コミット**(`requirements/fonts-pin.txt`)で取得し、全 14 ファイルの SHA256 を `requirements/fonts.sha256` と照合する(`scripts/fetch_fonts.py --ref --verify`)。
  NDLOCR-Lite は `e57f7c6`(ONNX の SHA256 を検証)。同梱の Python と wheel は lock(バージョン + ハッシュ)。Linux のビルド環境のライブラリは `requirements/trial-build-linux.txt` でバージョンを固定
  (torch は CPU 版を PyTorch の配布元から、無ければ PyPI から)。外部アクションはすべてコミットの SHA。
- **重みは、開発のコンテナで作ったものとビット単位では一致しない**(CPU が違う。`docs/trial_rc.md` §7.3)。**束の出所の文書(`model_provenance.md`)は、ワークフローの中で、実際に作った重みの SHA256 で作る**
  (`scripts/trial_ci.py provenance`): §2 の SHA256 の表を実際の値にし、冒頭にビルドの記録(run・コミット・CPU・ライブラリの版・各段階のコマンドと所要時間・合成データの要約・検査 C1〜C3)を付ける。
  **C4・C6(拒否リストの走査)は、この束の重みには行っていない**(CI に拒否リストを置けないため)と、文書に明記する。代わりに、入力が合成データだけであること(C1〜C3)と、
  束に画像・PDF・CSV・DB・zip の類が入っていないこと(`verify_trial_bundle.py` の B7。拡張子と、先頭のバイト列〔PNG・JPEG・GIF・PDF・SQLite・TIFF〕。`--no-denylist`)を確かめる。
- **検査**: `verify_trial_bundle.py` の B1〜B7(B4 は、生成した出所の表と照合)→ **Windows のランナーで、この束そのもの**を、展開 → `setup.ps1`(同梱の Python・オフライン・ハッシュ検証)→
  開発用でないモードで起動 → HTTP の煙試験(取り込み〜認識〜確定〜cp932 の出力〜バックアップ〜復元)→ アンインストール(データと学習した世代の保持・削除・別フォルダの拒否)。
- **成果物**(artifact `trial-bundle`。保持 90 日): `yayoi-ocr-trial-0.1.0.zip`・`.sha256`・`model_provenance.generated.md`・`build-info.json`(run・コミット・重みの SHA256・各段階の記録・検査の結果)・`verify-output.txt`・`smoke-windows.json`。
  この artifact は、**GitHub のアカウントがあれば誰でもダウンロードできる**(public リポジトリの Actions の artifact。上の注意)。
- **権限**: `contents: read` だけ。公開の操作(タグ・Release)は、このワークフローにはない。

## 2. trial-eval.yml(重みの質の評価。公開の前の必須の確認。公開しない)

- **起動**: `.github/trial-eval-request`(`mode`・`build_run_id`)を変更して push する。手動でも起動できる。権限は `contents: read`・`actions: read` だけ。
  まず対象の run を `gh api` で取り、**ワークフローのパス(trial-build.yml)・ブランチ(`claude/handwritten-document-ocr-sgf76h`)・イベント(push か workflow_dispatch)・成功**を確かめる(`trial_ci.py run-check`)。
- **評価する重み**: 指定した run の artifact(`trial-bundle`)の **zip の中の `models/`**(実際に配る重み。基本と適応済みの両方。出荷設定の確信度つき)。
- **評価データ**: 学習・較正に一度も使っていない合成ページ = **最終ホールドアウト 200 ページ**(`docs/recognition.md`「確信度の較正」と同じレシピ・同じ seed の範囲。標準語彙・一般語彙 x 未使用の書き手・未知フォント〔評価専用の yusei・klee〕各 50 ページ)。
  CI のジョブの中で作って捨てる(artifact にしない)。ページのメタデータと正解の文字列は、コンテナで作った元の評価データと一致することを確認済み(画像の画素は CPU で変わる)。
- **指標**: 列ごと(日付・科目・摘要・金額)と全体の、値の正解率(空欄を除く)・要確認率・見逃し率(要確認でないのに間違い ÷ 全セル)。`scripts/eval_recognize.py` と同じ処理(`yayoi_ocr.recognize.evaluate`)。
- **合格の基準**(director 指示。点推定。基本の重みと適応済みの重みの**両方**に適用): 全体の見逃し率 ≤ 0.5%・金額の見逃し率 ≤ 0.1%・列ごとの値の正解率が、コンテナでの出荷時の評価
  から 3 ポイントを超えて下がっていない・評価したページが 200 以上。**基準を満たさなければ、公開は不可**(ゲートのステップが run を失敗にする)。
  - 「コンテナでの出荷時の評価」の値は、コンテナで出荷した重み(RC の束)を、元の最終ホールドアウトに、この評価と**同じコード**で当てて測り直した値(`scripts/trial_ci.py` の `BASELINE_ACCURACY`)。
    見逃し数(適応済み: 日付 12・科目 21・摘要 8・金額 2)と要確認率は `docs/recognition.md` の表と一致する。表の「値の正解率」は空欄の扱いが少し違うので、基準には測り直した値を使う。
  - 基準の値(列ごとの値の正解率。空欄を除く): 適応済み = 日付 0.586・科目 0.853・摘要 0.443・金額 0.642 / 基本 = 日付 0.409・科目 0.726・摘要 0.203・金額 0.540。
    コンテナで出荷した重みの見逃し率は、適応済み 0.185%(金額 0.034%)・基本 0.142%(金額 0%)で、基準の中に収まっている。
- **結果**: `eval-result.json`(全指標・基準・各チェックの成否・`passed`・**評価した zip の SHA256**・ビルドの run とコミット)を artifact「`trial-eval-result`」に保存する(90 日)。mini の評価は「`trial-eval-result-mini`」(重みが使い物にならないので、ゲートは働かず、記録だけ)。

## 3. trial-publish.yml(公開。ユーザーのクリックでのみ動く)

- **起動**: `workflow_dispatch` だけ(入力なし)。GitHub の画面で、リポジトリの **Actions → 「Trial publish」→ 「Run workflow」** を押す。他のトリガー(push・schedule・pull_request など)は付けない。
- **公開の内容は、コミット済みの 2 つのファイルで決まる**(押す前に、中身を読める):
  - `docs/release/trial-v0.1.0.json`: ビルドの `run_id`・評価の `eval_run_id`・zip の `zip_sha256`・`tag`(`trial-v0.1.0`)・`commit`(タグを付けるコミット = ビルドのコミット)・`zip_name`・`title`。
  - `docs/release/trial-v0.1.0.md`: Release の本文(zip の SHA256 を含む)。公開する文章。
- **verify ジョブ**(`contents: read`・`actions: read`): JSON と本文の検査(本文の SHA256 = JSON の値)→ ビルドの run が「**trial-build.yml の、`claude/handwritten-document-ocr-sgf76h` ブランチの、push か workflow_dispatch の、成功した実行**」で、
  head のコミットが JSON と同じであること・評価の run が「trial-eval.yml の、同じブランチの、成功した実行」であること → コミットがこのリポジトリの履歴にあること →
  **同じタグ・同じ Release がすでにあれば、上書きせずに失敗** → artifact `trial-bundle` をダウンロードして、SHA256・ビルドの記録(mode が full・run・コミット・検査 OK・Windows の煙試験が**その zip の SHA256 に対して** OK)・B1〜B7 を再確認 →
  artifact `trial-eval-result` をダウンロードして、**評価が、その zip の SHA256 に対するもので、合格であること**(ビルドの run・コミット・評価の run が JSON と一致)を確認。
- **publish ジョブ**(`contents: write` は**このジョブにだけ**): もう一度、既存のタグ・Release がないことと、成果物・評価の一致を確かめる → **pre-release を作成**(タグを指定のコミットに作り、zip と `.sha256` を添付)→
  **アセットをダウンロードし直して SHA256 を照合**(認証ありと、認証なし = 一般の利用者と同じ経路の両方。さらに B1〜B7)→ **一致しなければ、Release とタグを削除して失敗にする**。
  成功すると、実行の要約に Release の URL と SHA256 が出る。
- トークンは `GITHUB_TOKEN` だけ(PAT・secrets は使わない)。

### 公開までの手順(full のビルドの開始は、director の指示を待つ)

1. (director の指示で)開発側(worker)が `trial-build` を full で動かし、成功を確認する。run_id・zip の SHA256・各検査と Windows の煙試験の結果が報告される。ビルドが終わると、artifact は GitHub のアカウントがあれば誰でも取得できる状態になる(上の注意)。
2. 開発側が `trial-eval` を動かし、**合格**を確認する(不合格なら、公開は不可。重みを直して 1 からやり直す)。評価の結果(数値)が報告される。
3. (拒否リストの走査は、行わない。director の判断。下の「4. 限界・注意」の拒否リストの項を参照。ユーザーが案 1〔トークン付きのセッション〕を選んだ場合の走査の手順は、`docs/release/runbook_token_session.md` §3。)
4. 開発側が `docs/release/trial-v0.1.0.json` と `.md` をコミットする(SHA256・run_id・eval_run_id が入る)。
5. checker が 2 つのワークフロー(と、JSON・本文)を再確認する。
6. **ユーザーは、Run workflow を押す前に、`docs/release/trial-v0.1.0.json` の `run_id`・`eval_run_id`・`commit`・`zip_sha256` が、開発側から報告された値と一致すること、
   `docs/release/trial-v0.1.0.md` の本文を、目で確かめる。** JSON は、書き込み権限のある人が書けるので、別の run を指すように書き換えられていないことの確認が、最後の関門になる。
7. ユーザーが、**GitHub の画面で「Trial publish」を Run workflow する**。
8. 実行が成功したら、Release の URL と SHA256 が実行の要約に出る。**zip の SHA256 は、Release とは別の経路でも、利用者に伝える**。

### 取り下げ

公開後に問題が見つかったら、GitHub の画面で Release を「Delete」し(タグも削除)、`docs/release/` の内容を直して、ビルドからやり直す。trial-publish は既存のタグ・Release があると失敗するので、
同じタグを再利用するときは、先に Release とタグを削除する。

## 4. 限界・注意

- **public リポジトリの Release のアセットは、誰でもダウンロードできる**(zip には、重みと、実サンプルを使った処理の開示つきの `model_provenance.md` が入る)。ユーザーはこの方式を承知のうえで選択している。
  **Actions の artifact も、GitHub のアカウントがあれば誰でも取得できる**(上の注意。ユーザーが承認した配布物と同じ内容)。
- ビルドは、GitHub のランナーの CPU・torch の版で重みを作る。**開発のコンテナの重みとは SHA256 が一致しない**(`docs/trial_rc.md` §7.3)。重みの質は、`trial-eval`(上の 2.)で確かめる。
- **拒否リストの走査(C4・C6)について(限界。director の判断で、限界として受け入れる)**: CI に拒否リスト(実データの語)を置けない。また、開発のコンテナには GitHub の認証がなく、
  artifact を取得できない(拒否リストは、実データの語を含むので、リポジトリにも artifact にもなく、開発のコンテナにしかない)。このため、**CI でビルドした重みには C4・C6 を行っていない**。
  受け入れる理由は次のとおり。
  ① **CI の重みは、合成データだけから作られている**: 入力(合成帳票の生成器)が実サンプルに触れない(C1: 実サンプルの置き場を参照するコードは許可リストのみ)・学習と較正のデータの全ページが生成器の署名
  (seed・書き手 ID・OFL フォント名)を持つ(C2)・データセット間で seed・書き手 ID が重ならない(C3)ことを、CI の中で検査している(`model_provenance.md` の §0)。
  ② **開発のコンテナで、同じ手順で作った重み**には、C4(拒否リストの走査)と C6 を実行し、**検出 0 件**だった(`docs/model_provenance.md` §4)。
  ③ **CI の重みの質は、`trial-eval` で確かめる**(上の 2.)。
  束の混入検査 B7 は、画像・PDF・CSV・DB・zip の類(拡張子と、先頭のバイト列)と、アプリの wheel の中身だけで、重みのテンソルは見ない。
  ユーザーが案 1(トークン付きのセッションで作り直して公開)を選んだ場合は、そのセッションで artifact を取得して走査する手順を、`docs/release/runbook_token_session.md` §3(2a)に置いた。
  ただし、拒否リストは今の開発のコンテナにしかないので、**走査できるのは、今のコンテナが残っている場合だけ**。
- **既知の限界・次回の改善(Linux のビルド・評価の依存は、版の固定であって、ハッシュの固定ではない)**: `requirements/trial-build-linux.txt` は constraints(版の固定)で、ハッシュは固定していない。
  Windows の束に入るパッケージ(利用者の PC に入るもの)だけが、ハッシュ付きの lock〔`requirements/trial-win-cp311.txt`〕。
  torch は、PyTorch の CPU 版の配布元から入れる。trial-eval は、ローカル版 `+cpu` を指定し、取れなければ失敗させる(PyPI の別の build に切り替えない)。
  trial-build の Linux のジョブは、まだ「配布元で失敗したら PyPI から入れる」形のまま(実行中の full の run がこの形で動いているため、変えない。director の判断で、full の作り直しはしない)。
  受け入れる理由(director の判断): ① 利用者の PC に入るパッケージは、ハッシュ付きの lock で固定済み。② Linux 側は学習のためだけに使い、生成される重みはテンソルだけ
  (アプリは `torch.load(weights_only=True)` と `np.load(allow_pickle=False)` で読む)なので、依存が改ざんされた場合の影響の経路が限られる。③ 重みの質は trial-eval で確かめる。
  **次に束を作り直すとき(試用版 0.2 など)には、Linux 側もハッシュ付きの lock に揃える。**
- Windows の実機(IME・高 DPI・WebView2 の保存ダイアログ・実際の速度・弥生への実取り込み)の確認は、引き続き実機の項目(`docs/windows_checklist.md`)。
- Actions の実行時間・成果物の容量は、GitHub の制限(public リポジトリの標準のランナーは無料枠)に従う。full の artifact は約 400 MB。

## 5. 部品(`scripts/trial_ci.py`・試験)

| 部品 | 役割 |
|---|---|
| `scripts/trial_ci.py plan` | 合成データのシャード・各段階の引数の計画(full / mini) |
| `scripts/trial_ci.py run STAGE` | shard / celltype / adapt / rehearsal / dump / fit の実行(コマンドは `docs/trial_rc.md` §7.2 のとおり)と、段階の記録(CPU・ライブラリの版・所要時間) |
| `scripts/trial_ci.py summary / provenance / augment` | シャードの要約 → 出所の文書とビルドの記録 → zip と検査の結果の書き足し |
| `scripts/trial_ci.py eval / eval-merge` | 重みの質の評価(最終ホールドアウト)と、合格の基準での判定・zip の SHA256 への結び付け(trial-eval が使う) |
| `scripts/trial_ci.py run-check / release-spec / release-check` | 実行(ワークフロー・ブランチ・イベント・コミット・成功)の確認、公開の内容(JSON と本文)の検査、artifact・評価との照合(trial-eval・trial-publish が使う) |
| `scripts/verify_trial_bundle.py --no-denylist` | 束の検査 B1〜B7(CI では拒否リストなし) |
| `scripts/fetch_fonts.py --ref --verify` | フォントを固定コミットで取得し、SHA256 を照合 |
| `tests/test_trial_ci.py` | 計画(シャードがレシピのページを過不足なく覆う・重ならない)・出所の文書(実際の SHA256)・公開の内容と成果物の照合 |
| `tests/test_workflows.py` | ワークフローの安全の約束(SHA 固定・権限・トリガー・上書きしない・ロールバック) |
