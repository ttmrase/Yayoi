# 試用版の配布物を GitHub Actions でビルドして、ユーザーのクリックで公開する

配布するのは、**GitHub Actions でビルドした束**(開発のコンテナの束は参考として残してあるだけ)。トークンをチャットやコンテナで扱わず、**公開の最終操作を、ユーザー本人が GitHub の画面で
「Run workflow」を押すこと**にする(押すことが、公開の承認になる)。副産物として、**配布する束そのもの(適応済みの重みを含む)を Windows のランナーで煙試験できる**
(コンテナの束は、重みを CI に置けないため、できなかった)。

```
.github/trial-build-request を更新して push ─▶ [trial-build.yml]  ビルドと検査だけ(公開しない)
                                                  plan → shards(合成データ 16 並列)→ base-weights / celltype / adapt → dumps(4 並列)→ confidence
                                                       → bundle(出所の文書を実際の重みで作る・束を作る・B1〜B7)→ smoke-windows(この束を Windows で)→ finalize
                                                  成果物: artifact「trial-bundle」(zip・.sha256・出所の文書・ビルドの記録・検査の結果。90 日)
docs/release/trial-v0.1.0.{json,md} をコミット   ─▶ (公開の内容が、リポジトリに残る。押す前に中身を確かめられる)
ユーザーが GitHub の画面で「Trial publish」を Run workflow ─▶ [trial-publish.yml]  verify(読み取り)→ publish(contents: write はここだけ)
                                                  pre-release を作成 → アセットをダウンロードし直して SHA256 を照合 → 一致しなければ Release とタグを削除して失敗
```

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
  この artifact は、GitHub にログインした人だけがダウンロードできる(公開の配布物ではない)。
- **権限**: `contents: read` だけ。公開の操作(タグ・Release)は、このワークフローにはない。

## 2. trial-publish.yml(公開。ユーザーのクリックでのみ動く)

- **起動**: `workflow_dispatch` だけ(入力なし)。GitHub の画面で、リポジトリの **Actions → 「Trial publish」→ 「Run workflow」** を押す。他のトリガー(push・schedule・pull_request など)は付けない。
- **公開の内容は、コミット済みの 2 つのファイルで決まる**(押す前に、中身を読める):
  - `docs/release/trial-v0.1.0.json`: ビルドの `run_id`・zip の `zip_sha256`・`tag`(`trial-v0.1.0`)・`commit`(タグを付けるコミット = ビルドのコミット)・`zip_name`・`title`。
  - `docs/release/trial-v0.1.0.md`: Release の本文(zip の SHA256 を含む)。公開する文章。
- **verify ジョブ**(`contents: read`・`actions: read`): JSON と本文の検査(本文の SHA256 = JSON の値)→ 指定の run が「Trial build」の**成功した**実行で、そのコミットであること → コミットがこのリポジトリの履歴にあること →
  **同じタグ・同じ Release がすでにあれば、上書きせずに失敗** → artifact `trial-bundle` をダウンロードして、SHA256・ビルドの記録(mode が full・run・コミット・検査 OK・Windows の煙試験 OK)・B1〜B7 を再確認。
- **publish ジョブ**(`contents: write` は**このジョブにだけ**): もう一度、既存のタグ・Release がないことと、成果物の一致を確かめる → **pre-release を作成**(タグを指定のコミットに作り、zip と `.sha256` を添付)→
  **アセットをダウンロードし直して SHA256 を照合**(認証ありと、認証なし = 一般の利用者と同じ経路の両方。さらに B1〜B7)→ **一致しなければ、Release とタグを削除して失敗にする**。
  成功すると、実行の要約に Release の URL と SHA256 が出る。
- トークンは `GITHUB_TOKEN` だけ(PAT・secrets は使わない)。

### 公開までの手順

1. 開発側(worker)が `trial-build` を full で動かし、成功を確認する。run_id・zip の SHA256・各検査と煙試験の結果が報告される。
2. 開発側が `docs/release/trial-v0.1.0.json` と `.md` をコミットする(SHA256 と run_id が入る)。内容は `docs/release/` で読める。
3. checker が 2 つのワークフローをレビューする(権限・固定・トークンの扱い・公開の条件)。
4. ユーザーが、上の JSON・本文を確かめたうえで、**GitHub の画面で「Trial publish」を Run workflow する**。
5. 実行が成功したら、Release の URL と SHA256 が実行の要約に出る。**zip の SHA256 は、Release とは別の経路でも、利用者に伝える**。

### 取り下げ

公開後に問題が見つかったら、GitHub の画面で Release を「Delete」し(タグも削除)、`docs/release/` の内容を直して、ビルドからやり直す。trial-publish は既存のタグ・Release があると失敗するので、
同じタグを再利用するときは、先に Release とタグを削除する。

## 3. 限界・注意

- **public リポジトリの Release のアセットは、誰でもダウンロードできる**(zip には、重みと、実サンプルを使った処理の開示つきの `model_provenance.md` が入る)。ユーザーはこの方式を承知のうえで選択している。
- ビルドは、GitHub のランナーの CPU・torch の版で重みを作る。**開発のコンテナの重みとは SHA256 が一致しない**。重みの質(要確認の判定の見逃し率など)は、合成の評価用データでの評価で、同程度であることを
  確かめる必要がある(`docs/trial_rc.md` §7.4 の「一致しなかった場合」)。**この確認は、公開の判断材料として、director が行う**(CI は、重みの質の評価を自動では行わない。束の検査と煙試験は、動くことの確認)。
- Windows の実機(IME・高 DPI・WebView2 の保存ダイアログ・実際の速度・弥生への実取り込み)の確認は、引き続き実機の項目(`docs/windows_checklist.md`)。
- Actions の実行時間・成果物の容量は、GitHub の制限(public リポジトリの標準のランナーは無料枠)に従う。full の artifact は約 400 MB。

## 4. 部品(`scripts/trial_ci.py`・試験)

| 部品 | 役割 |
|---|---|
| `scripts/trial_ci.py plan` | 合成データのシャード・各段階の引数の計画(full / mini) |
| `scripts/trial_ci.py run STAGE` | shard / celltype / adapt / rehearsal / dump / fit の実行(コマンドは `docs/trial_rc.md` §7.2 のとおり)と、段階の記録(CPU・ライブラリの版・所要時間) |
| `scripts/trial_ci.py summary / provenance / augment` | シャードの要約 → 出所の文書とビルドの記録 → zip と検査の結果の書き足し |
| `scripts/trial_ci.py release-spec / release-check` | 公開の内容(JSON と本文)の検査、artifact との照合(trial-publish が使う) |
| `scripts/verify_trial_bundle.py --no-denylist` | 束の検査 B1〜B7(CI では拒否リストなし) |
| `scripts/fetch_fonts.py --ref --verify` | フォントを固定コミットで取得し、SHA256 を照合 |
| `tests/test_trial_ci.py` | 計画(シャードがレシピのページを過不足なく覆う・重ならない)・出所の文書(実際の SHA256)・公開の内容と成果物の照合 |
| `tests/test_workflows.py` | ワークフローの安全の約束(SHA 固定・権限・トリガー・上書きしない・ロールバック) |
