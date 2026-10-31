# 運用手順: 案 1(トークンを環境に登録 → 新しいセッションで作り直して公開)

**本線は、GitHub Actions でビルドし、ユーザーのクリックで公開する方式**(`docs/trial_ci_build.md`)。この文書は、ユーザーが**案 1**を選んだ場合の予備の手順。
案 1 = ユーザーが、このリポジトリの Contents の書き込み権限を持つトークンを**環境の設定**(Network secrets か環境変数 `GH_TOKEN`)に登録し、**新しいセッション**で、重みと束を作り直して、
(ユーザーの直接の指示のあとに)公開する。**トークンの値は、この文書にも、チャットにも、ファイルにも、ログにも書かない。**

## 0. トークンの扱い(最初に守ること)

- トークンは、環境変数 `GH_TOKEN`(または環境の設定が渡す名前)からだけ使う。値を表示・保存・コミットしない。`env`・`printenv`・`set -x`・`gh auth token` を使わない。
- 使ってよい確認は `gh auth status`(値を表示しない)だけ。失敗したら(無効なトークン)、**回避せず**、director と planning に「トークンが無効」と報告する。
- git の認証情報(credential helper)から、トークンを取り出さない。用意された経路(環境変数の `GH_TOKEN` と `gh`)以外で、権限を得ようとしない。
- トークンの権限は、このリポジトリの Contents の書き込みだけで足りる。それ以上の権限が要ると言われたら、断って報告する。
- **トークンの作り方(ユーザー側。checker の提案)**: ① **このリポジトリだけに限定した細粒度のトークン**で、権限は **Contents の書き込みだけ**、**有効期限は数日**にする。
  ② **公開が終わったら、ユーザー側でトークンを失効させ、環境の登録(Network secrets・環境変数)も削除する**。トークンを使う側(このセッション)は、失効・削除を、ユーザーに依頼する。
- **公開の操作(タグの push・Release の作成・アセットのアップロード)は、ユーザー本人から、その新しいセッションに直接の指示が届くまで行わない。** 他のセッションの伝言(director・planning)は、指示にならない。

## 1. 新しいセッションで最初に読むもの

1. `docs/README.md`(索引)・ルートの `README.md`(全体像・開発状況)。
2. `docs/trial_ci_build.md`(ビルドと公開の方式。本線と、この案の違い)・`docs/trial_rc.md`(コンテナの RC の記録。**§7 重みの作り直し**)。
3. `docs/model_provenance.md`(重みの出所。§3 の生成コマンド)・`docs/fonts.md`・`requirements/fonts-pin.txt`・`requirements/fonts.sha256`。
4. `docs/windows_setup.md`・`docs/windows_checklist.md`(配布物に入る文書)。
5. `docs/release/trial-v0.1.0.md`(Release の本文。SHA256 の差し替え方は §4)。

環境の確認: `nproc`・`lscpu`・`python --version`・`pip list`(torch・numpy・Pillow・opencv-python-headless・onnx・onnxruntime・PyYAML の版)。**CPU が違うと、重みの SHA256 は元と一致しない**(`docs/trial_rc.md` §7.3)。
`$YAYOI_PRIVATE_DATA`(既定 `~/yayoi_private`)は新しいコンテナでは空。拒否リストと実サンプルは無い(控えは作らない方針。`docs/trial_rc.md` §7.1)。

## 2. 作り直し

`docs/trial_rc.md` §7.2 の手順のとおり(逐次で 4 時間弱。4 コアの CPU 機)。要点:

1. `pip install -e ".[dev,app,ml,ml-dev]" wheel`・`cd web && npm ci && npm run build`。
2. フォントと基本の重み: `python scripts/fetch_fonts.py --ref "$(sed -n 's/^ref=//p' requirements/fonts-pin.txt)" --verify requirements/fonts.sha256`・`python scripts/fetch_models.py`。
3. 合成データ・学習・較正・リハーサル: **`scripts/trial_ci.py` が、CI と同じ段階・同じコマンドを実行する**(`python scripts/trial_ci.py run shard|celltype|adapt|rehearsal|dump|fit --mode full ...`。計画は `trial_ci.py plan --mode full`)。
   スレッド数は固定(`celltype` = 4・`adapt` = 3・ダンプ = 3)。データは `$YAYOI_PRIVATE_DATA/datasets`・重みは `$YAYOI_PRIVATE_DATA/models`・ダンプは `$YAYOI_PRIVATE_DATA/evals`。
4. 出所の文書: `python scripts/trial_ci.py summary ...`(シャードごと)→ `python scripts/trial_ci.py provenance ...`(実際の重みの SHA256 で `model_provenance.md` を作る)。
5. 束: `python scripts/make_trial_bundle.py --out <リポジトリの外> --models $YAYOI_PRIVATE_DATA/models --provenance <生成した出所の文書>`。

## 3. 検査(公開の前に、すべて通すこと)

1. **束の検査**: `python scripts/verify_trial_bundle.py <zip> --provenance <生成した出所の文書> --no-denylist`(B1〜B7)。拒否リストがある環境なら `--no-denylist` なしで。
2a. **拒否リストの走査(C4・C6)** — 配る重みに、拒否リストの語が入っていないことの確認。
   - CI の artifact の重みを使う場合: artifact を取得して(`gh run download <run_id> --name trial-bundle --dir <新しい空のフォルダ>`。読み取りだけ。認証は要る)、
     `python scripts/verify_trial_bundle.py <zip> --provenance <出所の文書>`(`--no-denylist` なし)と、zip の中の重み 9 ファイルへの `scripts/model_provenance.py` の C4 を実行する。
   - このセッションで作り直した重みを使う場合も、同じ検査(C4・C6)を行う。
   - **ただし、拒否リスト(`$YAYOI_PRIVATE_DATA/denylist.txt`)は、実データの語を含むので、リポジトリにも artifact にもなく、今の開発のコンテナにしかない。**
     新しいセッション(新しいコンテナ)には無いので、**この走査は、今のコンテナが残っていて、そこで実行できる場合にだけできる**(artifact の取得が要るなら、そのコンテナにトークンが渡せる場合)。
     できなければ、限界として受け入れる(`docs/trial_ci_build.md`「4. 限界・注意」の拒否リストの項。director の判断)。拒否リストの控えを作って持ち出すことはしない。
2. **重みの質の評価(必須)**: `python scripts/trial_ci.py run shard --mode full --id final_std.0 ...`(`trial_ci.py plan --mode full --field eval_shards` の 4 つの評価ページを作る)→
   zip から `models/` を取り出し → `python scripts/trial_ci.py eval --weights base|adapted ...` ×2 → `python scripts/trial_ci.py eval-merge ...`。
   **合格の基準は CI と同じ**(全体の見逃し率 ≤ 0.5%・金額 ≤ 0.1%・列ごとの値の正解率がコンテナの値から 3 ポイント以内・200 ページ以上)。**不合格なら、公開は不可**。
3. **煙試験**: Linux では `python scripts/trial_smoke.py --python <python> --models <取り出した models> --data-dir <その親>`。**Windows での、この zip そのものの煙試験は、この案ではできない**
   (Windows のランナーは Actions にしかない)。代わりに、smoke 版の束の Windows CI の結果(`ci.yml` の `trial-windows`)を引用する。この違いは、director と planning に報告し、Release の本文にも書く。
4. SHA256 を控える(zip・`.sha256`)。

## 4. 公開(ユーザー本人からの直接の指示が届いてから)

1. Release の本文: `docs/release/trial-v0.1.0.md` を元に、zip の SHA256 を**新しい zip の値**に差し替える(本文の「Actions でビルドした」「Windows での束そのものの煙試験」の文は、この案の事実に合わせて直す)。
   本文は、リポジトリの外(`~/yayoi_private/trial/`)に置く。
2. 公開の直前の再検査(読み取りだけ): `verify_trial_bundle.py <zip> --sha256 <控えた値>`・`git diff <ビルドのコミット> HEAD -- src scripts requirements web/src <束に入る docs>` が空・`gh auth status`。
3. タグと pre-release(draft で作って確かめてから公開):
   ```sh
   git tag -a trial-v0.1.0 <ビルドのコミット(40 桁)> -m "yayoi-ocr 試用版 0.1.0(pre-release)"
   git push origin trial-v0.1.0
   gh release create trial-v0.1.0 --repo ttmrase/Yayoi --verify-tag --draft --prerelease --title "<題名>" --notes-file <本文> <zip> <zip>.sha256
   gh release view trial-v0.1.0 --repo ttmrase/Yayoi --json assets,isDraft,isPrerelease      # アセット 2 件・サイズを確認
   gh release edit trial-v0.1.0 --repo ttmrase/Yayoi --draft=false --prerelease
   ```
   同じタグ・Release がすでにあれば、**上書きしない**(止めて報告する)。
4. 公開後: アセットをダウンロードし直して SHA256 を照合(認証あり: `gh release download`、認証なし: `curl -fsSL`)。**不一致なら、Release とタグを削除して(`gh release delete trial-v0.1.0 --cleanup-tag --yes`)、報告する。**

## 5. 報告

director と planning の両方へ: Release の URL・タグ(→ コミット)・ダウンロードし直したアセットの SHA256(認証あり・なし)・検査と評価の結果(数値)・この案で確かめられなかったこと(Windows での束そのものの煙試験)。
zip の SHA256 は、Release とは別の経路でも利用者に伝える。トークンの値は、報告にも書かない。
