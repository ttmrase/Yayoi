# Windows での試用版セットアップ(インストーラは Phase 6)

試用版は「配布 zip + `setup.ps1`」で入れる。配布 zip は開発者が `scripts/make_trial_bundle.py` で作る(重みを含むので **リポジトリには置かない**)。

## 必要なもの

| もの | 内容 |
|---|---|
| OS | Windows 10 / 11(64 ビット) |
| Python | 3.11〜3.13(64 ビット)。<https://www.python.org/downloads/windows/>。インストール時に「Add python.exe to PATH」にチェック |
| WebView2 ランタイム | Windows 11 は標準で入っている。Windows 10 で無ければ Microsoft の Evergreen Standalone Installer(x64)を入れる。無いとウィンドウが開かない |
| 空き容量 | 約 3 GB(PyTorch の CPU 版・重み・データ) |
| 弥生 | やよいの青色申告 25(デスクトップ版)。出力した CSV を取り込む |

## 手順(利用者)

1. 配布 zip を展開する(`setup.ps1`・`wheels\`・`models\` などが入っている)。
2. PowerShell で、展開したフォルダに移って実行する:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\setup.ps1
   ```

   - 配布物のハッシュ(`MANIFEST.sha256`)を検証し、`%LOCALAPPDATA%\yayoi-ocr` に専用の仮想環境(`venv\`)を作って依存を入れ、重み(`models\`)をコピーし、デスクトップにショートカットを作る。
   - `wheels\` に PyTorch の wheel が入っていれば**完全にオフライン**で入る(`--with-wheels` で作った配布物)。無ければ、セットアップの間だけ pip がネットワークを使う(PyTorch の CPU 版)。
   - アプリ本体は実行時にネットワークを使わない。
3. デスクトップの **yayoi-ocr** から起動する(`%LOCALAPPDATA%\yayoi-ocr\start-yayoi-ocr.cmd`)。コンソールが最小化されて動いている(閉じるとアプリも終わる。エラーはここに出る)。
4. 初回起動の **初期設定** で、支払手段(帳票の「科目」欄に書かれる名前 → 貸方科目・補助科目)、ページのタイトル(→ 借方科目・補助科目・税区分)、事業者区分・課税方式・経理方式を入れる。
   弥生の登録名と**完全に一致**させる(一致しないと、取り込みで弥生に「マッチングリスト」が出る)。
5. 手書き帳票の PDF・画像を取り込み → 認識結果を確認 → 確定 → 「出力」で CSV を保存 → **弥生で直接取り込む(Excel で開かない)**。取り込みは元に戻せないので、**事前に弥生でバックアップ**を取る。

## 保存データの保護

- 取り込んだ画像・切り出し画像・確定した値(DB)・出力した CSV は `%LOCALAPPDATA%\yayoi-ocr` に入る。**アプリ自身は暗号化しない**。**BitLocker などのディスク暗号化を推奨する**(紛失・盗難への備え)。共有 PC では専用のアカウントで使う。
- 設定の「元画像の保管」で、出力後に元画像を削除できる。「学習」画面の「リセット」で、確定したセルの切り出し画像(学習データ)を削除できる。

## 精度レポート(開発者に送るもの)

「学習」画面の **精度レポートを保存**(`accuracy_report.json`)は、確定したページ数・要確認率・直した割合・理由コードごとの件数・確認時間だけの統計ファイル。
**帳票の内容(日付・科目・摘要・金額・タイトル・ファイル名)は一切含まれない**。中身は送る前にメモ帳で確認できる。

## 配布物の作り方(開発者)

```sh
cd web && npm ci && npm run build && cd ..          # フロントエンドのビルド(app/static に出力。.gitignore 済み)
pip install -e ".[dev,app,ml]"                       # build 用
python scripts/make_trial_bundle.py [--with-wheels]  # $YAYOI_PRIVATE_DATA/trial/yayoi-ocr-trial-<version>.zip
```

- 重みは `YAYOI_PRIVATE_DATA/models`(または `--models`)から、**許可リストの名前だけ**を入れる(評価用の画像・正解・学習データが紛れ込まない)。素の重みが足りないと失敗する。適応済みの重み(`models/adapted/`)があれば入れる。
- `--with-wheels` は Windows 向け(`win_amd64`)の依存の wheel を取得して同梱する(開発機のネットワークを使う。配布先はオフラインでよい)。
- 配布 zip は **コミットしない**(出力先がリポジトリの内側なら拒否する)。配布物の SHA256 を利用者に別の経路で伝える。
- 出所・ライセンス: `NOTICE.txt` に NDLOCR-Lite(CC BY 4.0)・baudm/parseq(Apache-2.0)を記載(`docs/dependencies.md`)。

## うまくいかないとき

| 症状 | 対処 |
|---|---|
| `setup.ps1` が実行できない | `-ExecutionPolicy Bypass` をつけて実行する(上の手順のとおり) |
| ウィンドウが開かない | WebView2 ランタイムが無い。上の表のとおり入れる |
| 認識が始まらない・遅い | 最初のページは重みの読み込みで 10 秒ほどかかる。2 ページ目以降は 1 ページ 10 秒前後(CPU の性能による)。`YAYOI_TORCH_THREADS` で計算に使うスレッド数を固定できる |
| 取り込めないファイルがある | 画面に理由が出る(開けないファイル・大きすぎる画像・ページ数が多すぎる等)。PDF は 1 ファイル数十ページまで |
| ウイルス対策ソフトが認識プロセスを止める | `venv\Scripts\python.exe` を除外に入れる(`docs/windows_checklist.md` F6) |
