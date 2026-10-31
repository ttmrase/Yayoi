# Phase 6(配布・インストーラ)に向けた注意

インストーラは Phase 6。それまでの試用版は「セットアップのスクリプト + 手順書」(`docs/windows_setup.md`。オフライン版のみ。依存は `requirements/trial-win-cp311.txt` のバージョン固定 + ハッシュ。重みの出所は `docs/model_provenance.md`)。ここには、配布するときに守る事項を集める。

## 保存データの保護(checker I-3)

アプリのデータ領域(`%LOCALAPPDATA%\yayoi-ocr`)には、取り込んだ実帳票の画像・切り出し画像・確定した値(DB)・出力した CSV が入る。
**アプリ自身は保存データを暗号化しない**。権限と暗号化は OS の保護に従う:

- Windows のユーザーアカウントごとに分かれた領域に置く(他のユーザーからは読めない既定の ACL)。共有 PC では専用のアカウントで使う。
- **BitLocker(または同等のディスク暗号化)の有効化を推奨**する(紛失・盗難への備え)。手順書に明記する。
- アプリの設定「元画像の保管」で、出力後に元画像を削除できる。「学習」画面の「リセット」で、確定したセルの切り出し画像(学習データ)を削除できる。
  削除はデータ領域の中だけを対象にする。
- バックアップは利用者の責任(データ領域のフォルダを閉じた状態でコピー)。弥生へのインポート前には、**弥生側でバックアップを取る**(インポートは元に戻せない)。

## 認識ワーカーが `spawn` の別プロセスであること

- 認識(PyTorch)とアップロードの検査(`app/probe.py`)は別プロセスで動く。PyInstaller などで固めるときは、起動スクリプトの先頭で `multiprocessing.freeze_support()` を呼び、
  `if __name__ == "__main__":` の中でだけアプリを起動する(ガードが無いと子プロセスがアプリを再起動する)。
- 検査は `sys.executable -I -m yayoi_ocr.app.probe` で起動している。実行ファイルに固めた場合は、同じ実行ファイルを検査用のサブコマンドで起動する形に差し替える(`uploads.inspect_file`)。

## WebView2 ランタイム

Windows 10 の一部には無い。オフラインインストーラ(Evergreen の固定版)を同梱するか、手順書に導入方法を書く。無い PC では分かるメッセージを出す(チェックリスト A4)。

## 重みとフォント

- 認識の重み(NDLOCR-Lite 由来の PARSeq・適応済みの重み・確信度モデル)は**リポジトリにコミットしない**。配布物には、オフラインで同梱する(`scripts/make_trial_bundle.py` が作る zip)か、
  事前に取得する手順(`scripts/fetch_models.py`)にする。出所・ライセンスは `docs/dependencies.md`。
- 合成帳票用のフォントは学習・評価用で、アプリの実行には要らない。

## 通信

アプリは実行時にネットワークを使わない(127.0.0.1 のサーバだけ)。ファイアウォールの許可ダイアログが出ないこと、オフラインで動くことをチェックリスト A1・A2 で確かめる。
開発専用の `--token` / `--port` / `--no-window` / `--backend fake` は `--dev` が無いと受け付けない(プロセス一覧からトークンが見えるため)。

## ログ・ライセンス・手引き・アンインストール・バックアップ(試用版に入れたもの)

- **ログ**(`app/logs.py`): `%LOCALAPPDATA%\yayoi-ocr\logs\app.log`。1 MiB で回転し、世代は 5 つまで(最大 約 6 MiB)、更新から 30 日を過ぎたものは起動時に消す。
  **値(摘要・金額・ファイル名・設定の名前・パス)は出さない**: 呼び出し側は ID・件数・コード・例外の型名だけを出し、さらに `ValueFreeFormatter` が**例外のメッセージとトレースバックを書かない**(型名だけ)。
  サーバ(uvicorn)の未処理の例外も同じ設定に流す(`launcher` が `log_config=None` で uvicorn の既定のログ設定を外す。既定のままだと、入力値を含む例外の文面とトレースバックが標準エラーに出る)。
  保証は `tests/test_logs.py`(一通りの操作の後で、入力した値・ファイル名・パスがログに無い。未処理のサーバ例外が型名だけで、ファイルにもコンソールにも値が出ない)。標準エラー(最小化したコンソール)にも WARNING 以上を出す。
- **ライセンス表示**(`app/licenses.py`・`scripts/gen_licenses.py`・画面の「ライセンス」): 主な利用物の出典・ライセンス・変更点(NDLOCR-Lite の CC BY 4.0、PARSeq の Apache-2.0、Python の PSF など)は固定で持ち、
  依存ライブラリの一覧は**配布物の作成時に lock と wheel の METADATA から生成**する(バージョンは lock が正)。ライセンス本文(各 wheel の LICENSE・NOTICE、npm の LICENSE)は配布物の `licenses\` に入れる。
- **利用者向けの手引き**(`docs/user_manual.md` → 配布物の `manual.html`・`user_manual.md`): 初期設定・取り込み・確認(キー操作)・出力と弥生への取り込み・学習画面の見方(提案モード・自動停止)・バックアップ・困ったとき・アンインストール。
- **アンインストール**(`scripts/windows/uninstall.ps1`): 既定ではプログラムだけ削除してデータは残す。`-DeleteData` で、データも(「DELETE」の入力で確認して)削除する。削除はインストール先(名前が `yayoi-ocr`)の中の既知の名前だけ。
  `-Force` でも、**そのフォルダが本アプリのものだという印**(`start-yayoi-ocr.cmd`・`app.sqlite3`・`venv`・`python`・`models` のどれか)が無ければ削除しない。再解析ポイント(ジャンクション・シンボリックリンク)を含む場合も、辿らずに拒否する。
  データ領域とインストール先は同じフォルダで、`models\` には出荷時の重みと学習した世代(`models\<数字>\`)が同居する。**既定(データを残す)では、プログラムの一部である出荷時の重みだけを消し、学習した世代は残す**。`-DeleteData` では `models\` ごと消す。
  通常の削除はディスクの内容を完全には消さないので、BitLocker か `cipher /w` を案内する。
- **バックアップ・復元**(`docs/backup_restore.md`)。**WebView2 のダウンロード**は pywebview の `ALLOW_DOWNLOADS` を True にしないと取り消されるため、`launcher.configure_webview` で有効にしている。
- **Windows の CI**(`.github/workflows/ci.yml` の `trial-windows`): 束を作り、`setup.ps1` を Windows PowerShell 5.1 で実行し、開発用でないモードで起動して HTTP で通す(`scripts/trial_smoke.py`)。
  この CI は、`setup.ps1` の重みのコピーの不具合(置き場が無いまま `Copy-Item src\*` するとファイルとして書かれる)を見つけた。
  実配置(データ領域 = インストール先)で、バックアップの復元(準備 → 再起動 → 入れ替え)と、アンインストール(データを残す/消す・別フォルダの拒否)も通す。
  未対応(煙の試験としての限界): GitHub Actions の外部アクションは版の固定(コミット SHA)までしていない。`fetch_fonts` は固定のハッシュを持たない(煙の試験の入力にだけ使い、配布物の検証ではない。配布物の重み・フォントは作成時に `model_provenance` の SHA256 と照合する)。
- **束の検査**(`scripts/verify_trial_bundle.py`、試験は `tests/test_verify_trial_bundle.py`): 作った zip を展開せずに読み、zip の SHA256 の控え・MANIFEST・`models/` と `model_provenance.md` の表の過不足のない一致・
  wheel と lock の対応(名前・版・ハッシュ)・同梱の Python の固定値・画像/CSV/DB の混入・拒否リストの語を確かめる。作った束の記録は `docs/trial_rc.md`。
- **GPU**: `docs/phase6_gpu_design.md`(試用は CPU 版。判断材料の整理だけ)。
