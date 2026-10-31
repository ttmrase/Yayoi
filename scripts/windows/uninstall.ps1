# yayoi-ocr 試用版のアンインストール(Windows 10/11 / PowerShell 5.1 以降)。
# 使い方:
#     powershell -ExecutionPolicy Bypass -File .\uninstall.ps1                 ... プログラムだけ削除(データは残す)
#     powershell -ExecutionPolicy Bypass -File .\uninstall.ps1 -DeleteData     ... データも削除("DELETE" と入力して確認)
# 削除するのは、インストール先(既定: %LOCALAPPDATA%\yayoi-ocr)の中の、このアプリが作ったものだけです。
#   プログラム: python\ venv\ models\ start-yayoi-ocr.cmd README.txt NOTICE.txt などの文書、デスクトップのショートカット
#   データ(-DeleteData): app.sqlite3 images\ uploads\ pagedata\ exports\ models(世代)は上の models\ に含まれる training\ backups\ restore\ restore-undo\ logs\
# 通常のファイル削除はディスクの内容を完全には消しません。廃棄・譲渡の前は、BitLocker を有効にするか、cipher /w で空き領域を消去してください。
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "yayoi-ocr"),
    [switch]$DeleteData,
    [switch]$Force
)
$ErrorActionPreference = "Stop"
function Fail($msg) { Write-Host "エラー: $msg" -ForegroundColor Red; exit 1 }

# 安全のため: 名前が yayoi-ocr で終わる、ドライブの直下でもホームでもないフォルダだけを対象にする
$full = [System.IO.Path]::GetFullPath($InstallDir).TrimEnd("\")
$leaf = Split-Path -Leaf $full
$root = [System.IO.Path]::GetPathRoot($full).TrimEnd("\")
if ($leaf -ne "yayoi-ocr" -and -not $Force) { Fail "インストール先の名前が yayoi-ocr ではありません($full)。間違いなければ -Force を付けてください。" }
if ($full -eq $root -or $full -eq $env:USERPROFILE.TrimEnd("\") -or $full -eq $env:LOCALAPPDATA.TrimEnd("\")) { Fail "このフォルダは削除できません($full)" }
if (-not (Test-Path $full)) { Write-Host "インストール先が見つかりません: $full(削除するものはありません)"; exit 0 }

# 動いているアプリがあれば、終了を促す(この venv の python が動いているときは削除できない)
$running = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($full, [System.StringComparison]::OrdinalIgnoreCase) }
if ($running) { Fail "yayoi-ocr が動いています。アプリと黒い小さな画面を閉じてから、もう一度実行してください。" }

if ($DeleteData) {
    Write-Host "データ(取り込んだ画像・確定した値・学習した世代・出力・ログ)も削除します: $full" -ForegroundColor Yellow
    Write-Host "必要なら、先に「バックアップを作成」で保存してください。削除したデータは戻せません。"
    $answer = Read-Host "続けるには DELETE と入力してください"
    if ($answer -cne "DELETE") { Write-Host "中止しました(何も削除していません)"; exit 1 }
}

$program = @("python", "venv", "models", "start-yayoi-ocr.cmd", "README.txt", "NOTICE.txt", "windows_setup.md", "windows_checklist.md", "yayoi_export.md", "user_manual.md", "manual.html", "licenses")
$data = @("app.sqlite3", "app.sqlite3-wal", "app.sqlite3-shm", "images", "uploads", "pagedata", "exports", "training", "backups", "restore", "restore-undo", "restore-result.json", "logs")
$targets = $program
if ($DeleteData) {
    $targets = $targets + $data + @("uninstall.ps1")  # データも消すときは、このスクリプトも消す(フォルダごと無くす)
    # 世代の重みはデータ(models\<プロファイル>\gen_XXXX)だが、models\ ごと消えるので上の models に含まれる
}
$removed = 0
foreach ($name in $targets) {
    $p = Join-Path $full $name
    if (Test-Path $p) {
        # 念のため: 解決したパスが、インストール先の直下であること
        $resolved = (Resolve-Path $p).Path
        if (-not $resolved.StartsWith($full + "\", [System.StringComparison]::OrdinalIgnoreCase)) { Fail "対象がインストール先の外です: $resolved" }
        Remove-Item -Recurse -Force $p
        $removed++
    }
}
# デスクトップのショートカット
try {
    $lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "yayoi-ocr.lnk"
    if (Test-Path $lnk) { Remove-Item -Force $lnk; $removed++ }
} catch { Write-Warning "ショートカットを削除できませんでした。手で削除してください。" }
# 何も残っていなければ、フォルダそのものも削除する
if (-not (Get-ChildItem -Force $full -ErrorAction SilentlyContinue)) { Remove-Item -Force $full }

Write-Host ""
if ($DeleteData) { Write-Host "プログラムとデータを削除しました($removed 件)。" -ForegroundColor Green }
else {
    Write-Host "プログラムを削除しました($removed 件)。データは残してあります: $full" -ForegroundColor Green
    Write-Host "データも消すときは、-DeleteData を付けてもう一度実行してください。"
}
Write-Host "バックアップの zip、弥生に渡した CSV、持ち出した精度レポートは、この操作では消えません。それぞれ手で削除してください。"
