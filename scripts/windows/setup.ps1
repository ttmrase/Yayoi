# yayoi-ocr 試用版のセットアップ(Windows 10/11 / PowerShell 5.1 以降)。
# 使い方: 配布 zip を展開したフォルダで
#     powershell -ExecutionPolicy Bypass -File .\setup.ps1
# 何をするか: ① Python 3.11 以降と WebView2 を確認 ② 配布物のハッシュを検証 ③ 専用の仮想環境を作って依存を入れる
#            ④ 認識の重みをコピー ⑤ 起動用のショートカットを作る ⑥ 動作確認(インポートとヘルプ)。
# wheels\ に torch の wheel が同梱されていれば完全にオフラインで入る。無ければ pip が(このセットアップの間だけ)ネットワークを使う。
# アプリ本体は実行時にネットワークを使わない。
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "yayoi-ocr"),
    [switch]$SkipVerify
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Step($msg) { Write-Host ""; Write-Host "== $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "エラー: $msg" -ForegroundColor Red; exit 1 }

Step "1/6 Python と WebView2 の確認"
$py = $null
foreach ($cand in @(@("py", "-3.12"), @("py", "-3.11"), @("python"))) {
    try {
        $out = & $cand[0] @($cand[1..($cand.Length - 1)]) -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) {
            $v = [version]$out
            if ($v -ge [version]"3.11" -and $v -lt [version]"3.14") { $py = $cand; Write-Host "Python $out を使います"; break }
        }
    } catch { }
}
if (-not $py) { Fail "Python 3.11〜3.13 が見つかりません。https://www.python.org/downloads/windows/ から 64 ビット版を入れ、再実行してください(「Add python.exe to PATH」にチェック)。" }

$wv2Keys = @(
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
    "HKLM:\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
    "HKCU:\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
)
$wv2 = $wv2Keys | Where-Object { Test-Path $_ }
if ($wv2) { Write-Host "WebView2 ランタイム: あり" }
else { Write-Warning "WebView2 ランタイムが見つかりません。Microsoft の「Evergreen Standalone Installer」(x64)を入れてください(Windows 11 には標準で入っています)。" }

if (-not $SkipVerify) {
    Step "2/6 配布物の検証"
    $manifest = Join-Path $here "MANIFEST.sha256"
    if (Test-Path $manifest) {
        $bad = 0
        foreach ($line in Get-Content $manifest -Encoding UTF8) {
            if (-not $line.Trim()) { continue }
            $hash, $rel = $line -split "  ", 2
            $path = Join-Path $here ($rel -replace "/", "\")
            if (-not (Test-Path $path)) { Write-Host "  無い: $rel" -ForegroundColor Red; $bad++; continue }
            if ((Get-FileHash $path -Algorithm SHA256).Hash.ToLower() -ne $hash.ToLower()) { Write-Host "  不一致: $rel" -ForegroundColor Red; $bad++ }
        }
        if ($bad -gt 0) { Fail "配布物が壊れています($bad 件)。zip をもう一度展開してください。" }
        Write-Host "ハッシュ一致"
    } else { Write-Warning "MANIFEST.sha256 がありません(検証を省略)" }
} else { Step "2/6 配布物の検証(省略)" }

Step "3/6 仮想環境と依存の導入"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$venv = Join-Path $InstallDir "venv"
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    & $py[0] @($py[1..($py.Length - 1)]) -m venv $venv
    if ($LASTEXITCODE -ne 0) { Fail "仮想環境を作れませんでした" }
}
$vpy = Join-Path $venv "Scripts\python.exe"
$wheels = Join-Path $here "wheels"
$app = Get-ChildItem $wheels -Filter "yayoi_ocr-*.whl" | Select-Object -First 1
if (-not $app) { Fail "wheels\yayoi_ocr-*.whl がありません" }
$offline = [bool](Get-ChildItem $wheels -Filter "torch-*.whl" -ErrorAction SilentlyContinue)
if ($offline) {
    Write-Host "同梱の wheel から入れます(オフライン)"
    & $vpy -m pip install --no-index --find-links $wheels "$($app.FullName)[app,desktop,ml]"
} else {
    Write-Host "torch(CPU 版)を取得します(ネットワークを使います。これはセットアップの間だけです)"
    & $vpy -m pip install --upgrade pip
    & $vpy -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    & $vpy -m pip install --find-links $wheels "$($app.FullName)[app,desktop,ml]"
}
if ($LASTEXITCODE -ne 0) { Fail "依存の導入に失敗しました" }

Step "4/6 認識の重みをコピー"
$models = Join-Path $InstallDir "models"
if (-not (Test-Path (Join-Path $here "models"))) { Fail "models\ がありません" }
Copy-Item -Recurse -Force (Join-Path $here "models\*") $models -ErrorAction SilentlyContinue
if (-not (Test-Path $models)) { New-Item -ItemType Directory -Force -Path $models | Out-Null; Copy-Item -Recurse -Force (Join-Path $here "models\*") $models }
Write-Host "重み: $models"

Step "5/6 起動用ファイルとショートカット"
Copy-Item -Force (Join-Path $here "start-yayoi-ocr.cmd") (Join-Path $InstallDir "start-yayoi-ocr.cmd")
foreach ($f in @("README.txt", "NOTICE.txt", "windows_setup.md", "windows_checklist.md", "yayoi_export.md")) {
    if (Test-Path (Join-Path $here $f)) { Copy-Item -Force (Join-Path $here $f) $InstallDir }
}
try {
    $desktop = [Environment]::GetFolderPath("Desktop")
    $sh = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $desktop "yayoi-ocr.lnk"))
    $sh.TargetPath = Join-Path $InstallDir "start-yayoi-ocr.cmd"
    $sh.WorkingDirectory = $InstallDir
    $sh.WindowStyle = 7  # 最小化
    $sh.Save()
    Write-Host "デスクトップに yayoi-ocr のショートカットを作りました"
} catch { Write-Warning "ショートカットを作れませんでした。$InstallDir\start-yayoi-ocr.cmd を直接起動してください。" }

Step "6/6 動作確認"
$env:YAYOI_MODELS_DIR = $models
& $vpy -c "import yayoi_ocr, torch, webview, fastapi; print('yayoi-ocr', yayoi_ocr.__version__, '/ torch', torch.__version__)"
if ($LASTEXITCODE -ne 0) { Fail "インポートの確認に失敗しました" }
& $vpy -m yayoi_ocr.app --help | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "アプリの起動確認に失敗しました" }
& $vpy -c "from yayoi_ocr.recognize import variants; print('使える重み:', variants.available_variants())"

Write-Host ""
Write-Host "完了しました。デスクトップの yayoi-ocr、または $InstallDir\start-yayoi-ocr.cmd から起動します。" -ForegroundColor Green
Write-Host "データの置き場: $InstallDir(取り込んだ画像・確定した値・出力した CSV が入ります。BitLocker などのディスク暗号化を推奨します)"
