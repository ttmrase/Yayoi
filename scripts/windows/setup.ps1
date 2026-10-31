# yayoi-ocr 試用版のセットアップ(Windows 10/11 / PowerShell 5.1 以降)。
# 使い方: 配布 zip を展開したフォルダで
#     powershell -ExecutionPolicy Bypass -File .\setup.ps1
# 何をするか: ① Python 3.11 以降と WebView2 を確認 ② 配布物のハッシュを検証 ③ 専用の仮想環境を作って依存を入れる
#            ④ 認識の重みをコピー ⑤ 起動用のショートカットを作る ⑥ 動作確認(インポートとヘルプ)。
# 完全にオフラインで入る(wheels\ の依存は、バージョン固定 + ハッシュつきの requirements-lock.txt のとおり。ネットワークは使わない)。
# Python は zip に同梱(python-standalone.tar.gz。python-build-standalone の CPython 3.11)。システムの Python は要らない
# (同梱が無い版では、PC の Python 3.11 か、同梱の python-installer.exe を使う)。
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
function Test-Python311($exe, $pyargs) {
    try {
        $out = & $exe @($pyargs) -c "import sys; print('%d.%d.%d' % sys.version_info[:3]); print(sys.maxsize > 2**32)" 2>$null
        return ($LASTEXITCODE -eq 0 -and $out -and ($out -join " ") -match "^3\.11\.\d+ True$")
    } catch { return $false }
}
function Find-Python311 {
    $cands = @(
        @{ exe = "py"; args = @("-3.11") },
        @{ exe = "python"; args = @() },
        @{ exe = (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"); args = @() }
    )
    foreach ($c in $cands) { if (Test-Python311 $c.exe $c.args) { return $c } }
    return $null
}
$py = $null
$bundled = Join-Path $here "python-standalone.tar.gz"
$pyHome = Join-Path $InstallDir "python"
if (Test-Path $bundled) {
    # 同梱の Python(python-build-standalone)。展開前に、固定した SHA256 と一致するか確かめる
    $pin = @{}
    foreach ($line in Get-Content (Join-Path $here "python-standalone.pin") -Encoding UTF8) {
        if ($line -match "^\s*([a-z0-9_]+)=(.+)$") { $pin[$Matches[1]] = $Matches[2].Trim() }
    }
    if (-not $pin["sha256"]) { Fail "python-standalone.pin が壊れています" }
    if ((Get-FileHash $bundled -Algorithm SHA256).Hash.ToLower() -ne $pin["sha256"].ToLower()) { Fail "同梱の Python の SHA256 が固定値と違います。zip をもう一度展開してください。" }
    $pyExe = Join-Path $pyHome "python.exe"
    if (-not (Test-Python311 $pyExe @())) {
        $tar = Join-Path $env:SystemRoot "System32\tar.exe"
        if (-not (Test-Path $tar)) { Fail "tar.exe が見つかりません(Windows 10 version 1803 以降が必要です)" }
        New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
        if (Test-Path $pyHome) { Remove-Item -Recurse -Force $pyHome }
        Write-Host "同梱の Python を展開します: $pyHome"
        & $tar -xzf $bundled -C $InstallDir
        if ($LASTEXITCODE -ne 0) { Fail "同梱の Python を展開できませんでした" }
    }
    if (Test-Python311 $pyExe @()) { $py = @{ exe = $pyExe; args = @() } }
}
if (-not $py) { $py = Find-Python311 }
if (-not $py) {
    $installer = Join-Path $here "python-installer.exe"
    if (Test-Path $installer) {
        # 署名者が Python Software Foundation で署名が有効なものだけを実行する(zip の中の MANIFEST では差し替えを防げないため)
        $sig = Get-AuthenticodeSignature $installer
        if ($sig.Status -ne "Valid" -or $sig.SignerCertificate.Subject -notmatch "Python Software Foundation") {
            Fail "python-installer.exe の署名を確認できません(状態: $($sig.Status))。実行しません。"
        }
        Write-Host "Python 3.11 が無いので、同梱のインストーラで導入します(署名者: Python Software Foundation。現在のユーザー用。管理者権限は要りません)"
        $p = Start-Process -FilePath $installer -ArgumentList "/quiet", "InstallAllUsers=0", "PrependPath=0", "Include_launcher=0", "Include_test=0", "Include_doc=0", "Shortcuts=0" -Wait -PassThru
        if ($p.ExitCode -ne 0) { Fail "Python の導入に失敗しました(終了コード $($p.ExitCode))" }
        $py = Find-Python311
    }
}
if (-not $py) { Fail "Python 3.11(64 ビット)が見つかりません。この zip には Python が入っていません。python.org の Windows installer (64-bit) の 3.11 系を入れてから、もう一度実行してください。" }
Write-Host "Python 3.11 を使います: $($py.exe) $($py.args -join ' ')"

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

Step "3/6 仮想環境と依存の導入(オフライン・ハッシュ検証つき)"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$venv = Join-Path $InstallDir "venv"
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    & $py.exe @($py.args) -m venv $venv
    if ($LASTEXITCODE -ne 0) { Fail "仮想環境を作れませんでした" }
}
$vpy = Join-Path $venv "Scripts\python.exe"
$wheels = Join-Path $here "wheels"
$lock = Join-Path $here "requirements-lock.txt"
$app = Get-ChildItem $wheels -Filter "yayoi_ocr-*.whl" | Select-Object -First 1
$proxy = Get-ChildItem $wheels -Filter "proxy_tools-*.whl" | Select-Object -First 1
if (-not $app -or -not $proxy -or -not (Test-Path $lock)) { Fail "配布物が足りません(wheels\yayoi_ocr-*.whl・proxy_tools-*.whl・requirements-lock.txt)" }
# 依存はすべて lock のバージョン・ハッシュと一致するものだけを入れる(--no-index: ネットワークを使わない)
& $vpy -m pip install --no-index --find-links $wheels --no-deps --require-hashes -r $lock
if ($LASTEXITCODE -ne 0) { Fail "依存の導入に失敗しました(ハッシュが合わない場合は、zip をもう一度展開してください)" }
# このアプリと、sdist しか無かった純 Python の依存(配布物のハッシュは MANIFEST.sha256 で検証済み)
& $vpy -m pip install --no-index --no-deps $app.FullName $proxy.FullName
if ($LASTEXITCODE -ne 0) { Fail "アプリの導入に失敗しました" }
& $vpy -m pip check
if ($LASTEXITCODE -ne 0) { Fail "依存の整合性の確認に失敗しました" }

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
