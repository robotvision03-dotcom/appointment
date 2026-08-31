# Download Persian Vosk STT + Mana Piper TTS on Windows (no Google).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }
Set-Location $Root

$VoskDir = Join-Path $Root "models\vosk-model-fa"
$PiperDir = Join-Path $Root "models\piper-voice-fa"
New-Item -ItemType Directory -Force -Path $VoskDir, $PiperDir | Out-Null

$PiperOnnx = Join-Path $PiperDir "fa_IR-mana-medium.onnx"
$PiperJson = Join-Path $PiperDir "fa_IR-mana-medium.onnx.json"
if (-not (Test-Path $PiperOnnx)) {
    Write-Host "==> Piper Mana (~64MB)"
    curl.exe -L --fail -o $PiperOnnx "https://huggingface.co/MahtaFetrat/Mana-Persian-Piper/resolve/main/fa_IR-mana-medium.onnx?download=true"
    curl.exe -L --fail -o $PiperJson "https://huggingface.co/MahtaFetrat/Mana-Persian-Piper/resolve/main/fa_IR-mana-medium.onnx.json?download=true"
} else {
    Write-Host "Piper already present"
}

$HasVosk = (Test-Path (Join-Path $VoskDir "conf\model.conf")) -or (Test-Path (Join-Path $VoskDir "am\final.mdl"))
if (-not $HasVosk) {
    Write-Host "==> Vosk Persian small (~60MB)"
    $Zip = Join-Path $Root "models\vosk-model-small-fa-0.5.zip"
    curl.exe -L --fail -o $Zip "https://alphacephei.com/vosk/models/vosk-model-small-fa-0.5.zip"
    $Tmp = Join-Path $Root "models\_vosk_unpack"
    if (Test-Path $Tmp) { Remove-Item -Recurse -Force $Tmp }
    Expand-Archive -Path $Zip -DestinationPath $Tmp -Force
    $Inner = Get-ChildItem $Tmp -Directory | Where-Object { $_.Name -like "vosk-model*" } | Select-Object -First 1
    if (-not $Inner) { throw "Zip did not contain a vosk-model* folder" }
    Copy-Item -Path (Join-Path $Inner.FullName "*") -Destination $VoskDir -Recurse -Force
    Remove-Item -Recurse -Force $Tmp
    Remove-Item -Force $Zip
} else {
    Write-Host "Vosk already present"
}

Write-Host "Done. Restart: py -m src"
Write-Host "Expect: models\vosk-model-fa\conf\model.conf"
