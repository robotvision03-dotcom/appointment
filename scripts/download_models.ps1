# Download Whisper large Farsi v1 STT + Mana Piper TTS on Windows.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }
Set-Location $Root

$PiperDir = Join-Path $Root "models\piper-voice-fa"
New-Item -ItemType Directory -Force -Path $PiperDir | Out-Null

$PiperOnnx = Join-Path $PiperDir "fa_IR-mana-medium.onnx"
$PiperJson = Join-Path $PiperDir "fa_IR-mana-medium.onnx.json"
if (-not (Test-Path $PiperOnnx)) {
    Write-Host "==> Piper Mana (~64MB)"
    curl.exe -L --fail -o $PiperOnnx "https://huggingface.co/MahtaFetrat/Mana-Persian-Piper/resolve/main/fa_IR-mana-medium.onnx?download=true"
    curl.exe -L --fail -o $PiperJson "https://huggingface.co/MahtaFetrat/Mana-Persian-Piper/resolve/main/fa_IR-mana-medium.onnx.json?download=true"
} else {
    Write-Host "Piper already present"
}

Write-Host "==> Whisper large Farsi v1 (Hugging Face → CTranslate2)"
if (Get-Command py -ErrorAction SilentlyContinue) {
    py -m src.download_whisper
} else {
    python -m src.download_whisper
}

Write-Host "Done. Restart: py -m src"
Write-Host "Expect: models\whisper-large-fa-v1\model.bin"
