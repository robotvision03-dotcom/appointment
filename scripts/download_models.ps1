$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }
Set-Location $Root
Write-Host "==> Shenava-Koochik-v1.5"
if (Get-Command py -ErrorAction SilentlyContinue) {
    py -m src download-shenava
} else {
    python -m src download-shenava
}
Write-Host "Restart: py -m src"
