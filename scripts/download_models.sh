#!/usr/bin/env bash
# Download Whisper large Farsi v1 (STT) + Mana Piper (TTS).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p models models/piper-voice-fa

echo "==> Piper Persian voice: Mana-Persian-Piper (~64MB)"
PIPER_ONNX="models/piper-voice-fa/fa_IR-mana-medium.onnx"
PIPER_JSON="models/piper-voice-fa/fa_IR-mana-medium.onnx.json"
if [[ ! -f "$PIPER_ONNX" ]]; then
  curl -L --fail -o "$PIPER_ONNX" \
    "https://huggingface.co/MahtaFetrat/Mana-Persian-Piper/resolve/main/fa_IR-mana-medium.onnx?download=true"
  curl -L --fail -o "$PIPER_JSON" \
    "https://huggingface.co/MahtaFetrat/Mana-Persian-Piper/resolve/main/fa_IR-mana-medium.onnx.json?download=true"
else
  echo "Piper model already present"
fi

echo "==> Whisper large Farsi v1 (vhdm/whisper-large-fa-v1, CTranslate2 int8)"
python -m src download-whisper

echo "Done."
echo "  STT: models/whisper-large-fa-v1"
echo "  TTS: $PIPER_ONNX"
echo "Restart: python -m src"
