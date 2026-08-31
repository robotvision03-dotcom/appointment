#!/usr/bin/env bash
# Download offline Persian STT (Vosk) + TTS (Mana Piper). No Google / Twilio.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p models/vosk-model-fa models/piper-voice-fa

echo "==> Piper Persian voice: Mana-Persian-Piper (~64MB, Iranian-trained)"
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

echo "==> Vosk Persian STT: vosk-model-small-fa-0.5 (~60MB)"
if [[ ! -f models/vosk-model-fa/am/final.mdl && ! -f models/vosk-model-fa/conf/model.conf ]]; then
  TMP="$(mktemp -d)"
  # Small first (usable on CPU). Large 0.5 (~1GB) or 0.42 (~1.6GB) are more accurate.
  curl -L --fail -o "$TMP/vosk-fa.zip" \
    "https://alphacephei.com/vosk/models/vosk-model-small-fa-0.5.zip" \
    || curl -L --fail -o "$TMP/vosk-fa.zip" \
    "https://alphacephei.com/vosk/models/vosk-model-fa-0.5.zip"
  unzip -q "$TMP/vosk-fa.zip" -d "$TMP"
  INNER="$(find "$TMP" -maxdepth 1 -type d -name 'vosk-model*' | head -n1)"
  if [[ -n "${INNER}" ]]; then
    cp -a "${INNER}/." models/vosk-model-fa/
  fi
  rm -rf "$TMP"
else
  echo "Vosk model already present"
fi

echo "Done."
echo "  STT: models/vosk-model-fa"
echo "  TTS: $PIPER_ONNX"
echo "Restart: python -m src"
echo "Health: GET http://127.0.0.1:38471/health  (stt.available and tts.available should be true)"
