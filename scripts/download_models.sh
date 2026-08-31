#!/usr/bin/env bash
# Download Persian Vosk + Piper models into ./models
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p models/vosk-model-fa models/piper-voice-fa

echo "==> Piper Persian voice (fa_IR mena-medium)"
PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/mena-medium"
if [[ ! -f models/piper-voice-fa/fa_IR-mena-medium.onnx ]]; then
  curl -L --fail -o models/piper-voice-fa/fa_IR-mena-medium.onnx \
    "${PIPER_BASE}/fa_IR-mena-medium.onnx"
  curl -L --fail -o models/piper-voice-fa/fa_IR-mena-medium.onnx.json \
    "${PIPER_BASE}/fa_IR-mena-medium.onnx.json"
else
  echo "Piper model already present"
fi

echo "==> Vosk Persian model (vosk-model-fa-0.5, ~1GB)"
if [[ ! -f models/vosk-model-fa/am/final.mdl && ! -f models/vosk-model-fa/conf/model.conf ]]; then
  TMP="$(mktemp -d)"
  curl -L --fail -o "$TMP/vosk-fa.zip" \
    "https://alphacephei.com/vosk/models/vosk-model-fa-0.5.zip" \
    || curl -L --fail -o "$TMP/vosk-fa.zip" \
    "https://alphacephei.com/vosk/models/vosk-model-small-fa-0.5.zip" \
    || { echo "Could not download Vosk Persian model. Place it at models/vosk-model-fa/"; exit 0; }
  unzip -q "$TMP/vosk-fa.zip" -d "$TMP"
  INNER="$(find "$TMP" -maxdepth 1 -type d -name 'vosk-model*' | head -n1)"
  if [[ -n "${INNER}" ]]; then
    rsync -a "${INNER}/" models/vosk-model-fa/
  fi
  rm -rf "$TMP"
else
  echo "Vosk model already present"
fi

echo "Done. Set VOSK_MODEL_PATH and PIPER_MODEL_PATH in .env if you used custom locations."
