#!/usr/bin/env bash
# Pull a Persian-capable Ollama model. Tries several public names.
set -euo pipefail
if ! command -v ollama >/dev/null 2>&1; then
  echo "Install Ollama from https://ollama.com first."
  exit 1
fi

for model in \
  "${OLLAMA_MODEL:-persianllama:7b}" \
  "gemma3:4b" \
  "llama3.2:3b" \
  "qwen2.5:3b"; do
  echo "Trying ollama pull ${model}"
  if ollama pull "${model}"; then
    echo "Pulled ${model}. Set OLLAMA_MODEL=${model} in .env"
    exit 0
  fi
done
echo "Could not pull a model. Start ollama serve and try again."
exit 1
