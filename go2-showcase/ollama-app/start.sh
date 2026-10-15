#!/bin/bash
# Downloads Ollama into the container (the volume is noexec), serves it on loopback and
# pulls the showcase's vision model in the background.
set -u
BIN=/opt/ollama  # the persist volume is mounted noexec
if [ ! -x $BIN/bin/ollama ]; then
  echo "downloading Ollama"
  mkdir -p $BIN.tmp
  if curl -fsSL https://ollama.com/download/ollama-linux-arm64.tar.zst | zstd -d | tar -x -C $BIN.tmp \
     || curl -fsSL https://ollama.com/download/ollama-linux-arm64.tgz | tar -xz -C $BIN.tmp; then
    rm -rf $BIN && mv $BIN.tmp $BIN
  else
    echo "Ollama download failed"; sleep 60; exit 1
  fi
fi
export LD_LIBRARY_PATH=$BIN/lib/ollama:${LD_LIBRARY_PATH:-}
$BIN/bin/ollama --version
$BIN/bin/ollama serve &
SERVER=$!
until $BIN/bin/ollama list >/dev/null 2>&1; do sleep 1; done
echo "pulling qwen3-vl:4b"
$BIN/bin/ollama pull qwen3-vl:4b 2>&1 | tr '\r' '\n' | awk 'NR%50==1 || /success|error/' && echo "model ready"
wait $SERVER
