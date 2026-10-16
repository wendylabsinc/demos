#!/bin/bash
# Downloads a pinned Ollama release into the container (the volume is noexec), serves it on
# loopback and pulls the showcase's vision model in the background.
set -u
# 0.40.4 is the release verified on the Jetson Orin (its cuda_v13 libraries). Override with
# OLLAMA_VERSION only after checking a newer one.
VERSION=${OLLAMA_VERSION:-0.40.4}
BIN=/opt/ollama  # the persist volume is mounted noexec
if [ ! -x $BIN/bin/ollama ]; then
  echo "downloading Ollama $VERSION"
  rm -rf $BIN.tmp && mkdir -p $BIN.tmp
  set -o pipefail
  if curl -fsSL "https://github.com/ollama/ollama/releases/download/v$VERSION/ollama-linux-arm64.tar.zst" \
     | zstd -d | tar -x -C $BIN.tmp; then
    rm -rf $BIN && mv $BIN.tmp $BIN
  else
    echo "Ollama $VERSION download failed"; sleep 60; exit 1
  fi
  set +o pipefail
fi
export LD_LIBRARY_PATH=$BIN/lib/ollama:${LD_LIBRARY_PATH:-}
$BIN/bin/ollama --version
$BIN/bin/ollama serve &
SERVER=$!
until $BIN/bin/ollama list >/dev/null 2>&1; do sleep 1; done
echo "pulling qwen3-vl:4b"
$BIN/bin/ollama pull qwen3-vl:4b 2>&1 | tr '\r' '\n' | awk 'NR%50==1 || /success|error/' && echo "model ready"
wait $SERVER
