# syntax=docker/dockerfile:1.6
# Engine and model pins shared with ../NvidiaPAIR, tested there on Orin Nano.
ARG OLLAMA_IMAGE=ollama/ollama:0.40.13@sha256:b2f0122c476bc9fe7ce679d8bb4bd555351e25e2554eefb156f61cb0a89cf429
FROM --platform=$BUILDPLATFORM ${OLLAMA_IMAGE} AS model
ENV OLLAMA_MODELS=/opt/ollama-models OLLAMA_HOST=127.0.0.1:11434
RUN set -eu; ollama serve >/tmp/ollama.log 2>&1 & server=$!; \
    trap 'kill "$server"; wait "$server" || true' EXIT; \
    ready=0; for attempt in $(seq 1 60); do \
      if ollama list >/dev/null 2>&1; then ready=1; break; fi; sleep 1; \
    done; test "$ready" = 1; \
    ollama pull qwen2.5:0.5b; \
    printf '%s  %s\n' \
      6467b5fd1dd41b31569d53f7286b0db4250515449471171370b38a96ba2216fc \
      /opt/ollama-models/manifests/registry.ollama.ai/library/qwen2.5/0.5b | sha256sum -c -
FROM python:3.12-slim AS vision
COPY fetch_vision_model.py /fetch_vision_model.py
RUN python /fetch_vision_model.py

FROM ${OLLAMA_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
COPY --from=model /opt/ollama-models/ /opt/ollama-models/
COPY --from=vision /opt/vision-models/ /opt/ollama-models/
COPY app.py discovery.py index.html ./
ENV PATH=/opt/venv/bin:$PATH HOME=/data PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS=/opt/ollama-models \
    OLLAMA_KEEP_ALIVE=30s OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_CONTEXT_LENGTH=2048 \
    OLLAMA_LOAD_TIMEOUT=10m
ENTRYPOINT ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090"]
