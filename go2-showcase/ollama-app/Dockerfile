FROM ros:humble-ros-base
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl zstd \
    && rm -rf /var/lib/apt/lists/*
COPY start.sh /start.sh
ENV OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS=/data/models OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_KEEP_ALIVE=-1
CMD ["bash", "/start.sh"]
