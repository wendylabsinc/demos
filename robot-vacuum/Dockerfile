FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1 VACUUM_MEDIA_DIR=/data
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 libgl1 libglfw3 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copy application code
COPY . .
RUN uv sync --frozen

EXPOSE 18881
CMD ["uv", "run", "--frozen", "robot-vacuum", "serve", "--host", "0.0.0.0", "--port", "18881"]
