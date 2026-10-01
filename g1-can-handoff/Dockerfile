FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 libgl1 libglfw3 \
    && rm -rf /var/lib/apt/lists/*
ENV MUJOCO_GL=osmesa
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for better caching
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project

# Copy application code
COPY . .
RUN uv sync --frozen

CMD ["uv", "run", "g1_can_handoff"]
