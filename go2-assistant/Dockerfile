# Adapted from Wendy's Python API template; serves sim and hardware modes.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 libgl1 libglfw3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV MUJOCO_GL=osmesa PYTHONUNBUFFERED=1 GO2_HOST=127.0.0.1 GO2_MEMORY=/data/memory.sqlite3
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY . .
RUN uv sync --frozen
# Simulation bundle must be prepared on the developer host before building.
# Physical mode does not load the simulator, meshes, or locomotion policy.
EXPOSE 8787
CMD ["uv", "run", "--frozen", "go2_assistant", "serve"]
