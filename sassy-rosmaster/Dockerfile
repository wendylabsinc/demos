FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa LP_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libosmesa6 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY sassy_rosmaster ./sassy_rosmaster
COPY static ./static
COPY assets ./assets
COPY models ./models
EXPOSE 8096
CMD ["uv", "run", "--frozen", "python", "-m", "sassy_rosmaster"]
