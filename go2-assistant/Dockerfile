FROM ghcr.io/astral-sh/uv:0.13.0 AS uv
FROM ros:humble-ros-base

COPY --from=uv /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libosmesa6 libgl1 libglfw3 \
      ros-humble-rmw-cyclonedds-cpp ros-humble-geometry-msgs \
      ros-humble-nav-msgs ros-humble-sensor-msgs \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=osmesa PYTHONUNBUFFERED=1 GO2_HOST=127.0.0.1 \
    GO2_MEMORY=/data/memory.sqlite3 ROS_DOMAIN_ID=0 \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_LOCALHOST_ONLY=0 \
    ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project -o /tmp/requirements.txt \
    && uv pip install --system --requirement /tmp/requirements.txt
COPY . .
RUN uv pip install --system --no-deps .

# Prepare the verified assets/ bundle before builds that need learned vision;
# simulator mode additionally uses its policy and robot files.
EXPOSE 8787
CMD ["python3", "-m", "go2_assistant.cli", "serve"]
