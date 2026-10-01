FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates libosmesa6 libegl1 libgl1 libglib2.0-0 libgomp1 libatomic1 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    YARD_DRONE_HOST=0.0.0.0 YARD_DRONE_PORT=8895 YARD_DRONE_DATA=/data \
    YARD_DRONE_VISION_MODELS=/opt/models/vision
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY assets.lock.json models.lock.json ./
COPY yard_drone/__init__.py yard_drone/assets.py yard_drone/models.py ./yard_drone/
RUN python -m yard_drone.assets && python -m yard_drone.models
COPY vision-models.json download_vision_models.py ./
RUN python download_vision_models.py --models-root /opt/models/vision
COPY yard_drone ./yard_drone
RUN python -m yard_drone.assets --check
EXPOSE 8895
CMD ["python", "-m", "yard_drone.server"]
