FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgl1 libgl1-mesa-dri && rm -rf /var/lib/apt/lists/*
COPY offline/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
ENV MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /work/physical
COPY . .
ENTRYPOINT ["python", "offline/run.py"]
