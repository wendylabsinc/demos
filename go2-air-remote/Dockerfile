FROM python:3.12-slim

# sounddevice is imported by the WebRTC library and needs PortAudio at import time.
RUN apt-get update && apt-get install -y --no-install-recommends libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps \
       https://github.com/legion1581/unitree_webrtc_connect/archive/e0abc5780761539eff89a382a79319e5cf6ad1f4.tar.gz

COPY app.py index.html ./
EXPOSE 8883
CMD ["python", "-u", "app.py"]
