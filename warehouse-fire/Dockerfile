FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV FIRE_DATA_DIR=/data FIRE_HOST=0.0.0.0
EXPOSE 8788
CMD ["python", "demo.py"]
