FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py index.html *_pb2.py *_pb2_grpc.py ./
EXPOSE 8883
CMD ["python", "-u", "app.py"]
