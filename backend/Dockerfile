FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app .

RUN mkdir -p /app/temp_uploads && \
    mkdir -p /app/runs/track && \
    chmod -R 777 /app/temp_uploads /app/runs

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
# --workers 1 важно для GPU, чтобы избежать конфликтов доступа к модели из разных процессов Uvicorn