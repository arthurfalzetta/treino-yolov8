FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git build-essential python3-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install flask

ENV PYTHONUNBUFFERED=1

# Use string simples para o Render
CMD python servidor.py
