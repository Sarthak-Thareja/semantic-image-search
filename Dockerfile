# Multimodal Semantic Image-Text Retrieval System
FROM python:3.11-slim

WORKDIR /app

# Install system deps for PIL/torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/

COPY start.sh ./
RUN chmod +x start.sh

ENV PYTHONPATH=/app
ENV PORT=8000
EXPOSE $PORT

CMD ["./start.sh"]
