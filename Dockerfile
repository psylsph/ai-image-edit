# Multi-stage build for AI Image Editor (CPU-only)
# Stage 1: Download Real-ESRGAN models
FROM python:3.11-slim AS model-downloader

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.cache/ai-image-edit/realesrgan

RUN wget -q -O /root/.cache/ai-image-edit/realesrgan/RealESRGAN_x2plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

RUN wget -q -O /root/.cache/ai-image-edit/realesrgan/RealESRGAN_x4plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# Stage 2: Final image
FROM python:3.11-slim

EXPOSE 8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY --from=model-downloader /root/.cache/ai-image-edit /root/.cache/ai-image-edit

COPY . /home/app/
WORKDIR /home/app

RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /home/app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; r=urllib.request.urlopen('http://localhost:8000/'); assert r.status==200 and 'AI Image Editor' in r.read().decode()"

CMD ["python", "-m", "app.main"]
