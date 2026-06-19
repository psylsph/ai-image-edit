# Multi-stage build for AI Image Editor (Enhanced)
# Stage 1: Download Real-ESRGAN models
FROM python:3.12-slim AS model-downloader

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.cache/ai-image-edit/realesrgan

RUN wget -q -O /root/.cache/ai-image-edit/realesrgan/RealESRGAN_x2plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

RUN wget -q -O /root/.cache/ai-image-edit/realesrgan/RealESRGAN_x4plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# Stage 2: Final image
FROM python:3.12-slim

EXPOSE 8000
EXPOSE 9090

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    # Fonts for text rendering in comparison images
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /home/app

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy model cache from downloader stage
COPY --from=model-downloader /root/.cache/ai-image-edit /root/.cache/ai-image-edit

# Copy application code
COPY . /home/app/

# Create non-root user and set up temp directory
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /home/app && \
    mkdir -p /tmp/ai-image-edit && \
    chown -R appuser:appuser /tmp/ai-image-edit

USER appuser

# Enhanced health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; r=urllib.request.urlopen('http://localhost:8000/health'); assert r.status==200" || exit 1

# Run with monitoring enabled by default
ENV ENABLE_METRICS=true
ENV METRICS_PORT=9090

CMD ["python", "-m", "app.main"]
