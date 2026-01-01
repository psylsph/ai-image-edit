# CPU-Only AI Image Editor

A browser-based image processing application using FastAPI and Gradio. All processing is performed server-side using CPU or GPU if available.

## Features

- **Background Blur**: Uses BiRefNet for intelligent foreground/background separation
- **Film Grain**: Adds authentic film grain effect with luminance-aware scaling
- **Image Upscaling**: Uses Real-ESRGAN for 2× and 4× enlargement
- **Interactive UI**: Built with Gradio for easy image upload and processing

## Requirements

- Python 3.10 or higher
- CPU or GPU (automatically detects)
- ~160MB disk space for models (downloaded automatically)

## Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Install PyTorch with CPU support**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

## Running the Application

Start the server:
```bash
cd /path/to/ai-image-edit
python -m app.main
```

Or with uvicorn directly:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open your browser and navigate to:
```
http://localhost:8000
```

## Docker Deployment

### Quick Start

Build and run the container:

```bash
docker build -t ai-image-edit:latest .
docker run -p 8000:8000 ai-image-edit:latest
```

### Docker Compose

For easier management, use Docker Compose:

```bash
docker compose up --build -d
```

The application will be available at http://localhost:8000

### Persistent Model Cache

Models are cached in `/root/.cache/ai-image-edit/`. Mount a volume to persist downloaded models:

```bash
docker run -p 8000:8000 -v $HOME/.cache/ai-image-edit:/root/.cache/ai-image-edit ai-image-edit:latest
```

Or with Docker Compose, the cache volume is managed automatically.

### Building from Source

```bash
# Build the image
docker build -t ai-image-edit:latest .

# Run tests inside the container
docker run ai-image-edit:latest python -m pytest
```

## Model Downloads

On first run, the application will automatically download required models:

- **BiRefNet** (~100MB): Background removal model for foreground/background separation
- **Real-ESRGAN x2plus** (~17MB): Image upscaling model for 2× scaling
- **Real-ESRGAN x4plus** (~64MB): Image upscaling model for 4× scaling

Models are cached in `~/.cache/ai-image-edit/` for Real-ESRGAN, and rembg uses its own cache.

## Performance Expectations

Processing times (approximate, varies by hardware):

| Operation | Time (per image) |
|-----------|------------------|
| Background Blur | 2-5 seconds |
| Film Grain | 1-2 seconds |
| Upscaling 2× | 10-30 seconds (CPU) |
| Upscaling 4× | 20-60 seconds (CPU) |
| Full Pipeline | 15-90 seconds |

Times are for 1536×1536px images. Smaller images process faster.

**Note**: Real-ESRGAN runs on CPU only. GPU upscaling requires CUDA-compatible hardware.

## Image Pre-processing

Images are automatically resized to a maximum dimension of 1536 pixels before processing to balance quality and performance. The original aspect ratio is preserved.

## Project Structure

```
ai-image-edit/
├── requirements.txt
├── README.md
├── PLAN.md
├── AGENTS.md
├── app/
│   ├── main.py              # FastAPI + Gradio application
│   └── pipeline/
│       ├── __init__.py      # Pipeline orchestrator
│       ├── background.py    # Background blur (BiRefNet)
│       ├── grain.py         # Film grain effect
│   ├── upscale.py       # Image upscaling (Real-ESRGAN x2plus and x4plus)
│       └── object_removal.py # Stub for future implementation
└── models/
    └── model_downloader.py  # Automatic model download utilities
```

## API

The application serves a Gradio interface at the root path. A health check endpoint is available:

```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "message": "AI Image Editor is running",
  "device": "CUDA"
}
```

## Processing Pipeline

Images are processed in this order:

1. **Pre-process**: Resize to ≤1536px max dimension
2. **Background Blur**: Optional, uses BiRefNet for matting
3. **Film Grain**: Optional, adds luminance-aware grain
4. **Upscaling**: Optional, 2× or 4× using Real-ESRGAN

Intermediate outputs are displayed in tabs for debugging.

## Troubleshooting

### Slow Performance
Real-ESRGAN runs on CPU only. Processing large images (1536px+) can take 20-60 seconds. This is expected for CPU-only processing.

### Out of Memory
The application pre-processes images to 1536px. If you still experience memory issues, try closing other applications.

### Model Download Fails
Ensure you have an internet connection. Models are downloaded from GitHub releases.

## Future Extensions

Planned features (not yet implemented):

- Async job queue for large batches
- WebSocket progress updates
- Additional filter effects
- GPU acceleration for Real-ESRGAN using CUDA

## Development

See [AGENTS.md](AGENTS.md) for build, lint, and test commands.

## License

This project is for educational and creative use. Please respect the licenses of the underlying models (BiRefNet, Real-ESRGAN).
