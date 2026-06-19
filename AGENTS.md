# AGENTS.md - AI Image Editor Development Guide

## Build, Lint, and Test Commands

### Install Dependencies
```bash
pip install -r requirements.txt
pip install pytest pytest-cov pylint
pip install rembg
pip install git+https://github.com/xinntao/Real-ESRGAN.git
```

### Run Application
```bash
python -m app.main
# Or: uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Linting
```bash
# Run pylint on entire codebase
pylint app/ models/

# Run pylint on specific file
pylint app/pipeline/grain.py

# Run pylint with specific rating threshold
pylint app/ --fail-under=8.0
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage (target: 80%)
pytest --cov=app --cov-report=term-missing --cov-report=html

# Run single test file
pytest tests/test_grain.py

# Run single test
pytest tests/test_pipeline.py::test_preprocess_image

# Run tests matching pattern
pytest -k "grain" -v

# Run with verbose output
pytest -v --tb=short
```

### Docker
```bash
# Build image
docker build -t ai-image-edit:latest .

# Run container
docker run -p 8000:8000 ai-image-edit:latest

# Build and run with compose
docker-compose up --build -d

# Run tests in container
docker run ai-image-edit:latest python -m pytest

# View logs
docker-compose logs -f

# Stop containers
docker-compose down
```

### Generate Coverage Report
```bash
pytest --cov=app --cov-report=term-missing
open htmlcov/index.html  # View HTML report
```

---

## Code Style Guidelines

### Imports
- Standard library imports first, then third-party, then local
- Use absolute imports: `from pipeline.grain import apply_film_grain`
- Group imports by type with blank lines between groups
- Never use `import *`

```python
# Correct order
import io
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pipeline.grain import apply_film_grain
```

### Formatting
- Line length: 120 characters max
- Use 4 spaces for indentation (no tabs)
- Use Black for automatic formatting: `black app/ tests/`
- Add trailing commas in multi-line calls

### Types
- All function signatures must have type hints
- Use `typing` module for complex types
- Return types required for all public functions
- Private helpers may omit types for brevity

```python
from typing import Optional, Tuple, Dict

def process_image(image: Image.Image, scale: int = 2) -> Tuple[Image.Image, Optional[Image.Image]]:
    ...
```

### Naming Conventions
- **Functions**: `snake_case` (e.g., `apply_film_grain`)
- **Classes**: `PascalCase` (e.g., `ImageProcessor`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_DIMENSION`)
- **Private functions**: leading underscore (e.g., `_preprocess`)
- **Variables**: `snake_case` (e.g., `blur_strength`)
- **Type variables**: `PascalCase` (e.g., `T`)

### Error Handling
- Use specific exceptions (not bare `except:`)
- Log errors with meaningful messages
- Propagate model loading failures gracefully
- Return original image on processing errors

```python
try:
    model = _load_model()
except (RuntimeError, ImportError) as e:
    print(f"Warning: Model unavailable ({e}). Returning original.")
    return image, debug_image
```

### CPU/GPU Enforcement
- Use GPU if available, fall back to CPU automatically
- Get device via `from models.model_downloader import get_device`
- Use `torch.device("cuda")` if available, otherwise `torch.device("cpu")`
- Never hardcode `torch.device("cpu")` - always use the helper function

```python
from models.model_downloader import get_device

device = get_device()  # Returns cuda if available, cpu otherwise
model = model.to(device)
```

### Documentation
- All public functions need docstrings
- Document parameters and return values
- Include example usage for complex functions

```python
def upscale_image(image: Image.Image, scale: int = 2) -> Image.Image:
    """Upscale image using Real-ESRGAN.
    
    Args:
        image: Input PIL Image
        scale: Upscaling factor (2 or 4)
    
    Returns:
        Upscaled PIL Image
    """
```

### File Organization
```
app/
├── main.py              # FastAPI + Gradio entry point
├── pipeline/
│   ├── __init__.py      # Orchestrator exports
│   ├── background.py    # BiRefNet processing
│   ├── grain.py         # Film grain effect
│   ├── upscale.py       # Real-ESRGAN upscaling
│   └── object_removal.py # Stub (not implemented)
models/
└── model_downloader.py  # HuggingFace downloads
tests/
├── test_grain.py
├── test_pipeline.py
└── conftest.py
```

### Testing Requirements
- Aim for 80% code coverage
- Mock ML model loading in unit tests
- Test edge cases: empty, small images, max values
- Integration tests verify full pipeline
- Use pytest fixtures for common setup

```python
@pytest.fixture
def sample_image():
    return Image.new('RGB', (256, 256), color='white')

def test_preprocess_resizes_correctly(sample_image):
    result = preprocess_image(sample_image, max_dimension=100)
    assert max(result.size) == 100
```

### Gradio Interface
- Use Gradio 6.x API
- Include all processing parameters as inputs
- Return final image + intermediate debug images
- Provide download button for results

### Performance Notes
- CPU-only: expect 10-60s per operation
- Pre-process to 1536px max dimension
- Cache loaded models globally
- Log progress for long operations
