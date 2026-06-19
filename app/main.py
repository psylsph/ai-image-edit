import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging

from app.enhanced_ui import create_enhanced_ui
from app.models import PRESETS
from app.config import settings
from app.monitoring import start_metrics_server
from models.model_downloader import get_device_string

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


def create_fastapi_app():
    """Create and configure the FastAPI application."""
    device_info = get_device_string()
    app = FastAPI(
        title="AI Image Editor API",
        description=f"AI image processing API running on {device_info}",
        version="1.0.0"
    )
    
    # Add rate limit exception handler
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Define FastAPI routes BEFORE mounting Gradio
    @app.get("/health")
    async def health_check():
        """Health check endpoint (no rate limiting)."""
        return {
            "status": "healthy",
            "message": "AI Image Editor is running",
            "device": device_info,
            "version": settings.APP_VERSION
        }

    @app.get("/presets")
    async def list_presets():
        """List available presets."""
        return {
            "presets": [
                {
                    "name": name,
                    "description": preset.description,
                    "settings": preset.settings.model_dump()
                }
                for name, preset in PRESETS.items()
            ]
        }

    # Mount Gradio app at root path
    gradio_app = create_enhanced_ui()
    app = gr.mount_gradio_app(
        app,
        gradio_app,
        path="/",
        favicon_path=None
    )

    # Start Prometheus metrics server
    if settings.ENABLE_METRICS:
        start_metrics_server(settings.METRICS_PORT)

    return app


app = create_fastapi_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1
    )
