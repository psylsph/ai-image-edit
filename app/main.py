import tempfile
import os
from pathlib import Path
from PIL import Image
import gradio as gr
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging

from app.pipeline import process_pipeline
from app.models import ProcessRequest, PRESETS
from app.exceptions import ImageProcessingError, FileSizeError, ValidationError
from app.config import settings
from models.model_downloader import get_device_string

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


def validate_image_size(image: Image.Image) -> None:
    """Validate image size constraints."""
    # Check file size (approximate from PIL Image)
    import io
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    size_mb = len(buffer.getvalue()) / (1024 * 1024)
    
    if size_mb > settings.MAX_IMAGE_SIZE_MB:
        raise FileSizeError(size_mb, settings.MAX_IMAGE_SIZE_MB)
    
    logger.info(f"Image size validated: {size_mb:.2f}MB")


def process_image_handler(
    image: Image.Image,
    enable_background_blur: bool,
    blur_strength: int,
    enable_grain: bool,
    grain_intensity: float,
    enable_upscale: bool,
    upscale_factor: int
):
    """Handle image processing request from Gradio."""
    if image is None:
        return None, None, None, None, None, "⚠️ Please upload an image first."
    
    try:
        # Validate input
        request = ProcessRequest(
            enable_background_blur=enable_background_blur,
            blur_strength=blur_strength,
            enable_grain=enable_grain,
            grain_intensity=grain_intensity,
            enable_upscale=enable_upscale,
            upscale_factor=upscale_factor
        )
        
        # Validate image size
        validate_image_size(image)
        
        logger.info(f"Processing image: {image.size}, settings: {request.model_dump()}")
        
        # Process pipeline
        result = process_pipeline(
            image,
            enable_background_blur=request.enable_background_blur,
            blur_strength=request.blur_strength,
            enable_grain=request.enable_grain,
            grain_intensity=request.grain_intensity,
            enable_upscale=request.enable_upscale,
            upscale_factor=request.upscale_factor,
            debug=True
        )

        final_image = result['final']
        bg_mask = result.get('bg_mask', None)
        grain_result = result.get('grain_result', None)
        upscale_result = result.get('upscale_result', None)

        temp_file = None
        if final_image:
            temp_dir = settings.TEMP_DIR
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = Path(temp_dir) / f"processed_{id(final_image)}.png"
            final_image.save(temp_file, "PNG")

        status = "✅ Processing complete!"
        logger.info(f"Successfully processed image: {temp_file}")

        return final_image, temp_file, bg_mask, grain_result, upscale_result, status

    except FileSizeError as e:
        logger.error(f"File size error: {e.message}")
        return None, None, None, None, None, f"❌ {e.message}\n💡 {e.suggestion}"
    
    except ValidationError as e:
        logger.error(f"Validation error: {e.message}")
        return None, None, None, None, None, f"❌ {e.message}\n💡 {e.suggestion}"
    
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        error_msg = f"❌ Error: {str(e)}\n\n💡 Try:\n• Using a smaller image\n• Reducing processing steps\n• Different settings"
        return None, None, None, None, None, error_msg


def create_gradio_app():
    """Create and return the Gradio application."""
    device_info = get_device_string()
    css = """
    .container { max-width: 100%; margin: auto; }
    .primary-btn { background-color: #4CAF50; color: white; }
    .title { font-size: 1.5rem !important; }
    .output-image img { max-width: 100%; height: auto; }
    .input-image img { max-width: 100%; height: auto; }
    .gr-tab { min-width: 80px !important; padding: 8px 12px !important; }
    .gr-slider { min-width: 100px !important; }
    .gr-checkbox { min-width: auto !important; }
    .gr-radio { min-width: auto !important; }
    .gr-button { min-width: 120px !important; }
    .gr-image { max-width: 100% !important; }
    @media (max-width: 768px) {
        .gr-row { flex-direction: column !important; }
        .gr-column { min-width: 100% !important; width: 100% !important; }
        .title { font-size: 1.2rem !important; }
        .gr-group { padding: 10px !important; }
        .gr-markdown { font-size: 0.9rem !important; }
        .gr-slider { width: 100% !important; }
        .gr-button { width: 100% !important; }
    }
    """
    
    with gr.Blocks(title="AI Image Editor", css=css) as demo:
        gr.Markdown(f"# 🎨 AI Image Editor ({device_info})", elem_classes=["title"])

        with gr.Row(equal_height=False):
            with gr.Column(min_width=280, scale=1):
                input_image = gr.Image(
                    sources=["upload", "clipboard"],
                    type="pil",
                    label="📁 Upload Image",
                    elem_classes=["input-image"],
                    height=280
                )

                with gr.Group(elem_classes=["gr-group"]):
                    gr.Markdown("### ⚙️ Processing Options")

                    with gr.Row():
                        bg_blur = gr.Checkbox(
                            value=True,
                            label="🌫️ Background Blur",
                            elem_classes=["gr-checkbox"]
                        )
                        bg_strength = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=5,
                            step=1,
                            label="Blur Strength",
                            elem_classes=["gr-slider"],
                            show_label=True
                        )

                    with gr.Row():
                        grain = gr.Checkbox(
                            value=True,
                            label="🎬 Film Grain",
                            elem_classes=["gr-checkbox"]
                        )
                        grain_intensity = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="Intensity",
                            elem_classes=["gr-slider"],
                            show_label=True
                        )

                    with gr.Row():
                        upscale = gr.Checkbox(
                            value=True,
                            label="🔍 Upscaling",
                            elem_classes=["gr-checkbox"]
                        )
                        upscale_factor = gr.Radio(
                            choices=[2, 4],
                            value=2,
                            label="Scale",
                            elem_classes=["gr-radio"],
                            show_label=True
                        )

                submit_btn = gr.Button(
                    "🚀 Process Image",
                    variant="primary",
                    size="lg",
                    elem_classes=["gr-button"]
                )

                status_msg = gr.Markdown("", elem_classes=["status-msg"])

            with gr.Column(min_width=300, scale=2):
                with gr.Tabs(elem_classes=["gr-tabs"]):
                    with gr.Tab("✨ Final Output", elem_id="tab-final"):
                        output_image = gr.Image(
                            type="pil",
                            label="Processed Image",
                            elem_classes=["output-image"],
                            height=400
                        )
                        download_btn = gr.DownloadButton(
                            "💾 Download Image",
                            value=None,
                            variant="secondary",
                            elem_classes=["gr-button"]
                        )

                    with gr.Tab("🌫️ Background", elem_id="tab-bg"):
                        debug_bg = gr.Image(
                            type="pil", 
                            label="Background Mask", 
                            elem_classes=["output-image"],
                            height=400
                        )

                    with gr.Tab("🎬 Grain", elem_id="tab-grain"):
                        debug_grain = gr.Image(
                            type="pil", 
                            label="After Film Grain", 
                            elem_classes=["output-image"],
                            height=400
                        )

                    with gr.Tab("🔍 Upscaled", elem_id="tab-upscale"):
                        debug_upscale = gr.Image(
                            type="pil", 
                            label="After Upscaling", 
                            elem_classes=["output-image"],
                            height=400
                        )

        gr.Markdown(f"""
        ### ℹ️ About
        AI image processing on {device_info}.
        
        **Features:**
        - 🌫️ **Background Blur**: BiRefNet foreground/background separation
        - 🎬 **Film Grain**: Luminance-aware grain effect
        - 🔍 **Upscaling**: Real-ESRGAN 2×/4× enlargement
        
        **Limits:**
        - Max file size: {settings.MAX_IMAGE_SIZE_MB:.0f}MB
        - Max dimension: {settings.MAX_IMAGE_DIMENSION}px
        """, elem_classes=["gr-markdown"])

        submit_btn.click(
            fn=process_image_handler,
            inputs=[
                input_image, bg_blur, bg_strength,
                grain, grain_intensity,
                upscale, upscale_factor
            ],
            outputs=[
                output_image, download_btn,
                debug_bg, debug_grain, debug_upscale,
                status_msg
            ]
        )

    return demo


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
    gradio_app = create_gradio_app()
    app = gr.mount_gradio_app(
        app,
        gradio_app,
        path="/",
        favicon_path=None
    )

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
