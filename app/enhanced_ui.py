"""Enhanced Gradio UI with before/after comparison and presets."""

import gradio as gr
from typing import Optional, Tuple
from PIL import Image

from app.pipeline import process_pipeline
from app.models import ProcessRequest, PRESETS
from app.comparison import create_comparison_overlay, create_diff_overlay
from app.config import settings
from app.exceptions import ImageProcessingError
import logging

logger = logging.getLogger(__name__)


def validate_and_process(
    image: Image.Image,
    enable_background_blur: bool,
    blur_strength: int,
    enable_grain: bool,
    grain_intensity: float,
    enable_upscale: bool,
    upscale_factor: int
) -> Tuple[
    Optional[Image.Image],  # final_image
    Optional[Image.Image],  # comparison_image
    Optional[Image.Image],  # diff_image
    Optional[Image.Image],  # bg_mask
    Optional[Image.Image],  # grain_result
    Optional[Image.Image],  # upscale_result
    str  # status_message
]:
    """Validate input and process image with enhanced outputs.
    
    Returns:
        Tuple of (final, comparison, diff, bg_mask, grain, upscale, status)
    """
    if image is None:
        return None, None, None, None, None, None, "⚠️ Please upload an image first."
    
    try:
        # Validate
        request = ProcessRequest(
            enable_background_blur=enable_background_blur,
            blur_strength=blur_strength,
            enable_grain=enable_grain,
            grain_intensity=grain_intensity,
            enable_upscale=enable_upscale,
            upscale_factor=upscale_factor
        )
        
        logger.info(f"Processing with settings: {request.model_dump()}")
        
        # Process
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
        
        # Create comparison
        comparison_image = create_comparison_overlay(image, final_image)
        
        # Create diff overlay
        diff_image = create_diff_overlay(image, final_image)
        
        # Get intermediate results
        bg_mask = result.get('bg_mask', None)
        grain_result = result.get('grain_result', None)
        upscale_result = result.get('upscale_result', None)
        
        status = f"""✅ **Processing complete!**

**Settings applied:**
{'🌫️ Background Blur: ON (Strength: ' + str(request.blur_strength) + ')' if request.enable_background_blur else '🌫️ Background Blur: OFF'}
{'🎬 Film Grain: ON (Intensity: ' + str(request.grain_intensity) + ')' if request.enable_grain else '🎬 Film Grain: OFF'}
{'🔍 Upscaling: ON (' + str(request.upscale_factor) + 'x)' if request.enable_upscale else '🔍 Upscaling: OFF'}

**Output size:** {final_image.size[0]} x {final_image.size[1]} px
"""
        
        return final_image, comparison_image, diff_image, bg_mask, grain_result, upscale_result, status
        
    except ImageProcessingError as e:
        logger.error(f"Processing error: {e.message}")
        return None, None, None, None, None, None, f"❌ {e.message}\n💡 {e.suggestion}"
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return None, None, None, None, None, None, f"❌ Error: {str(e)}"


def apply_preset(preset_name: str) -> Tuple[int, int, float, bool, bool, bool]:
    """Apply a preset configuration.
    
    Args:
        preset_name: Name of the preset
    
    Returns:
        Tuple of (bg_blur, blur_strength, grain_intensity, grain_enabled, upscale_enabled, upscale_factor)
    """
    if preset_name not in PRESETS:
        preset_name = "minimal"
    
    preset = PRESETS[preset_name]
    settings = preset.settings
    
    return (
        1 if settings.enable_background_blur else 0,
        settings.blur_strength,
        settings.grain_intensity,
        settings.enable_grain,
        settings.enable_upscale,
        settings.upscale_factor
    )


def reset_to_defaults() -> Tuple[int, int, float, bool, bool, bool]:
    """Reset to default settings."""
    return (
        1,  # bg_blur enabled
        5,  # blur_strength
        0.5,  # grain_intensity
        True,  # grain_enabled
        True,  # upscale_enabled
        2  # upscale_factor
    )


def create_enhanced_ui():
    """Create enhanced Gradio UI with comparison and presets."""
    
    device_info = settings.APP_VERSION
    
    css = """
    .container { max-width: 100%; margin: auto; }
    .primary-btn { background-color: #4CAF50; color: white; }
    .preset-btn { 
        background-color: #2196F3; 
        color: white; 
        margin: 4px;
        min-width: 100px;
    }
    .title { font-size: 1.5rem !important; font-weight: bold; }
    .output-image img { max-width: 100%; height: auto; }
    .input-image img { max-width: 100%; height: auto; }
    .gr-tab { min-width: 80px !important; padding: 8px 12px !important; }
    .gr-slider { min-width: 100px !important; }
    .gr-checkbox { min-width: auto !important; }
    .gr-radio { min-width: auto !important; }
    .gr-button { min-width: 120px !important; }
    .gr-image { max-width: 100% !important; }
    .comparison-container {
        display: flex;
        gap: 10px;
        justify-content: center;
    }
    @media (max-width: 768px) {
        .gr-row { flex-direction: column !important; }
        .gr-column { min-width: 100% !important; width: 100% !important; }
        .title { font-size: 1.2rem !important; }
        .gr-group { padding: 10px !important; }
        .preset-btn { width: 100%; margin: 2px 0; }
    }
    """
    
    with gr.Blocks(title="AI Image Editor - Enhanced", css=css) as demo:
        gr.Markdown(f"# 🎨 AI Image Editor - Enhanced UI", elem_classes=["title"])
        
        with gr.Row(equal_height=False):
            with gr.Column(min_width=280, scale=1):
                # Input
                input_image = gr.Image(
                    sources=["upload", "clipboard"],
                    type="pil",
                    label="📁 Upload Image",
                    elem_classes=["input-image"],
                    height=280
                )
                
                # Presets
                with gr.Group(elem_classes=["gr-group"]):
                    gr.Markdown("### ⚡ Quick Presets")
                    
                    with gr.Row():
                        portrait_btn = gr.Button("👤 Portrait", size="sm", elem_classes=["preset-btn"])
                        landscape_btn = gr.Button("🏞️ Landscape", size="sm", elem_classes=["preset-btn"])
                    with gr.Row():
                        vintage_btn = gr.Button("🎬 Vintage", size="sm", elem_classes=["preset-btn"])
                        minimal_btn = gr.Button("✨ Minimal", size="sm", elem_classes=["preset-btn"])
                
                # Manual controls
                with gr.Group(elem_classes=["gr-group"]):
                    gr.Markdown("### ⚙️ Manual Controls")
                    
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
                        elem_classes=["gr-slider"]
                    )
                    
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
                        elem_classes=["gr-slider"]
                    )
                    
                    upscale = gr.Checkbox(
                        value=True,
                        label="🔍 Upscaling",
                        elem_classes=["gr-checkbox"]
                    )
                    upscale_factor = gr.Radio(
                        choices=[2, 4],
                        value=2,
                        label="Scale",
                        elem_classes=["gr-radio"]
                    )
                
                # Action buttons
                with gr.Row():
                    process_btn = gr.Button(
                        "🚀 Process",
                        variant="primary",
                        size="lg",
                        elem_classes=["gr-button"]
                    )
                    reset_btn = gr.Button(
                        "🔄 Reset",
                        variant="secondary",
                        elem_classes=["gr-button"]
                    )
                
                status_msg = gr.Markdown("", elem_classes=["status-msg"])
            
            with gr.Column(min_width=400, scale=2):
                # Output tabs
                with gr.Tabs(elem_classes=["gr-tabs"]):
                    with gr.Tab("✨ Final Result", elem_id="tab-final"):
                        output_image = gr.Image(
                            type="pil",
                            label="Processed Image",
                            elem_classes=["output-image"],
                            height=400
                        )
                    
                    with gr.Tab("🔄 Comparison", elem_id="tab-comparison"):
                        comparison_image = gr.Image(
                            type="pil",
                            label="Before vs After (Side by Side)",
                            elem_classes=["output-image"],
                            height=400
                        )
                    
                    with gr.Tab("🔍 Differences", elem_id="tab-diff"):
                        diff_image = gr.Image(
                            type="pil",
                            label="Difference Overlay (Changes Highlighted)",
                            elem_classes=["output-image"],
                            height=400
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
        
        # Info section
        gr.Markdown(f"""
        ### ℹ️ About
        **Enhanced** AI image processing with before/after comparison and quick presets.
        
        **Features:**
        - 🌫️ Background Blur with BiRefNet
        - 🎬 Luminance-aware film grain
        - 🔍 Real-ESRGAN upscaling (2x/4x)
        - 🔄 Before/after comparison
        - 🔍 Difference overlay
        - ⚡ Quick presets for common use cases
        
        **Presets:**
        - **👤 Portrait**: Background blur + light grain + 2x upscale
        - **🏞️ Landscape**: No blur + medium grain + 4x upscale
        - **🎬 Vintage**: Heavy grain, no processing
        - **✨ Minimal**: Light processing only
        
        **Limits:** Max {settings.MAX_IMAGE_SIZE_MB:.0f}MB file size
        """, elem_classes=["gr-markdown"])
        
        # Event handlers
        process_btn.click(
            fn=validate_and_process,
            inputs=[
                input_image, bg_blur, bg_strength,
                grain, grain_intensity,
                upscale, upscale_factor
            ],
            outputs=[
                output_image, comparison_image, diff_image,
                debug_bg, debug_grain, debug_upscale,
                status_msg
            ]
        )
        
        # Preset buttons
        portrait_btn.click(
            fn=lambda: apply_preset("portrait"),
            outputs=[bg_blur, bg_strength, grain_intensity, grain, upscale, upscale_factor]
        )
        
        landscape_btn.click(
            fn=lambda: apply_preset("landscape"),
            outputs=[bg_blur, bg_strength, grain_intensity, grain, upscale, upscale_factor]
        )
        
        vintage_btn.click(
            fn=lambda: apply_preset("vintage"),
            outputs=[bg_blur, bg_strength, grain_intensity, grain, upscale, upscale_factor]
        )
        
        minimal_btn.click(
            fn=lambda: apply_preset("minimal"),
            outputs=[bg_blur, bg_strength, grain_intensity, grain, upscale, upscale_factor]
        )
        
        # Reset button
        reset_btn.click(
            fn=reset_to_defaults,
            outputs=[bg_blur, bg_strength, grain_intensity, grain, upscale, upscale_factor]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_enhanced_ui()
    demo.launch()
