"""Enhanced Gradio UI with before/after comparison and presets."""

import gradio as gr
from typing import Optional, Tuple
from PIL import Image

from app.pipeline import process_pipeline
from app.pipeline.background import BG_MODELS
from app.pipeline.upscale import UPSCALE_MODES
from app.models import ProcessRequest, PRESETS
from app.comparison import create_comparison_overlay, create_diff_overlay
from app.config import settings
from app.exceptions import ImageProcessingError
import logging

logger = logging.getLogger(__name__)

# ── Global UI config ──────────────────────────────────────────────────
PRESET_NAMES = {
    "portrait": ("👤", "Portrait"),
    "landscape": ("🏞️", "Landscape"),
    "vintage": ("🎬", "Vintage"),
    "minimal": ("✨", "Minimal"),
}


def validate_and_process(
    image: Image.Image,
    enable_background_blur: bool,
    blur_strength: int,
    bg_model: str,
    bg_mode: str,
    enable_grain: bool,
    grain_intensity: float,
    enable_upscale: bool,
    upscale_factor: int,
    upscale_mode: str,
) -> Tuple[
    Optional[Image.Image],  # final_image
    Optional[Image.Image],  # comparison_image
    Optional[Image.Image],  # diff_image
    Optional[Image.Image],  # bg_mask
    Optional[Image.Image],  # grain_result
    Optional[Image.Image],  # upscale_result
    str  # status_message
]:
    """Validate input and process image with enhanced outputs."""
    if image is None:
        return None, None, None, None, None, None, "⚠️ Please upload an image first."

    try:
        request = ProcessRequest(
            enable_background_blur=enable_background_blur,
            blur_strength=blur_strength,
            bg_model=bg_model,
            bg_mode=bg_mode,
            enable_grain=enable_grain,
            grain_intensity=grain_intensity,
            enable_upscale=enable_upscale,
            upscale_factor=upscale_factor,
            upscale_mode=upscale_mode,
        )

        logger.info(f"Processing with settings: {request.model_dump()}")

        result = process_pipeline(
            image,
            enable_background_blur=request.enable_background_blur,
            blur_strength=request.blur_strength,
            bg_model=request.bg_model,
            bg_mode=request.bg_mode,
            enable_grain=request.enable_grain,
            grain_intensity=request.grain_intensity,
            enable_upscale=request.enable_upscale,
            upscale_factor=request.upscale_factor,
            upscale_mode=request.upscale_mode,
            debug=True
        )

        final_image = result['final']
        comparison_image = create_comparison_overlay(image, final_image)
        diff_image = create_diff_overlay(image, final_image)

        bg_mask = result.get('bg_mask', None)
        grain_result = result.get('grain_result', None)
        upscale_result = result.get('upscale_result', None)

        model_label = BG_MODELS.get(request.bg_model, request.bg_model)
        mode_label = "Remove (transparent)" if request.bg_mode == "remove" else "Blur"
        up_label = UPSCALE_MODES.get(request.upscale_mode, request.upscale_mode)
        status = f"""✅ **Processing complete!**

**Settings applied:**
{'🌫️ Background: ON (' + mode_label + ', ' + model_label + ')' if request.enable_background_blur else '🌫️ Background: OFF'}
{'🎬 Film Grain: ON (Intensity: ' + str(request.grain_intensity) + ')' if request.enable_grain else '🎬 Film Grain: OFF'}
{'🔍 Upscaling: ON (' + str(request.upscale_factor) + 'x, ' + up_label + ')' if request.enable_upscale else '🔍 Upscaling: OFF'}

**Output size:** {final_image.size[0]} x {final_image.size[1]} px"""

        return final_image, comparison_image, diff_image, bg_mask, grain_result, upscale_result, status

    except ImageProcessingError as e:
        logger.error(f"Processing error: {e.message}")
        return None, None, None, None, None, None, f"❌ {e.message}\n💡 {e.suggestion}"

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return None, None, None, None, None, None, f"❌ Error: {str(e)}"


def apply_preset(preset_name: str) -> tuple:
    """Apply a preset configuration."""
    if preset_name not in PRESETS:
        preset_name = "minimal"
    preset = PRESETS[preset_name]
    s = preset.settings
    return (
        1 if s.enable_background_blur else 0,
        s.blur_strength,
        s.bg_model,
        s.bg_mode,
        s.grain_intensity,
        s.enable_grain,
        s.enable_upscale,
        s.upscale_factor,
        s.upscale_mode,
    )


def reset_to_defaults() -> tuple:
    """Reset to default settings."""
    return ("u2net", "blur", 5, 0.5, True, True, 2, "interp")


# ── Mobile-first CSS (Gradio 6.x compatible) ────────────────────────
MOBILE_CSS = """
/* ══ Design tokens ══ */
:root {
  --accent: #4CAF50;
  --accent2: #2196F3;
  --radius: 12px;
  --pad: 14px;
  --touch-min: 48px;
  --bg-app: #0f0f0f;
  --bg-card: #1a1a1a;
  --bg-card-2: #222;
  --border: #333;
  --text: #e0e0e0;
  --text-dim: #888;

  /* Override Gradio 6 theme vars for dark mobile look */
  --block-background-fill: var(--bg-card) !important;
  --block-border-color: var(--border) !important;
  --block-radius: var(--radius) !important;
  --body-background-fill: var(--bg-app) !important;
  --body-text-color: var(--text) !important;
  --button-primary-background-fill: var(--accent) !important;
  --button-primary-text-color: #fff !important;
  --button-secondary-background-fill: var(--bg-card-2) !important;
  --button-secondary-text-color: var(--text) !important;
  --input-background-fill: var(--bg-card-2) !important;
  --input-border-color: var(--border) !important;
}

/* ══ Body / layout shell ══ */
body, .gradio-container, #root {
  max-width: 600px !important;
  margin: 0 auto !important;
  padding: 8px !important;
  background: var(--bg-app) !important;
  color: var(--text) !important;
}
/* Kill Gradio footer/social cruft */
footer, .gradio-footer, #footer { display: none !important; }

/* ══ Title ══ */
.title h1, .title {
  font-size: 1.3rem !important;
  font-weight: 800 !important;
  text-align: center !important;
  padding: 8px 0 2px !important;
  margin: 0 !important;
}

/* ══ Image upload / display ══ */
.input-image, .output-image {
  border-radius: var(--radius) !important;
  overflow: hidden;
}
.input-image img, .output-image img,
.input-image image, .output-image image {
  width: 100% !important;
  max-height: 45vh !important;
  object-fit: contain !important;
  border-radius: var(--radius) !important;
}

/* ══ Preset pills (horizontal scroll) ══ */
.preset-row {
  display: flex !important;
  gap: 8px !important;
  overflow-x: auto !important;
  -webkit-overflow-scrolling: touch;
  scrollbar-width: none;
  padding: 4px 0 8px !important;
  flex-wrap: nowrap !important;
}
.preset-row::-webkit-scrollbar { display: none !important; }
.preset-btn, .preset-btn button {
  flex: 0 0 auto !important;
  min-width: 90px !important;
  min-height: var(--touch-min) !important;
  border-radius: 24px !important;
  border: 1px solid var(--border) !important;
  background: var(--bg-card-2) !important;
  color: var(--text) !important;
  font-size: 0.85rem !important;
  font-weight: 600 !important;
  padding: 0 16px !important;
  white-space: nowrap !important;
  transition: background 0.15s, border-color 0.15s;
}
.preset-btn:hover, .preset-btn:active {
  background: var(--accent2) !important;
  border-color: var(--accent2) !important;
  color: #fff !important;
}

/* ══ Section groups (presets + adjustments) ══ */
.controls-group {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 10px var(--pad) !important;
  margin: 0 0 12px !important;
}

/* ══ Checkboxes — bigger touch targets ══ */
/* Target Gradio 6 checkbox component via wrapper descendant */
.svelte-1ih3qge, /* checkbox wrapper in some Gradio builds */
input[type="checkbox"] {
  width: 22px !important;
  height: 22px !important;
  min-width: 22px !important;
}
/* Increase checkbox row height for easier tapping */
label:has(input[type="checkbox"]),
.gradio-checkbox > label,
[data-testid="checkbox"] label {
  min-height: 44px !important;
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
  font-size: 0.95rem !important;
}

/* ══ Sliders — bigger grab area ══ */
.gr-slider input[type="range"],
input[type="range"] {
  width: 100% !important;
  height: 36px !important;
  cursor: pointer !important;
  -webkit-appearance: none !important;
  appearance: none !important;
  background: transparent !important;
}
/* Webkit thumb */
input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none !important;
  appearance: none !important;
  width: 28px !important;
  height: 28px !important;
  border-radius: 50% !important;
  background: var(--accent) !important;
  border: 2px solid #fff !important;
  cursor: pointer !important;
  margin-top: -10px;
}
/* Firefox thumb */
input[type="range"]::-moz-range-thumb {
  width: 28px !important;
  height: 28px !important;
  border-radius: 50% !important;
  background: var(--accent) !important;
  border: 2px solid #fff !important;
  cursor: pointer !important;
}
/* Webkit track */
input[type="range"]::-webkit-slider-runnable-track {
  height: 8px !important;
  border-radius: 4px !important;
  background: var(--bg-card-2) !important;
}
input[type="range"]::-moz-range-track {
  height: 8px !important;
  border-radius: 4px !important;
  background: var(--bg-card-2) !important;
}

/* ══ Radio (scale chooser) — segmented control ══ */
.gr-radio fieldset {
  display: flex !important;
  gap: 8px !important;
  border: none !important;
  padding: 0 !important;
  margin: 0 !important;
}
.gr-radio label,
.gr-radio .wrap label,
.gradio-radio label {
  flex: 1 !important;
  text-align: center !important;
  padding: 10px 4px !important;
  border-radius: 10px !important;
  border: 2px solid var(--border) !important;
  background: var(--bg-card-2) !important;
  font-size: 0.95rem !important;
  font-weight: 600 !important;
  min-height: 42px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  cursor: pointer !important;
  transition: all 0.15s;
}
.gr-radio input:checked + label,
.gr-radio input:checked ~ label,
.gradio-radio input:checked + label {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #fff !important;
}
/* Hide the actual radio circle — just use the label as button */
.gr-radio input[type="radio"],
.gradio-radio input[type="radio"] {
  position: absolute !important;
  opacity: 0 !important;
  width: 0 !important;
  height: 0 !important;
}

/* ══ Process button — full width, prominent ══ */
.primary, .primary button,
.gradio-button.primary,
[data-testid="primary"] {
  width: 100% !important;
  min-height: var(--touch-min) !important;
  font-size: 1.1rem !important;
  font-weight: 800 !important;
  border-radius: var(--radius) !important;
  border: none !important;
  background: var(--accent) !important;
  color: #fff !important;
  margin: 8px 0 4px !important;
  letter-spacing: 0.3px !important;
  box-shadow: 0 2px 8px rgba(76,175,80,0.3) !important;
}
.primary:active, .primary button:active {
  transform: scale(0.98) !important;
  opacity: 0.9 !important;
}

/* ══ Reset button ══ */
[data-testid="secondary"],
.gradio-button.secondary,
button[class*="secondary"] {
  min-height: 44px !important;
  border-radius: var(--radius) !important;
  border: 1px solid #555 !important;
  background: var(--bg-card-2) !important;
  color: var(--text) !important;
  font-size: 0.9rem !important;
  width: 100% !important;
}

/* ══ Tabs — horizontal scroll for mobile ══ */
.gr-tabs > .tab-nav,
.gradio-tabs > .tab-nav,
[class*="tabs"] > .tab-nav,
.tab-nav {
  display: flex !important;
  overflow-x: auto !important;
  -webkit-overflow-scrolling: touch;
  scrollbar-width: none;
  gap: 0 !important;
  padding: 0 !important;
  margin: 0 0 4px !important;
  border-bottom: 1px solid var(--border) !important;
}
.tab-nav::-webkit-scrollbar { display: none !important; }
.tab-nav button {
  flex: 0 0 auto !important;
  padding: 12px 14px !important;
  font-size: 0.8rem !important;
  white-space: nowrap !important;
  border: none !important;
  background: transparent !important;
  color: var(--text-dim) !important;
  font-weight: 600 !important;
  border-bottom: 3px solid transparent !important;
  cursor: pointer !important;
  transition: color 0.15s, border-color 0.15s;
}
.tab-nav button:hover,
.tab-nav button.selected,
.tab-nav button[aria-selected="true"] {
  color: #fff !important;
  border-bottom-color: var(--accent) !important;
}

/* ══ Status message ══ */
.status-msg {
  font-size: 0.85rem !important;
  line-height: 1.5 !important;
  margin: 4px 0 !important;
  padding: 10px !important;
  border-radius: 10px !important;
  background: var(--bg-card) !important;
}

/* ══ Info / markdown ══ */
.gr-markdown {
  font-size: 0.8rem !important;
  line-height: 1.5 !important;
  color: var(--text-dim) !important;
}

/* ══ Responsive: tablet+ gets wider, more breathing room ══ */
@media (min-width: 768px) {
  body, .gradio-container {
    max-width: 700px !important;
    padding: 16px !important;
  }
  .input-image img, .output-image img {
    max-height: 60vh !important;
  }
}

/* ══ Responsive: very small phones (≤375px) tighten things ══ */
@media (max-width: 375px) {
  body, .gradio-container {
    padding: 4px !important;
  }
  .preset-btn, .preset-btn button {
    min-width: 78px !important;
    padding: 0 12px !important;
    font-size: 0.8rem !important;
  }
  .tab-nav button {
    padding: 10px 10px !important;
    font-size: 0.75rem !important;
  }
}
"""


def create_enhanced_ui():
    """Create mobile-first Gradio UI with comparison and presets."""
    with gr.Blocks(
        title="AI Image Editor",
        css=MOBILE_CSS,
        theme=gr.themes.Soft()
    ) as demo:

        # ── Header ──
        gr.Markdown("# 🎨 AI Image Editor", elem_classes=["title"])

        # ── Upload ──
        input_image = gr.Image(
            sources=["upload", "clipboard"],
            type="pil",
            label="Upload Image",
            elem_classes=["input-image"],
            height=280,
        )

        # ── Presets (horizontal scrollable pills) ──
        with gr.Group(elem_classes=["controls-group"]):
            gr.Markdown("### ⚡ Quick Presets")
            with gr.Row(elem_classes=["preset-row"]):
                portrait_btn = gr.Button("👤 Portrait", elem_classes=["preset-btn"])
                landscape_btn = gr.Button("🏞️ Landscape", elem_classes=["preset-btn"])
                vintage_btn = gr.Button("🎬 Vintage", elem_classes=["preset-btn"])
                minimal_btn = gr.Button("✨ Minimal", elem_classes=["preset-btn"])

        # ── Controls ──
        with gr.Group(elem_classes=["controls-group"]):
            gr.Markdown("### ⚙️ Adjustments")

            bg_blur = gr.Checkbox(value=True, label="🌫️ Background")
            bg_mode = gr.Radio(
                choices=["blur", "remove"],
                value="blur",
                label="Mode",
                elem_classes=["gr-radio"],
            )
            bg_strength = gr.Slider(
                minimum=1, maximum=50, value=5, step=1,
                label="Blur Strength", elem_classes=["gr-slider"],
                visible=True
            )
            bg_model = gr.Dropdown(
                choices=list(BG_MODELS.keys()),
                value="u2net",
                label="Model",
                elem_classes=["gr-dropdown"],
                visible=True,
                allow_custom_value=False,
            )

            grain = gr.Checkbox(value=True, label="🎬 Film Grain")
            grain_intensity = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.5, step=0.1,
                label="Grain Intensity", elem_classes=["gr-slider"],
                visible=True
            )

            upscale = gr.Checkbox(value=True, label="🔍 Upscaling")
            upscale_factor = gr.Radio(
                choices=["1x", "2x", "4x"], value="2x",
                label="Scale", elem_classes=["gr-radio"],
                visible=True
            )
            upscale_mode = gr.Dropdown(
                choices=list(UPSCALE_MODES.keys()),
                value="interp",
                label="Mode",
                elem_classes=["gr-dropdown"],
                allow_custom_value=False,
            )

        # ── Action buttons ──
        process_btn = gr.Button("🚀 Process Image", variant="primary", elem_classes=["primary"])
        status_msg = gr.Markdown("", elem_classes=["status-msg"])

        # ── Output tabs ──
        with gr.Tabs(elem_classes=["gr-tabs"]):
            with gr.TabItem("✨ Result"):
                output_image = gr.Image(
                    type="pil", label="Processed Image",
                    elem_classes=["output-image"], height=400
                )
            with gr.TabItem("🔄 Before/After"):
                comparison_image = gr.Image(
                    type="pil", label="Before vs After",
                    elem_classes=["output-image"], height=400
                )
            with gr.TabItem("🔍 Diff"):
                diff_image = gr.Image(
                    type="pil", label="Difference Overlay",
                    elem_classes=["output-image"], height=400
                )
            with gr.TabItem("🌫️ Mask"):
                debug_bg = gr.Image(
                    type="pil", label="Background Mask",
                    elem_classes=["output-image"], height=400
                )
            with gr.TabItem("🎬 Grain"):
                debug_grain = gr.Image(
                    type="pil", label="After Film Grain",
                    elem_classes=["output-image"], height=400
                )
            with gr.TabItem("🔍 Upscaled"):
                debug_upscale = gr.Image(
                    type="pil", label="After Upscaling",
                    elem_classes=["output-image"], height=400
                )

        # ── Info ──
        gr.Markdown(f"""
### ℹ️ About
**AI Image Editor** — mobile-optimised
- 🌫️ Background Blur (8 rembg models)
- 🎬 Luminance-aware film grain
- 🔍 Upscaling: AI (Real-ESRGAN) or Fast (interpolation)
- Max file size: {settings.MAX_IMAGE_SIZE_MB:.0f}MB
""", elem_classes=["gr-markdown"])

        # ── Toggle visibility when mode changes ──
        # Hide blur strength slider when in "remove" mode
        bg_mode.change(
            fn=lambda m: gr.update(visible=(m == "blur")),
            inputs=bg_mode,
            outputs=bg_strength,
        )
        grain.change(
            fn=lambda v: gr.update(visible=v),
            inputs=grain,
            outputs=grain_intensity,
        )
        upscale.change(
            fn=lambda v: [gr.update(visible=v), gr.update(visible=v)],
            inputs=upscale,
            outputs=[upscale_factor, upscale_mode],
        )

        # ── Map radio "1x" / "2x" / "4x" back to int ──
        def _parse_scale(val: str) -> int:
            return int(val.replace("x", ""))

        # ── Process handler ──
        def _process_wrapper(img, bg, bmod, bstr, bgmdl, grn, gint, up, scale_str, up_mode):
            scale = _parse_scale(scale_str)
            return validate_and_process(img, bg, bstr, bgmdl, bmod, grn, gint, up, scale, up_mode)

        process_btn.click(
            fn=_process_wrapper,
            inputs=[
                input_image, bg_blur, bg_mode, bg_strength, bg_model,
                grain, grain_intensity,
                upscale, upscale_factor, upscale_mode,
            ],
            outputs=[
                output_image, comparison_image, diff_image,
                debug_bg, debug_grain, debug_upscale,
                status_msg,
            ]
        )

        # ── Presets ──
        def _preset_wrapper(name):
            bg_v, bs, bgmdl, bgmod, gi, grn_en, up_en, up_sc, up_sc_mode = apply_preset(name)
            return (
                bg_v, bgmod, bs, bgmdl, gi, grn_en, up_en,
                f"{up_sc}x" if up_sc in (1, 2, 4) else "2x", up_sc_mode
            )

        portrait_btn.click(
            fn=lambda: _preset_wrapper("portrait"),
            outputs=[bg_blur, bg_mode, bg_strength, bg_model, grain_intensity, grain, upscale, upscale_factor, upscale_mode]
        )
        landscape_btn.click(
            fn=lambda: _preset_wrapper("landscape"),
            outputs=[bg_blur, bg_mode, bg_strength, bg_model, grain_intensity, grain, upscale, upscale_factor, upscale_mode]
        )
        vintage_btn.click(
            fn=lambda: _preset_wrapper("vintage"),
            outputs=[bg_blur, bg_mode, bg_strength, bg_model, grain_intensity, grain, upscale, upscale_factor, upscale_mode]
        )
        minimal_btn.click(
            fn=lambda: _preset_wrapper("minimal"),
            outputs=[bg_blur, bg_mode, bg_strength, bg_model, grain_intensity, grain, upscale, upscale_factor, upscale_mode]
        )

        # ── Reset ──
        reset_defaults_btn = gr.Button("🔄 Reset to Defaults", variant="secondary")
        reset_defaults_btn.click(
            fn=lambda: (1, "blur", 5, "u2net", 0.5, True, True, "2x", "interp"),
            outputs=[bg_blur, bg_mode, bg_strength, bg_model, grain_intensity, grain, upscale, upscale_factor, upscale_mode]
        )

    return demo


if __name__ == "__main__":
    demo = create_enhanced_ui()
    demo.launch()
