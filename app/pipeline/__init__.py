from typing import Optional

from PIL import Image

from .background import process_background
from .grain import apply_film_grain
from .upscale import upscale_image


def preprocess_image(image: Image.Image, max_dimension: int = 1536) -> Image.Image:
    """Resize image maintaining aspect ratio, max dimension <= 1536px.

    Args:
        image: Input PIL Image
        max_dimension: Maximum dimension (default 1536)

    Returns:
        Resized PIL Image
    """
    width, height = image.size
    if max(width, height) <= max_dimension:
        return image

    if width > height:
        new_width = max_dimension
        new_height = int(height * max_dimension / width)
    else:
        new_height = max_dimension
        new_width = int(width * max_dimension / height)

    return image.resize((new_width, new_height), Image.LANCZOS)


def process_pipeline(
    image: Image.Image,
    enable_background_blur: bool = True,
    blur_strength: int = 15,
    enable_grain: bool = True,
    grain_intensity: float = 0.5,
    enable_upscale: bool = True,
    upscale_factor: int = 2,
    debug: bool = True
) -> dict:
    """Main pipeline orchestrator.

    Processing order:
    1. Pre-process (resize to <=1536px)
    2. Background Blur (optional)
    3. Film Grain (optional)
    4. Upscaling (optional)

    Args:
        image: Input PIL Image
        enable_background_blur: Enable background blur stage
        blur_strength: Blur strength (1-50)
        enable_grain: Enable film grain stage
        grain_intensity: Grain intensity (0.1-1.0)
        enable_upscale: Enable upscaling stage
        upscale_factor: Scale factor (2 or 4)
        debug: Include intermediate outputs

    Returns:
        dict with keys:
            'final': Final processed image
            'preprocessed': After pre-processing (if debug)
            'bg_result': After background blur (if enabled, if debug)
            'bg_mask': Background mask (if enabled, if debug)
            'grain_result': After film grain (if enabled, if debug)
            'grain_only': Grain only (if enabled, if debug)
            'upscale_result': After upscaling (if enabled, if debug)
    """
    result = {}

    preprocessed = preprocess_image(image, max_dimension=1536)
    if debug:
        result['preprocessed'] = preprocessed

    current_image = preprocessed
    current_debug_bg = None
    current_debug_grain = None
    current_debug_upscale = None
    current_bg_mask = None
    current_grain_only = None

    if enable_background_blur:
        current_image, current_debug_bg = process_background(current_image, blur_strength)
        if debug:
            result['bg_result'] = current_image
            result['bg_mask'] = current_debug_bg

    if enable_grain:
        current_image, current_debug_grain = apply_film_grain(current_image, grain_intensity)
        if debug:
            result['grain_result'] = current_image
            result['grain_only'] = current_debug_grain

    if enable_upscale:
        current_image, current_debug_upscale = upscale_image(current_image, upscale_factor)
        if debug:
            result['upscale_result'] = current_image

    result['final'] = current_image

    return result


def process_image_simple(
    image: Image.Image,
    enable_background_blur: bool = True,
    blur_strength: int = 15,
    enable_grain: bool = True,
    grain_intensity: float = 0.5,
    enable_upscale: bool = True,
    upscale_factor: int = 2
) -> Image.Image:
    """Simpler interface returning only the final image.

    Args:
        image: Input PIL Image
        enable_background_blur: Enable background blur stage
        blur_strength: Blur strength (1-50)
        enable_grain: Enable film grain stage
        grain_intensity: Grain intensity (0.1-1.0)
        enable_upscale: Enable upscaling stage
        upscale_factor: Scale factor (2 or 4)

    Returns:
        Final processed PIL Image
    """
    pipeline_result = process_pipeline(
        image,
        enable_background_blur=enable_background_blur,
        blur_strength=blur_strength,
        enable_grain=enable_grain,
        grain_intensity=grain_intensity,
        enable_upscale=enable_upscale,
        upscale_factor=upscale_factor,
        debug=False
    )
    return pipeline_result['final']


if __name__ == "__main__":
    print("Testing pipeline...")
    img = Image.new('RGB', (800, 600), color='white')
    result = process_pipeline(img)
    print(f"Input size: {img.size}")
    print(f"Final size: {result['final'].size}")
    print("Keys in result:", list(result.keys()))
    print("Pipeline test passed!")
