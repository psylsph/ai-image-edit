"""Additional image filters and effects."""

from typing import Tuple
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
import numpy as np
import logging

logger = logging.getLogger(__name__)


def adjust_brightness(
    image: Image.Image,
    factor: float = 0.0
) -> Image.Image:
    """Adjust image brightness.
    
    Args:
        image: Input PIL Image
        factor: Brightness adjustment (-1.0 to 1.0)
               Negative = darker, Positive = brighter
    
    Returns:
        Adjusted image
    """
    if factor == 0:
        return image
    
    enhancer = ImageEnhance.Brightness(image)
    # Convert factor from [-1, 1] to enhancement factor
    enhancement = 1.0 + factor
    return enhancer.enhance(enhancement)


def adjust_contrast(
    image: Image.Image,
    factor: float = 0.0
) -> Image.Image:
    """Adjust image contrast.
    
    Args:
        image: Input PIL Image
        factor: Contrast adjustment (-1.0 to 1.0)
               Negative = less contrast, Positive = more contrast
    
    Returns:
        Adjusted image
    """
    if factor == 0:
        return image
    
    enhancer = ImageEnhance.Contrast(image)
    enhancement = 1.0 + factor
    return enhancer.enhance(enhancement)


def adjust_saturation(
    image: Image.Image,
    factor: float = 0.0
) -> Image.Image:
    """Adjust image color saturation.
    
    Args:
        image: Input PIL Image
        factor: Saturation adjustment (-1.0 to 1.0)
               Negative = desaturated, Positive = more saturated
    
    Returns:
        Adjusted image
    """
    if factor == 0:
        return image
    
    enhancer = ImageEnhance.Color(image)
    enhancement = 1.0 + factor
    return enhancer.enhance(enhancement)


def adjust_sharpness(
    image: Image.Image,
    amount: float = 0.0
) -> Image.Image:
    """Adjust image sharpness.
    
    Args:
        image: Input PIL Image
        amount: Sharpness adjustment (-1.0 to 1.0)
                Negative = blurrier, Positive = sharper
    
    Returns:
        Adjusted image
    """
    if amount == 0:
        return image
    
    enhancer = ImageEnhance.Sharpness(image)
    enhancement = 1.0 + amount
    return enhancer.enhance(enhancement)


def add_vignette(
    image: Image.Image,
    intensity: float = 0.5,
    radius: float = 0.5
) -> Image.Image:
    """Add vignette effect (darkened corners) using numpy.

    Args:
        image: Input PIL Image
        intensity: Vignette strength (0.0 to 1.0)
        radius: Vignette radius (0.0 to 1.0)

    Returns:
        Image with vignette effect
    """
    if intensity <= 0:
        return image

    import numpy as np

    width, height = image.size
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Create normalized radial gradient using numpy (fast, no pixel loops)
    y_coords, x_coords = np.ogrid[:height, :width]
    center_y, center_x = height / 2.0, width / 2.0
    max_radius = max(width, height) * 0.5 * radius
    
    # Distance from center, normalized
    dist = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    dist = np.clip(dist / max_radius, 0, 1)
    
    # Apply intensity curve: outer pixels become fully dark, inner stay bright
    vignette_alpha = np.clip(1.0 - np.power(dist, max(0.1, 1.0 - intensity)), 0, 1)
    vignette_alpha = (vignette_alpha * 255).astype(np.uint8)
    
    # Build vignette layer: black with alpha mask
    vignette_layer = np.zeros((height, width, 4), dtype=np.uint8)
    vignette_layer[:, :, 3] = 255 - vignette_alpha  # alpha: 0=transparent, 255=opaque black
    
    vignette_pil = Image.fromarray(vignette_layer, mode='RGBA')
    
    # Composite vignette onto image
    result = Image.alpha_composite(image, vignette_pil)
    
    # Convert back to original mode
    if image.mode != 'RGBA':
        result = result.convert(image.mode)
    
    return result


def add_auto_enhance(image: Image.Image) -> Image.Image:
    """Apply automatic enhancements to image using histogram stretching.

    Args:
        image: Input PIL Image

    Returns:
        Enhanced image
    """
    import numpy as np

    # Convert to numpy
    arr = np.array(image.convert('RGB'))
    
    # Stretch histogram per channel (clip 1% low/high, then scale to 0-255)
    result = np.zeros_like(arr)
    for c in range(3):
        channel = arr[:, :, c]
        low, high = np.percentile(channel, [1, 99])
        if high > low:
            stretched = np.clip((channel.astype(float) - low) * 255.0 / (high - low), 0, 255)
        else:
            stretched = channel.astype(float)
        result[:, :, c] = stretched
    
    result_img = Image.fromarray(result.astype(np.uint8), mode='RGB')
    
    # Convert back to original mode if not RGB
    if image.mode != 'RGB':
        result_img = result_img.convert(image.mode)
    
    # Apply slight sharpening
    enhancer = ImageEnhance.Sharpness(result_img)
    result_img = enhancer.enhance(1.2)
    
    return result_img


def apply_color_adjustments(
    image: Image.Image,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    sharpness: float = 0.0
) -> Image.Image:
    """Apply multiple color adjustments at once.
    
    Args:
        image: Input PIL Image
        brightness: Brightness adjustment (-1.0 to 1.0)
        contrast: Contrast adjustment (-1.0 to 1.0)
        saturation: Saturation adjustment (-1.0 to 1.0)
        sharpness: Sharpness adjustment (-1.0 to 1.0)
    
    Returns:
        Adjusted image
    """
    result = image
    
    if brightness != 0:
        result = adjust_brightness(result, brightness)
    
    if contrast != 0:
        result = adjust_contrast(result, contrast)
    
    if saturation != 0:
        result = adjust_saturation(result, saturation)
    
    if sharpness != 0:
        result = adjust_sharpness(result, sharpness)
    
    return result


def apply_sepia(image: Image.Image, intensity: float = 0.8) -> Image.Image:
    """Apply sepia tone effect.
    
    Args:
        image: Input PIL Image
        intensity: Sepia intensity (0.0 to 1.0)
    
    Returns:
        Sepia-toned image
    """
    if intensity <= 0:
        return image
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply sepia transformation
    pixels = np.array(image)
    
    # Sepia matrix
    r = pixels[:, :, 0]
    g = pixels[:, :, 1]
    b = pixels[:, :, 2]
    
    sepia_r = 0.393 * r + 0.769 * g + 0.189 * b
    sepia_g = 0.349 * r + 0.686 * g + 0.168 * b
    sepia_b = 0.272 * r + 0.534 * g + 0.131 * b
    
    # Blend with original based on intensity
    result_pixels = np.stack([
        r * (1 - intensity) + sepia_r * intensity,
        g * (1 - intensity) + sepia_g * intensity,
        b * (1 - intensity) + sepia_b * intensity
    ], axis=2)
    
    # Clip to valid range
    result_pixels = np.clip(result_pixels, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result_pixels)


def apply_grayscale(image: Image.Image, intensity: float = 1.0) -> Image.Image:
    """Apply grayscale effect.
    
    Args:
        image: Input PIL Image
        intensity: Grayscale intensity (0.0 to 1.0)
    
    Returns:
        Grayscale image
    """
    if intensity <= 0:
        return image
    
    if intensity >= 1.0:
        return image.convert('L').convert('RGB')
    
    # Convert to grayscale
    gray = image.convert('L')
    
    # Blend with original
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    gray_rgb = gray.convert('RGB')
    result = Image.blend(image, gray_rgb, intensity)
    
    return result


def apply_blur(image: Image.Image, radius: float = 5.0) -> Image.Image:
    """Apply Gaussian blur effect.
    
    Args:
        image: Input PIL Image
        radius: Blur radius (0.0 to 50.0)
    
    Returns:
        Blurred image
    """
    if radius <= 0:
        return image
    
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_unsharp_mask(
    image: Image.Image,
    radius: float = 2.0,
    percent: int = 150,
    threshold: int = 3
) -> Image.Image:
    """Apply unsharp mask sharpening.
    
    Args:
        image: Input PIL Image
        radius: Radius of blur for unsharp mask (0.0 to 10.0)
        percent: Strength of sharpening (100 to 300)
        threshold: Threshold for applying sharpening (0 to 20)
    
    Returns:
        Sharpened image
    """
    if radius <= 0 or percent <= 100:
        return image
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply unsharp mask
    return image.filter(ImageFilter.UnsharpMask(
        radius=radius,
        percent=percent,
        threshold=threshold
    ))


if __name__ == "__main__":
    # Test filters
    img = Image.new('RGB', (200, 200), color='red')
    
    print("Testing filters...")
    
    # Test color adjustments
    adjusted = apply_color_adjustments(
        img,
        brightness=0.2,
        contrast=0.3,
        saturation=-0.2
    )
    print(f"✓ Color adjustments: {adjusted.size}")
    
    # Test vignette
    vignetted = add_vignette(img, intensity=0.5, radius=0.7)
    print(f"✓ Vignette: {vignetted.size}")
    
    # Test sepia
    sepia = apply_sepia(img, intensity=0.8)
    print(f"✓ Sepia: {sepia.size}")
    
    print("All filters tested successfully!")
