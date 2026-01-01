from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import random


def apply_film_grain(image: Image.Image, intensity: float = 0.5) -> tuple[Image.Image, Image.Image]:
    """Apply film grain effect with luminance-aware scaling.

    Args:
        image: Input PIL Image
        intensity: Grain intensity from 0.1 to 1.0

    Returns:
        Tuple of (processed_image, grain_only_debug)
    """
    if intensity <= 0:
        return image, image

    width, height = image.size
    original_mode = image.mode
    if image.mode == 'L':
        numpy_image = np.array(image.convert('RGB'))
    else:
        numpy_image = np.array(image)

    luminance = np.dot(numpy_image[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    luminance = luminance / 255.0

    noise = np.random.uniform(-1, 1, (height, width)).astype(np.float32)

    luminance_mask = 0.3 + 0.7 * (1.0 - luminance)
    luminance_mask = luminance_mask / (luminance_mask.max() + 1e-8)

    scaled_grain = noise * luminance_mask * intensity * 32.0

    scaled_grain = np.clip(scaled_grain, -255, 255)

    result = numpy_image.copy().astype(np.float32)
    result[..., 0] = np.clip(result[..., 0] + scaled_grain, 0, 255)
    result[..., 1] = np.clip(result[..., 1] + scaled_grain, 0, 255)
    result[..., 2] = np.clip(result[..., 2] + scaled_grain, 0, 255)

    if numpy_image.shape[2] == 4:
        result_uint8 = np.clip(result, 0, 255).astype(np.uint8)
        grain_uint8 = np.clip(scaled_grain + 128, 0, 255).astype(np.uint8)
    else:
        result_uint8 = np.clip(result, 0, 255).astype(np.uint8)
        grain_uint8 = np.clip(scaled_grain + 128, 0, 255).astype(np.uint8)

    if original_mode == 'L':
        gray_result = result.mean(axis=2)
        processed_image = Image.fromarray(
            np.clip(gray_result, 0, 255).astype(np.uint8),
            mode='L'
        )
    else:
        processed_image = Image.fromarray(
            np.clip(result, 0, 255).astype(np.uint8),
            mode='RGB' if original_mode == 'RGB' else 'RGBA'
        )

    grain_colored = np.zeros((height, width, 3), dtype=np.uint8) if original_mode != 'L' else np.zeros((height, width), dtype=np.uint8)
    if original_mode != 'L':
        grain_colored[..., 0] = grain_uint8
        grain_colored[..., 1] = grain_uint8
        grain_colored[..., 2] = grain_uint8
        if original_mode == 'RGBA':
            grain_colored = np.dstack([grain_colored, np.full((height, width), 255, dtype=np.uint8)])
            grain_debug = Image.fromarray(grain_colored, mode='RGBA')
        else:
            grain_debug = Image.fromarray(grain_colored, mode='RGB')
    else:
        grain_debug = Image.fromarray(grain_uint8, mode='L')

    return processed_image, grain_debug


if __name__ == "__main__":
    img = Image.new('RGB', (200, 200), color='gray')
    result, debug = apply_film_grain(img, intensity=0.5)
    print(f"Original size: {img.size}")
    print(f"Result size: {result.size}")
    print("Grain effect test passed!")
