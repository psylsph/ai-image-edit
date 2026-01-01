"""Background removal and blur processing using rembg."""

from PIL import Image, ImageFilter, ImageOps

from models.model_downloader import get_device

_DEVICE = get_device()
_SESSION = None


def _get_device():
    return _DEVICE


def _load_session():
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    try:
        from rembg import new_session
        _SESSION = new_session("u2net")
        return _SESSION
    except ImportError as exc:
        raise ImportError(
            "rembg package not found. "
            "Please install with: pip install rembg"
        ) from exc


def process_background(
        image: Image.Image,
        blur_strength: int = 15
) -> tuple[Image.Image, Image.Image]:
    """Apply background blur using rembg for matting.

    Args:
        image: Input PIL Image
        blur_strength: Gaussian blur strength for background (1-50)

    Returns:
        Tuple of (processed_image, mask_debug)
    """
    if blur_strength <= 0:
        return image, image

    original_size = image.size
    rgb_image = image.convert('RGB')

    try:
        session = _load_session()
    except ImportError as exc:
        print(f"Warning: rembg not available ({exc}). Returning original image.")
        return image, image

    try:
        from rembg import remove
        mask_image = remove(rgb_image, session=session)
        mask_pil = mask_image.convert('L').resize(original_size, Image.LANCZOS)
    except Exception as exc:
        print(f"Warning: Background removal failed ({exc}). Returning original image.")
        return image, image

    try:
        mask_colored = ImageOps.autocontrast(mask_pil.convert('L')).convert('RGB')
    except ImportError:
        mask_colored = Image.merge('RGB', [mask_pil, mask_pil, mask_pil])

    original_rgba = image.convert('RGBA')
    blurred = original_rgba.filter(ImageFilter.GaussianBlur(radius=blur_strength))

    binary_mask = mask_pil.point(lambda x: 255 if x > 10 else 0)
    result_rgba = Image.composite(original_rgba, blurred, binary_mask)

    foreground_pil = result_rgba.convert('RGB' if image.mode == 'RGB' else image.mode)

    return foreground_pil, mask_colored


if __name__ == "__main__":
    print("Testing background processing...")
    try:
        img = Image.new('RGB', (256, 256), color='blue')
        result, mask = process_background(img, blur_strength=10)
        print(f"Input size: {img.size}")
        print(f"Output size: {result.size}")
        print(f"Mask size: {mask.size}")
        print("Background processing test passed!")
    except Exception as exc:
        print(f"Test failed (expected if rembg not installed): {exc}")
