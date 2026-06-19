"""Background removal and blur processing using rembg."""

from PIL import Image, ImageFilter, ImageOps

from models.model_downloader import get_device

_DEVICE = get_device()
_SESSIONS: dict[str, object] = {}

# Available models with human-readable labels
# Curated from rembg's 19 built-in models — niche ones (cloth seg, custom,
# SAM) excluded.  Quality/speed tradeoffs noted for UI hint.
BG_MODELS = {
    "u2net":              "Auto (u2net)",
    "u2netp":             "Fast (u2netp)",
    "u2net_human_seg":    "People (human seg)",
    "birefnet-portrait":  "People (BiRefNet portrait)",
    "birefnet-general":   "BiRefNet (best)",
    "isnet-general-use":  "IS-Net",
    "bria-rmbg":          "BRIA RMBG",
    "silueta":            "Silueta",
}


def _get_device():
    return _DEVICE


def _load_session(model_name: str = "u2net"):
    """Load and cache a rembg session for the given model.

    rembg downloads the ONNX model from the model zoo on first use,
    so the first call for a new model will be slow.
    """
    if model_name not in BG_MODELS:
        model_name = "u2net"

    if model_name in _SESSIONS:
        return _SESSIONS[model_name]

    try:
        from rembg import new_session
        session = new_session(model_name)
        _SESSIONS[model_name] = session
        return session
    except ImportError as exc:
        raise ImportError(
            "rembg package not found. "
            "Please install with: pip install rembg"
        ) from exc


def process_background(
        image: Image.Image,
        blur_strength: int = 15,
        model_name: str = "u2net",
        mode: str = "blur",
) -> tuple[Image.Image, Image.Image]:
    """Background processing — blur or remove — using rembg for matting.

    Args:
        image: Input PIL Image
        blur_strength: Gaussian blur strength for background (1-50, blur mode only)
        model_name: rembg model to use (see BG_MODELS for options)
        mode: "blur" = blur background keeping foreground sharp,
              "remove" = cut out foreground with transparent background

    Returns:
        Tuple of (processed_image, mask_debug)
        In "remove" mode the processed image is RGBA (transparent bg).
    """
    original_size = image.size
    rgb_image = image.convert('RGB')

    try:
        session = _load_session(model_name)
    except ImportError as exc:
        print(f"Warning: rembg not available ({exc}). Returning original image.")
        return image, image

    try:
        from rembg import remove
        cutout = remove(rgb_image, session=session)
        cutout = cutout.convert('RGBA').resize(original_size, Image.LANCZOS)
    except Exception as exc:
        print(f"Warning: Background removal failed ({exc}). Returning original image.")
        return image, image

    # ── Debug mask (always the same regardless of mode) ──
    alpha = cutout.split()[-1]
    try:
        mask_colored = ImageOps.autocontrast(alpha.convert('L')).convert('RGB')
    except Exception:
        mask_colored = Image.merge('RGB', [alpha, alpha, alpha])

    # ── Remove mode: just return the cutout ──
    if mode == "remove":
        # Feather edges slightly for a cleaner cutout
        feathered = alpha.filter(ImageFilter.GaussianBlur(radius=1))
        original_rgba = image.convert('RGBA')
        # Apply feathered alpha to the original (preserves colour fidelity)
        result = original_rgba.copy()
        result.putalpha(feathered)
        return result, mask_colored

    # ── Blur mode ──
    if blur_strength <= 0:
        return image, mask_colored

    original_rgba = image.convert('RGBA')

    # Binary mask for sharp foreground / blurred background split
    binary_mask = alpha.point(lambda x: 255 if x > 128 else 0)
    soft_mask = binary_mask.filter(ImageFilter.GaussianBlur(radius=3))

    blurred_image = original_rgba.filter(ImageFilter.GaussianBlur(radius=blur_strength))
    result_rgba = Image.composite(cutout, blurred_image, soft_mask)
    result = result_rgba.convert('RGB' if image.mode == 'RGB' else image.mode)

    return result, mask_colored


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
