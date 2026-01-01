"""Device and utility functions for AI Image Editor."""

import torch

CUDA_AVAILABLE = None


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    global CUDA_AVAILABLE
    if CUDA_AVAILABLE is None:
        CUDA_AVAILABLE = torch.cuda.is_available()
    return CUDA_AVAILABLE


def get_device() -> torch.device:
    """Get appropriate device (GPU if available, otherwise CPU)."""
    if is_cuda_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_string() -> str:
    """Return device string for display."""
    if is_cuda_available():
        return "CUDA"
    return "CPU"


def preload_models() -> dict:
    """Check model availability status."""
    status = {}
    status["device"] = get_device_string()
    status["rembg"] = "Auto-downloaded by rembg package"
    status["realesrgan"] = "Auto-downloaded by py-real-esrgan package"
    return status


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")
    print("Model status:")
    results = preload_models()
    for name, info in results.items():
        print(f"  {name}: {info}")
    print("Done!")
