"""Model caching utilities for improved performance."""

from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# Global model cache
_model_cache = {}


def clear_model_cache():
    """Clear all cached models."""
    global _model_cache
    _model_cache.clear()
    logger.info("Model cache cleared")


def get_cached_model(model_type: str, model_name: str = None):
    """Get a model from cache or load it.
    
    Args:
        model_type: Type of model (birefnet, realesrgan, etc.)
        model_name: Specific model name (e.g., 'x2plus', 'x4plus')
    
    Returns:
        Cached or newly loaded model
    """
    cache_key = f"{model_type}_{model_name}" if model_name else model_type
    
    if cache_key in _model_cache:
        logger.debug(f"Using cached model: {cache_key}")
        return _model_cache[cache_key]
    
    logger.info(f"Loading model: {cache_key}")
    
    # Import model loaders here to avoid circular imports
    if model_type == "birefnet":
        from .background import _load_birefnet_model
        model = _load_birefnet_model()
    elif model_type == "realesrgan":
        from .upscale import _load_realesrgan_model
        model = _load_realesrgan_model(model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    _model_cache[cache_key] = model
    logger.info(f"Model cached: {cache_key}")
    
    return model


@lru_cache(maxsize=2)
def get_realesrgan_model(scale: int):
    """Get Real-ESRGAN model with LRU caching.
    
    Args:
        scale: Upscale factor (2 or 4)
    
    Returns:
        Cached Real-ESRGAN model
    """
    logger.info(f"Loading Real-ESRGAN x{scale} model")
    from .upscale import _load_realesrgan_model
    
    model_name = "x2plus" if scale == 2 else "x4plus"
    model = _load_realesrgan_model(model_name)
    
    logger.info(f"Real-ESRGAN x{scale} model loaded and cached")
    return model


@lru_cache(maxsize=1)
def get_birefnet_model():
    """Get BiRefNet model with LRU caching.
    
    Returns:
        Cached BiRefNet model
    """
    logger.info("Loading BiRefNet model")
    from .background import _load_birefnet_model
    
    model = _load_birefnet_model()
    logger.info("BiRefNet model loaded and cached")
    
    return model
