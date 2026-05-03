"""Batch processing utilities for multiple images."""

import asyncio
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
import logging

from PIL import Image
from . import process_pipeline
from app.config import settings

logger = logging.getLogger(__name__)


async def process_single_image(
    image: Image.Image,
    enable_background_blur: bool = True,
    blur_strength: int = 5,
    enable_grain: bool = True,
    grain_intensity: float = 0.5,
    enable_upscale: bool = True,
    upscale_factor: int = 2
) -> Tuple[Image.Image, dict]:
    """Process a single image and return result with metadata.
    
    Args:
        image: PIL Image to process
        enable_background_blur: Enable background blur
        blur_strength: Blur strength (1-50)
        enable_grain: Enable film grain
        grain_intensity: Grain intensity (0.1-1.0)
        enable_upscale: Enable upscaling
        upscale_factor: Scale factor (2 or 4)
    
    Returns:
        Tuple of (processed_image, metadata_dict)
    """
    try:
        result = process_pipeline(
            image,
            enable_background_blur=enable_background_blur,
            blur_strength=blur_strength,
            enable_grain=enable_grain,
            grain_intensity=grain_intensity,
            enable_upscale=enable_upscale,
            upscale_factor=upscale_factor,
            debug=False
        )
        
        metadata = {
            "success": True,
            "original_size": image.size,
            "final_size": result['final'].size,
            "error": None
        }
        
        return result['final'], metadata
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return image, {
            "success": False,
            "original_size": image.size,
            "final_size": image.size,
            "error": str(e)
        }


async def process_batch(
    images: List[Image.Image],
    enable_background_blur: bool = True,
    blur_strength: int = 5,
    enable_grain: bool = True,
    grain_intensity: float = 0.5,
    enable_upscale: bool = True,
    upscale_factor: int = 2,
    max_workers: int = 3
) -> List[Tuple[Image.Image, dict]]:
    """Process multiple images in parallel.
    
    Args:
        images: List of PIL Images
        enable_background_blur: Enable background blur
        blur_strength: Blur strength (1-50)
        enable_grain: Enable film grain
        grain_intensity: Grain intensity (0.1-1.0)
        enable_upscale: Enable upscaling
        upscale_factor: Scale factor (2 or 4)
        max_workers: Maximum number of parallel workers
    
    Returns:
        List of tuples (processed_image, metadata)
    """
    if len(images) > settings.MAX_BATCH_SIZE:
        raise ValueError(f"Batch size exceeds maximum of {settings.MAX_BATCH_SIZE}")
    
    logger.info(f"Processing batch of {len(images)} images with {max_workers} workers")
    
    # Process images in thread pool for CPU-bound operations
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(
                executor,
                process_single_image,
                image,
                enable_background_blur,
                blur_strength,
                enable_grain,
                grain_intensity,
                enable_upscale,
                upscale_factor
            )
            for image in images
        ]
        
        results = await asyncio.gather(*tasks)
    
    success_count = sum(1 for _, meta in results if meta.get("success"))
    logger.info(f"Batch processing complete: {success_count}/{len(images)} successful")
    
    return results


def save_batch_results(
    results: List[Tuple[Image.Image, dict]],
    output_dir: Path,
    prefix: str = "processed"
) -> List[Path]:
    """Save batch processing results to disk.
    
    Args:
        results: List of tuples (image, metadata) from process_batch
        output_dir: Directory to save results
        prefix: Filename prefix
    
    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for idx, (image, metadata) in enumerate(results):
        if metadata.get("success"):
            filename = f"{prefix}_{idx+1:03d}.png"
            output_path = output_dir / filename
            image.save(output_path, "PNG")
            saved_paths.append(output_path)
            logger.info(f"Saved: {output_path}")
        else:
            logger.warning(f"Skipping image {idx+1} due to processing error")
    
    return saved_paths


async def process_directory(
    input_dir: Path,
    output_dir: Path,
    enable_background_blur: bool = True,
    blur_strength: int = 5,
    enable_grain: bool = True,
    grain_intensity: float = 0.5,
    enable_upscale: bool = True,
    upscale_factor: int = 2,
    file_patterns: List[str] = None
) -> dict:
    """Process all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        enable_background_blur: Enable background blur
        blur_strength: Blur strength (1-50)
        enable_grain: Enable film grain
        grain_intensity: Grain intensity (0.1-1.0)
        enable_upscale: Enable upscaling
        upscale_factor: Scale factor (2 or 4)
        file_patterns: List of file patterns to match (e.g., ["*.jpg", "*.png"])
    
    Returns:
        Dictionary with processing statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default image file patterns
    if file_patterns is None:
        file_patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
    
    # Collect all images
    image_files = []
    for pattern in file_patterns:
        image_files.extend(input_dir.glob(pattern))
    image_files.extend(input_dir.glob(pattern.upper()))
    
    logger.info(f"Found {len(image_files)} images in {input_dir}")
    
    if not image_files:
        return {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "output_dir": str(output_dir)
        }
    
    # Load images
    images = []
    for img_file in image_files:
        try:
            img = Image.open(img_file)
            images.append(img)
        except Exception as e:
            logger.error(f"Failed to load {img_file}: {e}")
    
    # Process batch
    results = await process_batch(
        images,
        enable_background_blur=enable_background_blur,
        blur_strength=blur_strength,
        enable_grain=enable_grain,
        grain_intensity=grain_intensity,
        enable_upscale=enable_upscale,
        upscale_factor=upscale_factor
    )
    
    # Save results
    saved_paths = save_batch_results(results, output_dir)
    
    # Statistics
    successful = sum(1 for _, meta in results if meta.get("success"))
    failed = len(results) - successful
    
    stats = {
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "output_dir": str(output_dir),
        "saved_files": [str(p) for p in saved_paths]
    }
    
    logger.info(f"Directory processing complete: {stats}")
    return stats


if __name__ == "__main__":
    import asyncio
    
    async def test_batch():
        """Test batch processing."""
        # Create test images
        images = [
            Image.new('RGB', (100, 100), color='red'),
            Image.new('RGB', (100, 100), color='green'),
            Image.new('RGB', (100, 100), color='blue')
        ]
        
        results = await process_batch(
            images,
            enable_upscale=False,  # Skip upscaling for quick test
            enable_grain=False
        )
        
        print(f"Processed {len(results)} images")
        for idx, (img, meta) in enumerate(results):
            print(f"  {idx+1}. Success: {meta['success']}")
    
    asyncio.run(test_batch())
