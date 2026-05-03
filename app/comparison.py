"""Before/After comparison component for Gradio."""

import gradio as gr
from typing import Tuple, Optional
from PIL import Image


def create_comparison_view(
    before: Image.Image,
    after: Image.Image,
    label: str = "Before vs After"
) -> gr.components.Component:
    """Create a before/after comparison slider.
    
    Args:
        before: Original image
        after: Processed image
        label: Label for the comparison
    
    Returns:
        Gradio component for comparison
    """
    # For now, use Gradio's built-in Image component
    # In a future enhancement, we could implement a proper slider
    return gr.Image(
        value=after,
        label=label,
        type="pil"
    )


def create_side_by_side_comparison(
    before: Image.Image,
    after: Image.Image
) -> Tuple[Image.Image, Image.Image]:
    """Create side-by-side comparison images.
    
    Args:
        before: Original image
        after: Processed image
    
    Returns:
        Tuple of (before_image, after_image)
    """
    return before, after


def create_comparison_overlay(
    before: Image.Image,
    after: Image.Image,
    slider_position: float = 0.5
) -> Image.Image:
    """Create an overlay comparison with slider position.
    
    Args:
        before: Original image
        after: Processed image
        slider_position: Position from 0-1 (0 = all before, 1 = all after)
    
    Returns:
        Composite image showing comparison
    """
    from PIL import ImageDraw, ImageFont
    
    # Ensure images are the same size
    if before.size != after.size:
        after = after.resize(before.size)
    
    # Create composite
    width, height = before.size
    composite = Image.new('RGB', (width * 2, height))
    
    # Place images side by side
    composite.paste(before, (0, 0))
    composite.paste(after, (width, 0))
    
    # Add divider line
    draw = ImageDraw.Draw(composite)
    draw.line([(width, 0), (width, height)], fill=(255, 255, 255), width=3)
    
    # Add labels
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "BEFORE", fill=(255, 255, 255), font=font)
    draw.text((width + 10, 10), "AFTER", fill=(255, 255, 255), font=font)
    
    return composite


def create_diff_overlay(
    before: Image.Image,
    after: Image.Image,
    enhance: float = 5.0
) -> Image.Image:
    """Create a difference overlay to highlight changes.
    
    Args:
        before: Original image
        after: Processed image
        enhance: Enhancement factor for differences
    
    Returns:
        Difference image highlighting changes
    """
    from PIL import ImageChops, ImageEnhance
    
    # Ensure same size
    if before.size != after.size:
        after = after.resize(before.size)
    
    # Calculate difference
    diff = ImageChops.difference(before.convert('RGB'), after.convert('RGB'))
    
    # Enhance differences for visibility
    enhancer = ImageEnhance.Contrast(diff)
    diff = enhancer.enhance(enhance)
    
    return diff
