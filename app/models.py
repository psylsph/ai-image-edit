"""Pydantic models for request validation."""

from pydantic import BaseModel, Field
from typing import Optional, List


class ProcessRequest(BaseModel):
    """Image processing request with validation."""
    
    enable_background_blur: bool = True
    blur_strength: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Blur strength from 1-50"
    )
    bg_model: str = Field(
        default="u2net",
        description="Background removal model"
    )
    bg_mode: str = Field(
        default="blur",
        description="Background mode: 'blur' or 'remove'"
    )
    
    enable_grain: bool = True
    grain_intensity: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Grain intensity from 0.1-1.0"
    )
    
    enable_upscale: bool = True
    upscale_factor: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Upscale factor (2 or 4)"
    )
    upscale_mode: str = Field(
        default="esrgan",
        description="Upscale mode: 'esrgan' or 'interp'"
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "enable_background_blur": True,
                "blur_strength": 5,
                "bg_model": "u2net",
                "bg_mode": "blur",
                "enable_grain": True,
                "grain_intensity": 0.5,
                "enable_upscale": True,
                "upscale_factor": 2,
                "upscale_mode": "interp"
            }
        }

class BatchProcessRequest(BaseModel):
    """Batch processing request."""
    
    image_count: int = Field(
        ge=1,
        le=10,
        description="Number of images to process (max 10)"
    )
    settings: ProcessRequest


class ProcessingStatus(BaseModel):
    """Processing status response."""
    
    task_id: str
    status: str = Field(..., description="queued, processing, completed, failed")
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    stage: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
    result_url: Optional[str] = None


class PresetConfig(BaseModel):
    """Preset configuration for quick settings."""
    
    name: str = Field(..., description="Preset name")
    settings: ProcessRequest
    description: Optional[str] = None


# Preset configurations
PRESETS = {
    "portrait": PresetConfig(
        name="portrait",
        description="Optimized for portrait photos",
        settings=ProcessRequest(
            enable_background_blur=True,
            blur_strength=7,
            bg_model="u2net_human_seg",
            enable_grain=True,
            grain_intensity=0.3,
            enable_upscale=True,
            upscale_factor=2,
            upscale_mode="interp"
        )
    ),
    "landscape": PresetConfig(
        name="landscape",
        description="Optimized for landscape photos",
        settings=ProcessRequest(
            enable_background_blur=False,
            blur_strength=5,
            enable_grain=True,
            grain_intensity=0.5,
            enable_upscale=True,
            upscale_factor=4,
            upscale_mode="interp"
        )
    ),
    "vintage": PresetConfig(
        name="vintage",
        description="Vintage film look",
        settings=ProcessRequest(
            enable_background_blur=False,
            blur_strength=5,
            enable_grain=True,
            grain_intensity=0.8,
            enable_upscale=False,
            upscale_factor=1,
            upscale_mode="interp"
        )
    ),
    "minimal": PresetConfig(
        name="minimal",
        description="Minimal processing",
        settings=ProcessRequest(
            enable_background_blur=False,
            blur_strength=5,
            enable_grain=False,
            grain_intensity=0.1,
            enable_upscale=False,
            upscale_factor=1,
            upscale_mode="interp"
        )
    )
}
