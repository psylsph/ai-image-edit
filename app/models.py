"""Pydantic models for request validation."""

from pydantic import BaseModel, Field, validator
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
    
    @validator('upscale_factor')
    def validate_scale_factor(cls, v):
        """Validate that upscale factor is 2 or 4."""
        if v not in [1, 2, 4]:
            raise ValueError('Upscale factor must be 1, 2, or 4')
        return v
    
    @validator('blur_strength')
    def validate_blur_strength(cls, v):
        """Validate blur strength is reasonable."""
        if v > 50:
            raise ValueError('Blur strength cannot exceed 50')
        if v < 1:
            raise ValueError('Blur strength must be at least 1')
        return v
    
    @validator('grain_intensity')
    def validate_grain_intensity(cls, v):
        """Validate grain intensity is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Grain intensity must be between 0.0 and 1.0')
        return v
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "enable_background_blur": True,
                "blur_strength": 5,
                "enable_grain": True,
                "grain_intensity": 0.5,
                "enable_upscale": True,
                "upscale_factor": 2
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
    
    @validator('image_count')
    def validate_batch_size(cls, v):
        """Limit batch size to prevent abuse."""
        if v > 10:
            raise ValueError('Maximum batch size is 10 images')
        return v


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
            enable_grain=True,
            grain_intensity=0.3,
            enable_upscale=True,
            upscale_factor=2
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
            upscale_factor=4
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
            upscale_factor=1
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
            upscale_factor=1
        )
    )
}
