"""Custom exceptions for AI Image Editor."""


class ImageProcessingError(Exception):
    """Base exception for image processing errors."""
    
    def __init__(self, message: str, suggestion: str = None):
        """Initialize error with message and optional suggestion."""
        self.message = message
        self.suggestion = suggestion
        super().__init__(self.message)


class ValidationError(ImageProcessingError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: str = None):
        """Initialize validation error."""
        self.field = field
        suggestion = f"Check the '{field}' parameter" if field else "Check your input parameters"
        super().__init__(message, suggestion)


class ModelLoadError(ImageProcessingError):
    """Raised when a model fails to load."""
    
    def __init__(self, model_name: str, reason: str = None):
        """Initialize model load error."""
        message = f"Failed to load model: {model_name}"
        if reason:
            message += f" ({reason})"
        suggestion = "Check model cache and internet connection"
        super().__init__(message, suggestion)


class ProcessingTimeoutError(ImageProcessingError):
    """Raised when processing takes too long."""
    
    def __init__(self, operation: str, timeout_seconds: int):
        """Initialize timeout error."""
        message = f"{operation} timed out after {timeout_seconds} seconds"
        suggestion = "Try a smaller image or fewer processing steps"
        super().__init__(message, suggestion)


class FileSizeError(ImageProcessingError):
    """Raised when uploaded file is too large."""
    
    def __init__(self, file_size_mb: float, max_size_mb: float):
        """Initialize file size error."""
        message = f"File size ({file_size_mb:.1f}MB) exceeds maximum ({max_size_mb:.1f}MB)"
        suggestion = f"Compress your image to under {max_size_mb:.0f}MB"
        super().__init__(message, suggestion)


class ImageFormatError(ImageProcessingError):
    """Raised when image format is not supported."""
    
    def __init__(self, format: str, supported_formats: list[str]):
        """Initialize format error."""
        message = f"Unsupported image format: {format}"
        suggestion = f"Use one of: {', '.join(supported_formats)}"
        super().__init__(message, suggestion)


class RateLimitError(ImageProcessingError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, limit: int, period: int):
        """Initialize rate limit error."""
        message = f"Rate limit exceeded: {limit} requests per {period} seconds"
        suggestion = "Wait a moment before trying again"
        super().__init__(message, suggestion)


class TaskNotFoundError(ImageProcessingError):
    """Raised when a task ID is not found."""
    
    def __init__(self, task_id: str):
        """Initialize task not found error."""
        message = f"Task not found: {task_id}"
        suggestion = "The task may have expired or invalid ID"
        super().__init__(message, suggestion)
