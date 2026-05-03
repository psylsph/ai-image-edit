"""Monitoring and metrics for AI Image Editor."""

import time
import logging
from functools import wraps
from typing import Callable

# Try to import prometheus_client, fallback if not available
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client not available, metrics disabled")

logger = logging.getLogger(__name__)


if PROMETHEUS_AVAILABLE:
    # Processing metrics
    processing_time = Histogram(
        'image_processing_seconds',
        'Time spent processing images',
        ['operation', 'status']
    )
    
    processing_requests = Counter(
        'image_processing_requests_total',
        'Total number of image processing requests',
        ['operation', 'status']
    )
    
    # Model loading metrics
    model_load_time = Histogram(
        'model_load_seconds',
        'Time spent loading models',
        ['model_type']
    )
    
    # System metrics
    active_processing = Gauge(
        'active_processing_jobs',
        'Number of currently active processing jobs'
    )
    
    cache_hits = Counter(
        'model_cache_hits_total',
        'Total number of model cache hits',
        ['model_type']
    )
    
    cache_misses = Counter(
        'model_cache_misses_total',
        'Total number of model cache misses',
        ['model_type']
    )
    
    # Error metrics
    errors = Counter(
        'errors_total',
        'Total number of errors',
        ['error_type', 'operation']
    )
    
    # Request size metrics
    request_size = Histogram(
        'request_size_bytes',
        'Size of incoming requests',
        ['endpoint']
    )
    
    response_size = Histogram(
        'response_size_bytes',
        'Size of outgoing responses',
        ['endpoint']
    )
    
    # Concurrent requests
    concurrent_requests = Gauge(
        'concurrent_requests',
        'Number of concurrent requests being processed'
    )


def track_processing_time(operation: str):
    """Decorator to track processing time.
    
    Args:
        operation: Name of the operation being tracked
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not PROMETHEUS_AVAILABLE:
                return func(*args, **kwargs)
            
            start_time = time.time()
            status = "success"
            
            try:
                active_processing.inc()
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                errors.labels(
                    error_type=type(e).__name__,
                    operation=operation
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                processing_time.labels(
                    operation=operation,
                    status=status
                ).observe(duration)
                
                processing_requests.labels(
                    operation=operation,
                    status=status
                ).inc()
                
                active_processing.dec()
        
        return wrapper
    return decorator


def track_model_load(model_type: str):
    """Decorator to track model loading time.
    
    Args:
        model_type: Type of model being loaded
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not PROMETHEUS_AVAILABLE:
                return func(*args, **kwargs)
            
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            model_load_time.labels(model_type=model_type).observe(duration)
            
            return result
        return wrapper
    return decorator


def record_cache_hit(model_type: str):
    """Record a model cache hit."""
    if PROMETHEUS_AVAILABLE:
        cache_hits.labels(model_type=model_type).inc()
        logger.debug(f"Cache hit recorded for {model_type}")


def record_cache_miss(model_type: str):
    """Record a model cache miss."""
    if PROMETHEUS_AVAILABLE:
        cache_misses.labels(model_type=model_type).inc()
        logger.debug(f"Cache miss recorded for {model_type}")


def start_metrics_server(port: int = 9090):
    """Start the Prometheus metrics server.
    
    Args:
        port: Port to run metrics server on
    """
    if PROMETHEUS_AVAILABLE:
        try:
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    else:
        logger.warning("Prometheus not available, metrics server not started")


def get_metrics_summary() -> dict:
    """Get summary of current metrics.
    
    Returns:
        Dictionary with metrics summary
    """
    if not PROMETHEUS_AVAILABLE:
        return {"status": "disabled"}
    
    # This would require accessing the internal metric collectors
    # For now, return status
    return {
        "status": "enabled",
        "endpoints": {
            "metrics": f"http://localhost:{9090}/metrics"
        }
    }


# Performance monitoring utilities
class PerformanceMonitor:
    """Context manager for monitoring performance."""
    
    def __init__(self, operation: str):
        """Initialize performance monitor.
        
        Args:
            operation: Name of the operation
        """
        self.operation = operation
        self.start_time = None
        
    def __enter__(self):
        """Enter context and start timing."""
        self.start_time = time.time()
        if PROMETHEUS_AVAILABLE:
            active_processing.inc()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and record metrics."""
        duration = time.time() - self.start_time
        
        if PROMETHEUS_AVAILABLE:
            active_processing.dec()
            
            status = "error" if exc_type else "success"
            processing_time.labels(
                operation=self.operation,
                status=status
            ).observe(duration)
            
            processing_requests.labels(
                operation=self.operation,
                status=status
            ).inc()
        
        if exc_type:
            logger.error(f"Operation {self.operation} failed after {duration:.2f}s")
        else:
            logger.info(f"Operation {self.operation} completed in {duration:.2f}s")


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics system...")
    
    @track_processing_time("test_operation")
    def test_function():
        time.sleep(0.1)
        return "success"
    
    # Start metrics server
    start_metrics_server(port=9090)
    
    # Run test function
    result = test_function()
    print(f"Test function result: {result}")
    
    # Get summary
    summary = get_metrics_summary()
    print(f"Metrics summary: {summary}")
    
    print("Metrics test complete!")
