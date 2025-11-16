import time
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)

def timer(func: Callable) -> Callable:
    """
    Decorator to measure function execution time and log it
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            logger.debug(f"Function {func.__name__} executed in {execution_time:.2f}ms")
    
    return wrapper

class Timer:
    """
    Context manager for timing code blocks
    """
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        execution_time = (self.end_time - self.start_time) * 1000
        logger.info(f"{self.name} completed in {execution_time:.2f}ms")
    
    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

def measure_latency(func: Callable) -> Callable:
    """
    Higher-order function to measure latency and include it in response
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            
            # If result is a dictionary, add latency
            if isinstance(result, dict):
                result['latency_ms'] = (time.time() - start_time) * 1000
            
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    
    return wrapper