"""Decorators for the Alpaca Trading Pipeline."""

import time
import functools
from typing import Callable, Any, Optional, Type, Union
from threading import Lock
from collections import defaultdict, deque
import asyncio
from datetime import datetime, timedelta

from .logging import get_logger

logger = get_logger(__name__)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_failure: Optional[Callable] = None
):
    """Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch and retry
        on_failure: Optional callback function on final failure
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        # Final attempt failed
                        logger.error(
                            "Function failed after all retry attempts",
                            function=func.__name__,
                            attempts=max_attempts,
                            error=str(e)
                        )
                        
                        if on_failure:
                            on_failure(e, attempt + 1)
                        
                        raise e
                    
                    logger.warning(
                        "Function failed, retrying",
                        function=func.__name__,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        delay=current_delay,
                        error=str(e)
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        logger.error(
                            "Async function failed after all retry attempts",
                            function=func.__name__,
                            attempts=max_attempts,
                            error=str(e)
                        )
                        
                        if on_failure:
                            if asyncio.iscoroutinefunction(on_failure):
                                await on_failure(e, attempt + 1)
                            else:
                                on_failure(e, attempt + 1)
                        
                        raise e
                    
                    logger.warning(
                        "Async function failed, retrying",
                        function=func.__name__,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        delay=current_delay,
                        error=str(e)
                    )
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            if last_exception:
                raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


class RateLimiter:
    """Thread-safe rate limiter using token bucket algorithm."""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self.lock = Lock()
    
    def is_allowed(self) -> bool:
        """Check if a call is allowed under the rate limit."""
        with self.lock:
            now = time.time()
            
            # Remove old calls outside the time window
            while self.calls and self.calls[0] <= now - self.time_window:
                self.calls.popleft()
            
            # Check if we can make a new call
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            
            return False
    
    def wait_time(self) -> float:
        """Calculate how long to wait before next call is allowed."""
        with self.lock:
            if len(self.calls) < self.max_calls:
                return 0.0
            
            # Time until the oldest call expires
            oldest_call = self.calls[0]
            return max(0.0, oldest_call + self.time_window - time.time())


# Global rate limiters for different API endpoints
_rate_limiters = defaultdict(lambda: RateLimiter(200, 60))  # Default: 200 calls per minute


def rate_limit(
    max_calls: int = 200,
    time_window: float = 60.0,
    key: Optional[str] = None,
    wait: bool = True
):
    """Rate limiting decorator.
    
    Args:
        max_calls: Maximum number of calls allowed
        time_window: Time window in seconds
        key: Optional key for separate rate limiters
        wait: Whether to wait when rate limit is exceeded
    """
    def decorator(func: Callable) -> Callable:
        limiter_key = key or f"{func.__module__}.{func.__name__}"
        limiter = RateLimiter(max_calls, time_window)
        _rate_limiters[limiter_key] = limiter
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not limiter.is_allowed():
                if wait:
                    wait_time = limiter.wait_time()
                    if wait_time > 0:
                        logger.info(
                            "Rate limit exceeded, waiting",
                            function=func.__name__,
                            wait_time=wait_time
                        )
                        time.sleep(wait_time)
                        # Try again after waiting
                        if not limiter.is_allowed():
                            raise RuntimeError(f"Rate limit exceeded for {func.__name__}")
                else:
                    raise RuntimeError(f"Rate limit exceeded for {func.__name__}")
            
            return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            if not limiter.is_allowed():
                if wait:
                    wait_time = limiter.wait_time()
                    if wait_time > 0:
                        logger.info(
                            "Rate limit exceeded, waiting",
                            function=func.__name__,
                            wait_time=wait_time
                        )
                        await asyncio.sleep(wait_time)
                        # Try again after waiting
                        if not limiter.is_allowed():
                            raise RuntimeError(f"Rate limit exceeded for {func.__name__}")
                else:
                    raise RuntimeError(f"Rate limit exceeded for {func.__name__}")
            
            return await func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def timing(log_level: str = "DEBUG"):
    """Timing decorator to measure function execution time.
    
    Args:
        log_level: Log level for timing information
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                getattr(logger, log_level.lower())(
                    "Function execution completed",
                    function=func.__name__,
                    execution_time=f"{execution_time:.4f}s"
                )
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    "Function execution failed",
                    function=func.__name__,
                    execution_time=f"{execution_time:.4f}s",
                    error=str(e)
                )
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                getattr(logger, log_level.lower())(
                    "Async function execution completed",
                    function=func.__name__,
                    execution_time=f"{execution_time:.4f}s"
                )
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    "Async function execution failed",
                    function=func.__name__,
                    execution_time=f"{execution_time:.4f}s",
                    error=str(e)
                )
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def cache_result(ttl: float = 300.0):
    """Simple caching decorator with TTL.
    
    Args:
        ttl: Time to live in seconds
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}
        lock = Lock()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            with lock:
                now = time.time()
                
                # Check if we have a valid cached result
                if key in cache and key in cache_times:
                    if now - cache_times[key] < ttl:
                        logger.debug(
                            "Cache hit",
                            function=func.__name__,
                            key_hash=hash(key) % 10000
                        )
                        return cache[key]
                
                # Cache miss or expired, call function
                result = func(*args, **kwargs)
                cache[key] = result
                cache_times[key] = now
                
                # Clean up old entries
                expired_keys = [
                    k for k, t in cache_times.items()
                    if now - t >= ttl
                ]
                for k in expired_keys:
                    cache.pop(k, None)
                    cache_times.pop(k, None)
                
                logger.debug(
                    "Cache miss",
                    function=func.__name__,
                    key_hash=hash(key) % 10000,
                    cache_size=len(cache)
                )
                
                return result
        
        return wrapper
    
    return decorator


def validate_inputs(**validators):
    """Input validation decorator.
    
    Args:
        **validators: Keyword arguments with parameter names and validation functions
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate inputs
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(
                            f"Invalid value for parameter '{param_name}': {value}"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator