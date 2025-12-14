"""
RETRY UTILITIES - Robust retry logic with exponential backoff

Provides:
- Decorator for automatic retries
- Async retry support
- Customizable backoff strategies
- Error classification
"""

import asyncio
import functools
import random
import time
from dataclasses import dataclass
from typing import Type, Tuple, Optional, Callable, Any, Union
from loguru import logger


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True  # Add random jitter to prevent thundering herd
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    on_retry: Optional[Callable[[Exception, int], None]] = None


class RetryError(Exception):
    """Raised when all retries are exhausted."""
    def __init__(self, message: str, last_exception: Exception, attempts: int):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts


def calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool
) -> float:
    """Calculate delay for a given attempt with exponential backoff."""
    delay = min(base_delay * (exponential_base ** attempt), max_delay)
    if jitter:
        delay = delay * (0.5 + random.random())  # 50-150% of calculated delay
    return delay


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator for retrying synchronous functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delays
        retry_exceptions: Tuple of exceptions to retry on
        on_retry: Callback function called on each retry

    Example:
        @retry(max_retries=3, base_delay=1.0)
        def fetch_data():
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        delay = calculate_delay(
                            attempt, base_delay, max_delay, exponential_base, jitter
                        )
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {delay:.1f}s due to: {e}"
                        )

                        if on_retry:
                            on_retry(e, attempt + 1)

                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries} retries exhausted for {func.__name__}: {e}"
                        )

            raise RetryError(
                f"Failed after {max_retries + 1} attempts",
                last_exception,
                max_retries + 1
            )

        return wrapper
    return decorator


def async_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator for retrying async functions with exponential backoff.

    Example:
        @async_retry(max_retries=3, base_delay=1.0)
        async def fetch_data():
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        delay = calculate_delay(
                            attempt, base_delay, max_delay, exponential_base, jitter
                        )
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {delay:.1f}s due to: {e}"
                        )

                        if on_retry:
                            on_retry(e, attempt + 1)

                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries} retries exhausted for {func.__name__}: {e}"
                        )

            raise RetryError(
                f"Failed after {max_retries + 1} attempts",
                last_exception,
                max_retries + 1
            )

        return wrapper
    return decorator


async def retry_async_operation(
    operation: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> Any:
    """
    Retry an async operation with configurable backoff.

    Args:
        operation: Async function to retry
        *args: Positional arguments for operation
        config: RetryConfig instance (uses defaults if not provided)
        **kwargs: Keyword arguments for operation

    Returns:
        Result of successful operation

    Example:
        result = await retry_async_operation(
            fetch_data,
            url,
            config=RetryConfig(max_retries=5)
        )
    """
    if config is None:
        config = RetryConfig()

    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            return await operation(*args, **kwargs)
        except config.retry_exceptions as e:
            last_exception = e

            if attempt < config.max_retries:
                delay = calculate_delay(
                    attempt,
                    config.base_delay,
                    config.max_delay,
                    config.exponential_base,
                    config.jitter
                )
                logger.warning(
                    f"Retry {attempt + 1}/{config.max_retries} "
                    f"after {delay:.1f}s due to: {e}"
                )

                if config.on_retry:
                    config.on_retry(e, attempt + 1)

                await asyncio.sleep(delay)

    raise RetryError(
        f"Failed after {config.max_retries + 1} attempts",
        last_exception,
        config.max_retries + 1
    )


def retry_sync_operation(
    operation: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> Any:
    """
    Retry a sync operation with configurable backoff.

    Args:
        operation: Sync function to retry
        *args: Positional arguments for operation
        config: RetryConfig instance (uses defaults if not provided)
        **kwargs: Keyword arguments for operation

    Returns:
        Result of successful operation
    """
    if config is None:
        config = RetryConfig()

    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            return operation(*args, **kwargs)
        except config.retry_exceptions as e:
            last_exception = e

            if attempt < config.max_retries:
                delay = calculate_delay(
                    attempt,
                    config.base_delay,
                    config.max_delay,
                    config.exponential_base,
                    config.jitter
                )
                logger.warning(
                    f"Retry {attempt + 1}/{config.max_retries} "
                    f"after {delay:.1f}s due to: {e}"
                )

                if config.on_retry:
                    config.on_retry(e, attempt + 1)

                time.sleep(delay)

    raise RetryError(
        f"Failed after {config.max_retries + 1} attempts",
        last_exception,
        config.max_retries + 1
    )


# Common retry configurations
NETWORK_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    retry_exceptions=(ConnectionError, TimeoutError, OSError),
)

API_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    base_delay=2.0,
    max_delay=60.0,
    jitter=True,
)

DATABASE_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=0.5,
    max_delay=10.0,
)
