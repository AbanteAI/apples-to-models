import asyncio
from functools import wraps


def async_retry(tries=8, delay=0.1, backoff=2):
    """Retry decorator for async functions"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 0:
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    _tries -= 1
                    if _tries == 0:
                        raise
                    await asyncio.sleep(_delay)
                    _delay *= backoff
            return None

        return wrapper

    return decorator
