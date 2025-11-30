from functools import wraps

from loguru import logger


def log(func):
    """A decorator to log function calls, arguments, and return values."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Log function start and arguments
        logger.info(f"Function '{func.__name__}' | args: {args} | kwargs: {kwargs}")
        try:
            # Call the original function
            result = func(*args, **kwargs)
            # Log function end and return value
            logger.info(f"Function '{func.__name__}' finished, returned: {result}")
            return result
        except Exception as e:
            # Log any exceptions
            logger.error(
                f"Function '{func.__name__}' raised an exception: {e}", exc_info=True
            )
            raise

    return wrapper
