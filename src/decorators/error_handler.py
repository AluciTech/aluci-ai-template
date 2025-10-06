import traceback
from functools import wraps
from typing import Any
from utils.log_utils import log


def catch_errors(default_return: Any | None = None):
    """
    Decorator to catch and log unexpected exceptions.

    Args:
        default_return: Value to return if an exception occurs.

    Usage:
        @catch_errors()
        def some_func(): ...

        @catch_errors(default_return=(False, None, None))
        def get_data(): ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log(message=f"Exception: {e}\n{traceback.format_exc()}", level="ERROR")
                return default_return

        return wrapper

    return decorator
