from loguru import logger
import sys
from typing import Any, Dict, Literal


def setup_logs(config: Dict[str, Any] | None = None) -> None:
    """
    Initialize logs with:
        - Console sink for DEBUG logs only
        - Optional file sink for INFO and above, defined by 'config'

    Parameters:
        config (dict): A Loguru-compatible configuration dictionary for file-based logging.
    """

    logger.remove()  # Clear all existing handlers

    # Console sink that ONLY logs debug
    logger.add(
        sys.stdout,
        level="DEBUG",
        filter=lambda record: record["level"].name == "DEBUG",
    )

    # File sink that logs 'INFO' and above
    if config:
        logger.add(**config)

    logger.info("Logger has been successfully initialized with provided configuration.")


def log(
    *,
    message: str,
    level: Literal[
        "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ] = "DEBUG",
) -> None:
    """
    Log a message at the specified level.

    - DEBUG messages are printed to the terminal only.
    - INFO, WARNING, and ERROR messages are written to the configured log file only.

    Parameters:
        message (str): The message to log.
        level (str): Logging level. Can be one of: DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL.

    Raises:
        ValueError: If an invalid log level is provided.
    """
    level = level.upper()

    try:
        logger.opt(depth=1).log(level, message)
    except ValueError as e:
        raise ValueError(f"Invalid log level specified. | {e}")
