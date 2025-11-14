import logging
import sys
import os
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger with both file and console handlers.

    The log level can be overridden at runtime with the environment variable
    `VIDGRAPH_LOG_LEVEL` (e.g. DEBUG, INFO, WARNING).
    """
    logger = logging.getLogger(name)

    # Allow overriding level with env var for easier debugging without code edits
    env_level = os.getenv("VIDGRAPH_LOG_LEVEL")
    if env_level:
        try:
            level = getattr(logging, env_level.upper())
        except Exception:
            # fallback to provided level
            level = level

    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # File handler
    file_handler = logging.FileHandler(logs_dir / f"{name}.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger