"""Centralized logging via loguru with rich formatting."""
import sys
import os
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logger(level: str = "INFO", log_file: str = "logs/hypestrat.log") -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    logger.add(
        log_file,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="100 MB",
        retention="30 days",
        compression="gz",
        enqueue=True,
    )


setup_logger()

__all__ = ["logger", "setup_logger"]
