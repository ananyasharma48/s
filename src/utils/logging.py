"""Minimal structured logging setup."""

import logging
from pathlib import Path


def configure_logging(log_path: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure a plain, timestamped console logger and optional file handler."""
    logger = logging.getLogger("single_cell_proteomics_mil")
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_path is not None:
        destination = Path(log_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(destination, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
