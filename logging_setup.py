"""Shared project logging configuration."""

import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_FILE = PROJECT_ROOT / 'stack_feed.log'

_file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
_file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(
    logging.Formatter('%(name)s - %(levelname)s - %(message)s')
)


def setup_logger(name, level=logging.DEBUG):
    """Create a named logger that writes to the single root log file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if _file_handler not in logger.handlers:
        logger.addHandler(_file_handler)
    if _console_handler not in logger.handlers:
        logger.addHandler(_console_handler)

    return logger
