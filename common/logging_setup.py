"""
logging_setup.py

Shared logger configuration: writes timestamped messages to both the
console and a persistent log file under logs/.

Log files are opened in append mode (Python's logging default), so
repeated runs accumulate in the same file rather than overwriting each
other -- each run is marked with a separator line so individual runs
stay easy to find within the accumulated history.
"""

import logging
import os
import sys
import warnings
from pathlib import Path

from pydantic.warnings import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)


def setup_logger(name, log_file, log_dir=None):
    """Configure and return a named logger that writes to both a file and stdout.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        log_file: Filename (not full path) for the log file.
        log_dir: Directory in which to create the log file. Created if absent.
            Defaults to the LOG_DIR environment variable, or "logs" relative to
            the working directory. The env var exists for containers: the app's
            working directory is not necessarily writable by a non-root user,
            and a relative path resolves somewhere a mounted volume is not.

    Returns:
        logging.Logger: Configured logger instance. Returns the existing logger
        unchanged if it has already been initialised (idempotent).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    log_dir_path = Path(log_dir or os.environ.get("LOG_DIR", "logs"))
    try:
        log_dir_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        # A read-only or root-owned filesystem must not stop the process from
        # starting -- console logging still works, and losing the file log is a
        # far smaller problem than a container that will not boot.
        log_dir_path = None
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_dir_path is not None:
        try:
            file_handler = logging.FileHandler(log_dir_path / log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:
            # Same reasoning as the mkdir above: the directory can exist and
            # still be unwritable.
            pass

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def log_run_separator(logger, title):
    """A visible divider marking where a new run starts within an accumulated log file."""
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)
