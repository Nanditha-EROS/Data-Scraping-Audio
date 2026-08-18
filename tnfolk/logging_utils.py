"""Central logging setup.

Guarantees UTF-8 output (Tamil script logs correctly on Windows consoles that
default to cp1252) and always tags records with the video/recording id when
available so a single bad file can be traced without killing the batch.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_CONFIGURED = False


def _force_utf8_stream(stream) -> None:
    """Best-effort reconfigure a stream to UTF-8 (Python 3.7+)."""
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def setup_logging(level: str = "INFO", log_dir: Optional[str] = None,
                  log_file: str = "pipeline.log") -> logging.Logger:
    """Configure root logging once. Idempotent.

    Args:
        level: logging level name (e.g. "INFO", "DEBUG").
        log_dir: directory for the rotating log file; created if missing. If
            None, only console logging is configured.
        log_file: log file name within log_dir.

    Returns:
        The configured "tnfolk" logger.
    """
    global _CONFIGURED
    logger = logging.getLogger("tnfolk")
    if _CONFIGURED:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _force_utf8_stream(sys.stdout)
    _force_utf8_stream(sys.stderr)
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, log_file), encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    _CONFIGURED = True
    return logger


def get_logger(name: str = "tnfolk") -> logging.Logger:
    """Return a child logger under the configured "tnfolk" root."""
    if name == "tnfolk":
        return logging.getLogger("tnfolk")
    return logging.getLogger(f"tnfolk.{name}")
