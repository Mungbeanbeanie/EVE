"""Logging setup.

Fully implemented — this is plumbing. Use `logging.getLogger(__name__)` in each
module and call `setup_logging()` once at startup (main.py already does this).
"""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once, with a readable console format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
