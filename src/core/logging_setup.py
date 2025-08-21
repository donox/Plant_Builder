# core/logging_setup.py
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Union

# Prevent double configuration across reloads
_CONFIGURED = False

def setup_root_logger(
    level: int = logging.DEBUG,
    log_dir: Optional[Union[str, Path]] = None,
    console_level: Optional[int] = None,
) -> Path:
    """
    Configure root logger once, idempotently.

    Back-compat:
    - If someone calls setup_root_logger(DEBUG) thinking first arg is level,
      this still works (it's the default anyway).
    - If someone calls setup_root_logger(log_dir="..."), that also works via keyword.
    """
    global _CONFIGURED
    if _CONFIGURED:
        # already configured; return existing path (best effort)
        return Path(log_dir or Path.home() / ".plantbuilder") / "plantbuilder.log"

    # Guard: cope if the first arg was accidentally passed as log_dir=int (older API confusion)
    if isinstance(log_dir, int) and isinstance(level, int):
        # They probably did setup_root_logger(DEBUG) with an older signature in mind.
        level, log_dir = log_dir, None

    log_dir = Path(log_dir or Path.home() / ".plantbuilder")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "plantbuilder.log"

    root = logging.getLogger()
    root.setLevel(logging.WARNING)  # capture all; handlers filter below

    # Avoid duplicate handlers on reload
    root.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s:%(name)s:%(message)s"))
    root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(console_level if console_level is not None else level)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)

    _CONFIGURED = True
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Return a logger; configure root once on first use."""
    # Configure root if not configured yet
    global _CONFIGURED
    if not _CONFIGURED:
        try:
            setup_root_logger()
        except Exception:
            # Fall back to a minimal console-only logger to avoid import-time crashes
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            _CONFIGURED = True
    return logging.getLogger(name)
