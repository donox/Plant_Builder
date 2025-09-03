# core/logging_setup.py
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Union, List

# Prevent double configuration across reloads
_CONFIGURED = False

# Define custom logging level
COORD_LEVEL = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(COORD_LEVEL, "COORD")

DISPLAY_LEVELS = ["COORD", "ERROR", "WARNING", "INFO"]


def coord(self, message, *args, **kwargs):
    """Log with COORD level"""
    if self.isEnabledFor(COORD_LEVEL):
        self._log(COORD_LEVEL, message, args, **kwargs)


# Add the coord method to Logger class
logging.Logger.coord = coord


class LevelFilter(logging.Filter):
    """Filter that only allows specific log levels"""

    def __init__(self, allowed_levels: List[Union[int, str]]):
        super().__init__()
        self.allowed_levels = set()
        for level in allowed_levels:
            if isinstance(level, str):
                if level.upper() == "COORD":
                    level_num = COORD_LEVEL
                else:
                    level_num = logging.getLevelName(level.upper())
                    if isinstance(level_num, str):  # Unknown level
                        continue
            else:
                level_num = level
            self.allowed_levels.add(level_num)
        # print(f"DEBUG: Filter will allow levels: {self.allowed_levels}")

    def filter(self, record):
        allowed = record.levelno in self.allowed_levels
        # print(f"DEBUG: Record level {record.levelno} ({record.levelname}) -> {'ALLOWED' if allowed else 'BLOCKED'}")
        return allowed


def force_reset_logging():
    """Aggressively reset all logging configuration"""
    # Get all existing loggers
    loggers = [logging.getLogger(name) for name in logging.Logger.manager.loggerDict]
    loggers.append(logging.getLogger())  # Include root logger

    # Remove all handlers from all loggers
    for logger in loggers:
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True

    # Clear the root logger specifically
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.NOTSET)

    print("DEBUG: All loggers reset")


def setup_root_logger(
        level: int = logging.WARNING,
        log_dir: Optional[Union[str, Path]] = None,
        console_level: Optional[int] = None,
        allowed_levels: Optional[List[Union[int, str]]] = None,
) -> Path:
    """
    Configure root logger once, idempotently.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return Path(log_dir or Path.home() / ".plantbuilder") / "plantbuilder.log"

    print("DEBUG: Setting up root logger...")

    # Force reset everything
    force_reset_logging()

    if isinstance(log_dir, int) and isinstance(level, int):
        level, log_dir = log_dir, None

    log_dir = Path(log_dir or Path.home() / ".plantbuilder")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "plantbuilder.log"

    root = logging.getLogger()

    # Set root logger level
    if allowed_levels is not None:
        # Set to the lowest possible level to let everything through to handlers
        root.setLevel(1)  # Lower than DEBUG (10)
        print(f"DEBUG: Root level set to 1, filtering with: {allowed_levels}")
    else:
        root.setLevel(logging.DEBUG)
        print("DEBUG: Root level set to DEBUG")

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(1)  # Let filter handle it
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s:%(name)s:%(message)s"))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(1)  # Let filter handle it
    ch.setFormatter(logging.Formatter("%(message)s"))

    # Add level filter if specified
    if allowed_levels is not None:
        level_filter = LevelFilter(allowed_levels)
        fh.addFilter(level_filter)
        ch.addFilter(level_filter)
        print("DEBUG: Filters applied to both handlers")
    else:
        fh.setLevel(level)
        ch.setLevel(console_level if console_level is not None else level)

    root.addHandler(fh)
    root.addHandler(ch)

    print(f"DEBUG: Root logger has {len(root.handlers)} handlers")
    print(f"DEBUG: Root logger level: {root.level}")

    _CONFIGURED = True
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Return a logger; configure root once on first use."""
    global _CONFIGURED
    if not _CONFIGURED:
        try:
            setup_root_logger(allowed_levels=DISPLAY_LEVELS)
        except Exception as e:
            # print(f"DEBUG: Exception during setup: {e}")
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            _CONFIGURED = True

    logger = logging.getLogger(name)
    # print(f"DEBUG: Created logger '{name}', level={logger.level}, handlers={len(logger.handlers)}")
    return logger


def log_coord(logger_name: str, message: str, *args, **kwargs):
    """Helper function to log COORD messages"""
    logger = get_logger(logger_name)
    if logger.isEnabledFor(COORD_LEVEL):
        logger._log(COORD_LEVEL, message, args, **kwargs)

# at top
import os
# ... existing imports ...

# --- NEW: level normalization ---
def _normalize_levels(levels):
    out = []
    for lv in levels or []:
        if isinstance(lv, str):
            s = lv.upper()
            if s == "COORD":
                out.append(COORD_LEVEL)
            else:
                n = logging.getLevelName(s)
                if isinstance(n, int):
                    out.append(n)
        else:
            out.append(int(lv))
    return out

# --- NEW: change DISPLAY_LEVELS at runtime and reapply filters ---
def apply_display_levels(levels):
    """
    Update DISPLAY_LEVELS and (if already configured) reapply LevelFilter
    on all root handlers so only these levels show up.
    """
    global DISPLAY_LEVELS
    DISPLAY_LEVELS = list(levels)
    if not _CONFIGURED:
        return  # get_logger() will configure with the new list on first use

    root = logging.getLogger()
    # remove any prior LevelFilter and add a fresh one
    for h in root.handlers:
        h.filters = [f for f in h.filters if not isinstance(f, LevelFilter)]
        h.addFilter(LevelFilter(DISPLAY_LEVELS))
    # make sure root passes everything to handlers (filter will do the gating)
    root.setLevel(1)

# --- NEW: optional env override at import time ---
_env = os.getenv("PB_DISPLAY_LEVELS")  # e.g. "ERROR,WARNING,COORD"
if _env:
    apply_display_levels([p.strip() for p in _env.split(",") if p.strip()])
