# core/logging_setup.py
"""
Logging setup for PlantBuilder with custom COORD level.

Usage in modules:
    from core.logging_setup import get_logger
    logger = get_logger(__name__)

    logger.error("Error message")
    logger.warning("Warning message")
    logger.info("Progress message")
    logger.coord("Coordinate/geometry detail")
    logger.debug("Debug detail")

Default display levels: ERROR, WARNING, INFO
To include coordinate spam: set_display_levels(["ERROR", "WARNING", "INFO", "COORD"])
To see everything: set_display_levels(["ERROR", "WARNING", "INFO", "COORD", "DEBUG"])
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Union, Optional

# Custom COORD level between INFO (20) and WARNING (30)
COORD_LEVEL = 25
logging.addLevelName(COORD_LEVEL, "COORD")

# Default levels to display (console and file)
DEFAULT_DISPLAY_LEVELS = ["ERROR", "WARNING", "INFO", "DEBUG"]
_root_configured = False

# Global to track if we've configured the root logger
_root_configured = False


def coord(self, message, *args, **kwargs):
    """Log with COORD level for coordinate/geometry details"""
    if self.isEnabledFor(COORD_LEVEL):
        self._log(COORD_LEVEL, message, args, **kwargs)


# Add coord method to Logger class
logging.Logger.coord = coord


class LevelFilter(logging.Filter):
    """
    Filter that only allows specific log levels.
    Unlike setLevel(), this can skip intermediate levels.
    Example: Allow ERROR, WARNING, INFO but skip DEBUG and COORD
    """

    def __init__(self, allowed_levels: List[Union[int, str]]):
        super().__init__()
        self.allowed_levels = set()
        for level in allowed_levels:
            if isinstance(level, str):
                level_upper = level.upper()
                if level_upper == "COORD":
                    level_num = COORD_LEVEL
                else:
                    level_num = logging.getLevelName(level_upper)
                    if isinstance(level_num, str):  # Unknown level name
                        continue
            else:
                level_num = int(level)
            self.allowed_levels.add(level_num)

    def filter(self, record):
        return record.levelno in self.allowed_levels


def _get_log_directory() -> Path:
    """Get or create the log directory"""
    log_dir = Path.home() / ".plantbuilder"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _clear_log_file(log_path: Path):
    """Clear the log file by writing empty content"""
    log_path.write_text("", encoding="utf-8")


def setup_logging(
        display_levels: Optional[List[str]] = None,
        log_dir: Optional[Path] = None,
        clear_log: bool = True
) -> Path:
    """
    Configure root logger with file and console handlers.

    Args:
        display_levels: List of level names to display (e.g., ["ERROR", "WARNING", "INFO"])
                       If None, uses DEFAULT_DISPLAY_LEVELS
        log_dir: Directory for log file (default: ~/.plantbuilder)
        clear_log: If True, clear log file on startup

    Returns:
        Path to log file
    """
    global _root_configured

    # Use defaults if not specified
    if display_levels is None:
        display_levels = DEFAULT_DISPLAY_LEVELS

    if log_dir is None:
        log_dir = _get_log_directory()

    log_path = log_dir / "plantbuilder.log"

    # Only configure once (idempotent)
    if _root_configured:
        return log_path

    # Clear log file if requested
    if clear_log:
        _clear_log_file(log_path)

    # Get root logger
    root = logging.getLogger()
    root.setLevel(1)  # Pass everything to handlers, let filter decide

    # Remove any existing handlers (in case of reload)
    root.handlers.clear()

    # Create level filter
    level_filter = LevelFilter(display_levels)

    # File handler - detailed format
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(1)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )
    file_handler.addFilter(level_filter)
    root.addHandler(file_handler)

    # Console handler - simple format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(1)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.addFilter(level_filter)
    root.addHandler(console_handler)

    _root_configured = True

    return log_path


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module name.
    Configures root logger on first call.

    Usage:
        logger = get_logger(__name__)
    """
    global _root_configured

    if not _root_configured:
        log_path = setup_logging()
        # Only print this once on first logger creation
        print(f"Logging to: {log_path}")

    return logging.getLogger(name)


def set_display_levels(levels: List[str]):
    """
    Change which log levels are displayed at runtime.

    Args:
        levels: List of level names (e.g., ["ERROR", "WARNING", "INFO", "COORD", "DEBUG"])

    Example:
        # Show only errors and warnings
        set_display_levels(["ERROR", "WARNING"])

        # Include coordinate details
        set_display_levels(["ERROR", "WARNING", "INFO", "COORD"])
    """
    if not _root_configured:
        # Will be applied when setup_logging() is called
        return

    # Update filter on all root handlers
    root = logging.getLogger()
    new_filter = LevelFilter(levels)

    for handler in root.handlers:
        # Remove old LevelFilter instances
        handler.filters = [f for f in handler.filters if not isinstance(f, LevelFilter)]
        # Add new filter
        handler.addFilter(new_filter)