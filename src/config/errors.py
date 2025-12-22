# src/config/errors.py

class ConfigError(Exception):
    """Base class for configuration-related errors."""
    pass


class IncludeCycleError(ConfigError):
    """Raised when config includes form a cycle."""
    pass
