"""
Persistent recent-files list stored in FreeCAD's parameter store.
Falls back silently when FreeCAD is not available (e.g. unit tests).
"""
from __future__ import annotations
import os

_PARAM_PATH = "User parameter:BaseApp/Preferences/PlantBuilder"
_PARAM_KEY  = "RecentFiles"
_MAX_RECENT = 8


def _get_param():
    try:
        import FreeCAD
        return FreeCAD.ParamGet(_PARAM_PATH)
    except Exception:
        return None


def load_recent() -> list[str]:
    """Return up to _MAX_RECENT paths, most-recent first, skipping missing files."""
    param = _get_param()
    if param is None:
        return []
    raw = param.GetString(_PARAM_KEY, "")
    if not raw:
        return []
    return [p for p in raw.split("\n") if p and os.path.isfile(p)][:_MAX_RECENT]


def push_recent(path: str) -> None:
    """Prepend path, deduplicate, trim to _MAX_RECENT."""
    param = _get_param()
    if param is None:
        return
    paths = [p for p in load_recent() if p != path]
    paths.insert(0, path)
    param.SetString(_PARAM_KEY, "\n".join(paths[:_MAX_RECENT]))


def clear_recent() -> None:
    param = _get_param()
    if param is None:
        return
    param.SetString(_PARAM_KEY, "")
