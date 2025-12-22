# plantbuilder/config/loader.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Set, Optional

import yaml

from .errors import ConfigError, IncludeCycleError


_VAR_RE = re.compile(r"\$\{([^}]+)\}")  # ${name} or ${ENV:HOME}


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge dict b into dict a and return merged dict (non-destructive).
    - dict + dict -> recursively merge
    - list + list -> by default REPLACE (see notes below)
    - scalar -> replace
    """
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"Top-level YAML must be a mapping/dict: {path}")
        return data
    except Exception as e:
        raise ConfigError(f"Failed to read YAML '{path}': {e}") from e


def _resolve_string(s: str, params: Dict[str, Any]) -> Any:
    """
    Resolve ${var} substitutions.
    - If the *entire* string is exactly one variable (e.g. "${diameter}"),
      return the underlying value with its original type (float/int/bool/etc).
    - If the variable appears inside a larger string, coerce to str and substitute.
    """
    s = s.strip()

    # Exact match: "${...}" -> return typed value
    m = re.fullmatch(r"\$\{([^}]+)\}", s)
    if m:
        key = m.group(1).strip()
        if key.startswith("ENV:"):
            env_key = key.split(":", 1)[1].strip()
            return os.environ.get(env_key, "")
        if key not in params:
            raise ConfigError(f"Unknown parameter '{key}' in string: {s}")
        return params[key]  # <-- typed, not str()

    # Otherwise do string substitution
    def repl(match: re.Match) -> str:
        key = match.group(1).strip()
        if key.startswith("ENV:"):
            env_key = key.split(":", 1)[1].strip()
            return os.environ.get(env_key, "")
        if key not in params:
            raise ConfigError(f"Unknown parameter '{key}' in string: {s}")
        return str(params[key])

    return _VAR_RE.sub(repl, s)




def _resolve_params(obj: Any, params: Dict[str, Any]) -> Any:
    if isinstance(obj, str):
        return _resolve_string(obj, params) if "${" in obj else obj
    if isinstance(obj, list):
        return [_resolve_params(x, params) for x in obj]
    if isinstance(obj, dict):
        return {k: _resolve_params(v, params) for k, v in obj.items()}
    return obj


def _norm_include_path(base_dir: str, inc: str, yaml_base_dir: Optional[str]) -> str:
    """
    - If inc is relative:
        - first resolve relative to config file directory
        - optionally allow 'base/...' to resolve under yaml_base_dir
    """
    inc = inc.strip()
    if os.path.isabs(inc):
        return inc

    # Special handling: "base/..." maps to packaged yaml base dir
    if yaml_base_dir and (inc.startswith("base/") or inc.startswith("base\\")):
        return os.path.join(yaml_base_dir, inc.replace("base/", "").replace("base\\", ""))

    return os.path.normpath(os.path.join(base_dir, inc))


@dataclass(frozen=True)
class LoadedConfig:
    path: str
    data: Dict[str, Any]


def load_config(
    config_path: str,
    *,
    yaml_base_dir: Optional[str] = None,
    validate: bool = True,
) -> LoadedConfig:
    """
    Load a config file with:
    - include processing
    - deep merging
    - parameter resolution

    yaml_base_dir: filesystem dir that corresponds to your packaged base YAML,
                  e.g. plantbuilder/yaml/base
    """
    config_path = os.path.abspath(config_path)
    base_dir = os.path.dirname(config_path)

    visited: Set[str] = set()
    merged: Dict[str, Any] = {}

    def load_one(path: str) -> Dict[str, Any]:
        abspath = os.path.abspath(path)
        if abspath in visited:
            raise IncludeCycleError(f"Include cycle detected at: {abspath}")
        visited.add(abspath)

        data = _read_yaml(abspath)

        includes = data.get("include", []) or []
        if includes and not isinstance(includes, list):
            raise ConfigError(f"'include' must be a list in {abspath}")

        # Load includes first
        acc: Dict[str, Any] = {}
        for inc in includes:
            if not isinstance(inc, str):
                raise ConfigError(f"include entries must be strings: {abspath}")
            inc_path = _norm_include_path(os.path.dirname(abspath), inc, yaml_base_dir)
            acc = _deep_merge(acc, load_one(inc_path))

        # Then merge this file over included base
        data_no_include = dict(data)
        data_no_include.pop("include", None)

        return _deep_merge(acc, data_no_include)

    merged = load_one(config_path)

    # Params: allow includes to define params, but project file can override
    params = merged.get("params", {}) or {}
    if not isinstance(params, dict):
        raise ConfigError("'params' must be a mapping/dict")

    merged = _resolve_params(merged, params)

    if validate:
        _validate_minimal(merged, config_path)

    return LoadedConfig(path=config_path, data=merged)


def _validate_minimal(cfg: Dict[str, Any], path: str) -> None:
    # Your Driver expects these sections today:
    # metadata/global_settings/output_files/workflow :contentReference[oaicite:6]{index=6}
    for key in ("global_settings", "output_files", "workflow"):
        if key not in cfg:
            raise ConfigError(f"Missing required top-level key '{key}' in {path}")

    if not isinstance(cfg["workflow"], list):
        raise ConfigError(f"'workflow' must be a list in {path}")
