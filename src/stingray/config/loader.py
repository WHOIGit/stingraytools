"""Load YAML configuration with bundled defaults and a safe fallback.

Each setting file (``calibrations``, ``units``, ``columns``, ``processing``)
ships with the package in ``config/defaults``. A user can override any of them
by pointing ``--config-dir`` (or the ``STINGRAY_CONFIG_DIR`` environment
variable) at a folder holding a file of the same name.

An override only needs the entries it changes: it is merged over the bundled
file, so anything it leaves out falls back to the default. A missing or
unreadable override file is reported and the defaults are used. Lists, such as
a sensor's calibration entries, are replaced as a whole, not merged.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR_ENV = "STINGRAY_CONFIG_DIR"
DEFAULTS_DIR = Path(__file__).parent / "defaults"

_config_dir: Path | None = None


def set_config_dir(path: str | Path | None) -> None:
    """Use ``path`` as the override folder for this process (None to clear)."""
    global _config_dir
    _config_dir = Path(path).expanduser() if path else None
    _load.cache_clear()


def override_dir() -> Path | None:
    if _config_dir is not None:
        return _config_dir
    value = os.environ.get(CONFIG_DIR_ENV)
    return Path(value).expanduser() if value else None


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


@lru_cache(maxsize=None)
def _load(name: str, override: str) -> dict[str, Any]:
    defaults = read_yaml(DEFAULTS_DIR / f"{name}.yaml")
    if not override:
        return defaults

    folder = Path(override)
    path = folder / f"{name}.yaml"
    if not folder.is_dir():
        logger.warning("Config directory not found: %s; using bundled %s", folder, name)
        return defaults
    if not path.is_file():
        logger.info("No %s.yaml in %s; using bundled defaults", name, folder)
        return defaults
    try:
        user = read_yaml(path)
    except Exception as exc:
        logger.warning("Ignoring unreadable %s (%s); using bundled %s", path, exc, name)
        return defaults
    logger.info("Using %s over bundled %s defaults", path, name)
    return deep_merge(defaults, user)


def load_config(name: str) -> dict[str, Any]:
    """Return the named setting file, merged over the bundled defaults."""
    folder = override_dir()
    return deepcopy(_load(name, str(folder) if folder else ""))
