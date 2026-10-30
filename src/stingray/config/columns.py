# src/stingray/config/columns.py

from __future__ import annotations

from stingray.config.loader import load_config


def sled_columns() -> dict[str, str]:
    """Raw/merged column name -> output column name, from columns.yaml."""
    return {str(raw): str(out) for raw, out in load_config("columns").items()}
