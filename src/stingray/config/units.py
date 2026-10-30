# src/stingray/config/units.py

from __future__ import annotations

from stingray.config.loader import load_config


def sled_units() -> dict[str, str]:
    """Output column name -> display unit, from units.yaml."""
    return {str(name): str(unit) for name, unit in load_config("units").items()}


def unit_for(column: str) -> str:
    """Unit of an output column; empty when none is configured."""
    return sled_units().get(column, "")
