"""Select a sensor calibration from calibrations.yaml."""

from __future__ import annotations

import logging
import datetime as dt

import pandas as pd

from stingray.config.loader import load_config

logger = logging.getLogger(__name__)


def _entry_start(entry: dict) -> pd.Timestamp:
    value = entry.get("valid_from")
    if value is None:
        return pd.Timestamp.min
    if isinstance(value, (dt.date, dt.datetime)):
        return pd.Timestamp(value)
    return pd.Timestamp(str(value))


def get_calibration(
    sensor: str,
    date=None,
    name: str | None = None,
    required: tuple[str, ...] = (),
) -> dict:
    """
    Return the calibration entry for ``sensor``.

    Selection order: an explicit ``name``; otherwise the entry in force on
    ``date`` (latest ``valid_from`` on or before it); otherwise the sensor's
    ``default`` entry, or its newest entry when no default is set. A date
    before the first entry uses the first entry. ``required`` lists the
    coefficients the caller needs; a missing one is a clear error.
    """
    config = load_config("calibrations")
    section = config.get(sensor)
    if not isinstance(section, dict) or not section.get("entries"):
        raise ValueError(f"No calibration entries for sensor '{sensor}' in calibrations.yaml")
    entries = sorted(section["entries"], key=_entry_start)

    if name is not None:
        matches = [entry for entry in entries if str(entry.get("name")) == str(name)]
        if not matches:
            available = ", ".join(str(entry.get("name")) for entry in entries)
            raise ValueError(
                f"No calibration named {name!r} for {sensor}; available: {available}"
            )
        entry, reason = matches[0], f"requested name {name}"
    elif date is not None:
        when = pd.Timestamp(date)
        in_force = [entry for entry in entries if _entry_start(entry) <= when]
        if in_force:
            entry, reason = in_force[-1], f"in force on {when.date()}"
        else:
            entry = entries[0]
            reason = f"first entry (no calibration valid on {when.date()})"
            logger.warning("%s: %s precedes every calibration; using %s", sensor, when.date(), entry.get("name"))
    else:
        default = section.get("default")
        by_default = [entry for entry in entries if str(entry.get("name")) == str(default)]
        entry, reason = (by_default[0], "default") if by_default else (entries[-1], "newest entry")

    missing = [key for key in required if key not in entry]
    if missing:
        raise ValueError(
            f"Calibration '{entry.get('name')}' for {sensor} is missing: {', '.join(missing)}"
        )
    logger.info("Calibration %s: %s (%s)", sensor, entry.get("name"), reason)
    return entry
