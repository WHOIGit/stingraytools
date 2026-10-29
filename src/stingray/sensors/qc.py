"""First-pass quality control for raw sensor streams.

Run on each stream straight after it is read, before calibration: values that
fail the rules in ``qc.yaml`` are set to NaN. Each stream is checked
separately, with its own rules.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from stingray.config.loader import load_config

logger = logging.getLogger(__name__)

INVALID_TOLERANCE = 1e-3


def _out_of_range(values: np.ndarray, rule: dict) -> np.ndarray:
    bad = np.zeros(len(values), dtype=bool)
    if rule.get("min") is not None:
        bad |= values < rule["min"]
    if rule.get("max") is not None:
        bad |= values > rule["max"]
    for value in rule.get("invalid") or []:
        bad |= np.isclose(values, float(value), rtol=0.0, atol=INVALID_TOLERANCE)
    return bad


def apply_qc(
    df: pd.DataFrame,
    sensor: str,
    rules: dict | None = None,
    report: list | None = None,
) -> pd.DataFrame:
    """
    Set invalid values in one sensor stream to NaN.

    Parameters
    ----------
    df : DataFrame
        Raw readings of one sensor (changed in place and returned).
    sensor : str
        Stream name, as a section of qc.yaml (CTD, DVL, Fluorometer, ...).
    rules : dict, optional
        Column rules to use instead of the ones in qc.yaml.
    report : list, optional
        Receives one dict per checked column (sensor, column, rows, unparsable,
        flagged, percent) so the counts can be tabulated.

    A checked column is converted to numbers first; entries that cannot be read
    as numbers become NaN and are counted as ``unparsable``. A column the data
    does not have is skipped.
    """
    if rules is None:
        rules = load_config("qc").get(sensor) or {}
    if df.empty or not rules:
        return df

    for column, rule in rules.items():
        if column not in df.columns:
            logger.warning("QC %s: column %r not in the data; skipped", sensor, column)
            continue

        raw = df[column]
        values = pd.to_numeric(raw, errors="coerce")
        unparsable = int((values.isna() & raw.notna()).sum())
        bad = _out_of_range(values.to_numpy(dtype=float), rule)
        flagged = int(bad.sum())

        df[column] = values.mask(bad)

        if report is not None:
            report.append({
                "sensor": sensor,
                "column": column,
                "rows": len(df),
                "unparsable": unparsable,
                "flagged": flagged,
                "percent": 100.0 * flagged / len(df),
            })
        if flagged or unparsable:
            logger.info(
                "QC %s.%s: %s of %s values set to NaN (%.3f%%)%s",
                sensor,
                column,
                f"{flagged:,}",
                f"{len(df):,}",
                100.0 * flagged / len(df),
                f", plus {unparsable:,} unreadable" if unparsable else "",
            )
    return df
