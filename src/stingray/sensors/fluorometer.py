from __future__ import annotations

import numpy as np

from stingray.sensors.calibration import get_calibration

LINEAR_COEFFICIENTS = ("scale_factor", "dark_count")


def _linear(sensor: str, raw_value, year, date):
    params = get_calibration(sensor, date=date, name=year, required=LINEAR_COEFFICIENTS)
    raw = np.asarray(raw_value, dtype=np.float64)
    return params["scale_factor"] * (raw - params["dark_count"])


def calibrate_chlorophyll(raw_value, year: str | None = None, date=None):
    """
    Convert raw chlorophyll counts to calibrated chlorophyll concentration.

    The calibration comes from calibrations.yaml: the entry named ``year``, else
    the one in force on ``date``, else the sensor's default.
    """
    return _linear("chlorophyll", raw_value, year, date)


def calibrate_backscatter(raw_value, year: str | None = None, date=None):
    """
    Convert raw backscatter counts to calibrated backscatter.
    """
    return _linear("backscattering", raw_value, year, date)
