from __future__ import annotations

import numpy as np

from stingray.sensors.calibration import get_calibration


def calibrate_par(raw_value, year: str | None = None, date=None):
    """
    Convert raw PAR counts to calibrated PAR.

    The calibration comes from calibrations.yaml: the entry named ``year``, else
    the one in force on ``date``, else the sensor's default.
    """
    params = get_calibration("par", date=date, name=year, required=("a0", "a1", "Im"))
    raw = np.asarray(raw_value, dtype=np.float64)
    return params["Im"] * 10 ** ((raw - params["a0"]) / params["a1"])
