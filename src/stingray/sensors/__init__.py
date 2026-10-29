from __future__ import annotations

from stingray.sensors.fluorometer import (
    calibrate_backscatter,
    calibrate_chlorophyll,
)
from stingray.sensors.par import calibrate_par


def calibrate(sensor_type, raw_value, year=None):
    sensor_type = str(sensor_type).strip().lower()

    if sensor_type == "chlorophyll":
        return calibrate_chlorophyll(raw_value, year=year)
    elif sensor_type == "backscatter":
        return calibrate_backscatter(raw_value, year=year)
    elif sensor_type == "par":
        return calibrate_par(raw_value, year=year)

    raise ValueError(f"Unknown sensor type: {sensor_type}")