# utils/temporal.py

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd


MATLAB_EPOCH_OFFSET_DAYS = 367.0
DEFAULT_ORIGIN_DATE = datetime(1904, 1, 1)

def parse_datetimes(values):
    """
    Parse datetime strings quickly, falling back to tolerant parsing.

    A fixed ISO-8601 format avoids per-value format inference. Anything that
    is not strictly ISO-8601 is re-parsed with errors coerced to NaT, as before.
    """
    try:
        return pd.to_datetime(values, format="ISO8601")
    except (ValueError, TypeError):
        return pd.to_datetime(values, errors="coerce")


def convert_timestamp(timestamp, origin_date: datetime = DEFAULT_ORIGIN_DATE):
    """
    Convert seconds since origin_date to pandas datetime(s) and MATLAB datenum(s).
    """
    times = pd.to_datetime(timestamp, unit="s", origin=origin_date)
    matdate = datenum(times)
    return times, matdate


def matdate2timestamp(matdate, origin_date: datetime = DEFAULT_ORIGIN_DATE):
    """
    Convert MATLAB datenum(s) to seconds since origin_date.
    """
    def one(x):
        dt = datestr(x, as_str=False)
        return (dt - origin_date).total_seconds()

    if isinstance(matdate, (int, float, np.integer, np.floating)):
        return one(matdate)
    if isinstance(matdate, (list, tuple, np.ndarray, pd.Series)):
        return [one(x) for x in matdate]
    raise TypeError("Input must be a scalar or array-like MATLAB date number")


def rawdate2date(series: pd.Series) -> pd.Series:
    """
    Convert YYYYDDD integer/string values to pandas datetime.

    Example:
        2024032 -> 2024-02-01
    """
    s = series.astype(str).str.strip()
    year = s.str[:4]
    day_of_year = s.str[4:].astype(int) - 1
    return pd.to_datetime(year, format="%Y") + pd.to_timedelta(day_of_year, unit="D")


def rawtime2time(series: pd.Series) -> pd.Series:
    """
    Convert hour-decimal values to HH:MM:SS strings.

    Example:
        13.5 -> '13:30:00'
    """
    day_fractions = series.astype(float) / 24.0
    td = pd.to_timedelta(day_fractions, unit="D")

    return td.dt.components.apply(
        lambda x: f"{x['hours']:02}:{x['minutes']:02}:{x['seconds']:02}",
        axis=1,
    )

def _to_python_datetime(dt) -> datetime:
    if isinstance(dt, pd.Timestamp):
        return pd.to_datetime(dt).round("us").to_pydatetime()
    if isinstance(dt, datetime):
        return dt
    raise TypeError("Input must be a datetime.datetime or pandas.Timestamp")


def datenum(dts):
    """
    Convert datetime object(s) to MATLAB serial date number(s).
    """
    base_date = datetime(1, 1, 1)

    def date2num(dt) -> float:
        dt = _to_python_datetime(dt)
        delta = dt - base_date
        return (
            delta.days
            + delta.seconds / 86400.0
            + delta.microseconds / 86400.0 / 1e6
            + MATLAB_EPOCH_OFFSET_DAYS
        )

    if isinstance(dts, (datetime, pd.Timestamp)):
        return date2num(dts)
    if isinstance(dts, (list, tuple, np.ndarray, pd.Series)):
        return [date2num(dt) for dt in dts]
    raise TypeError(
        "Input must be a datetime-like object or an array-like of datetime-like objects"
    )


def datestr(date_numbers, fmt=0, as_str=False):
    """
    Convert MATLAB serial date number(s) to datetime object(s) or formatted string(s).

    Supported fmt values:
        0: '%d-%b-%Y %H:%M:%S'
        1: '%d-%b-%Y'
        2: '%m/%d/%y'
        3: '%Y-%m-%d'
        4: '%d/%m/%Y'
        5: '%H:%M:%S'
    """
    matlab_base_date = datetime(1, 1, 1)

    formats = {
        0: "%d-%b-%Y %H:%M:%S",
        1: "%d-%b-%Y",
        2: "%m/%d/%y",
        3: "%Y-%m-%d",
        4: "%d/%m/%Y",
        5: "%H:%M:%S",
    }

    def num2date(date_number):
        dt = matlab_base_date + timedelta(days=float(date_number) - MATLAB_EPOCH_OFFSET_DAYS)
        if not as_str:
            return dt
        if isinstance(fmt, str):
            return dt.strftime(fmt)
        return dt.strftime(formats.get(fmt, formats[0]))

    if date_numbers is None:
        return None
    if isinstance(date_numbers, (int, float, np.integer, np.floating)):
        return num2date(date_numbers)
    if isinstance(date_numbers, (list, tuple, np.ndarray, pd.Series)):
        return [num2date(num) for num in date_numbers]
    raise TypeError(
        "Input must be a scalar MATLAB date number or an array-like of date numbers"
    )


