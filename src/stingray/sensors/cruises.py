"""NES-LTER cruise list lookup shared by the CLI and batch flows."""

from __future__ import annotations

import pandas as pd

CRUISES_URL = "https://nes-lter-api.whoi.edu/api/ctd/cruises/get/all"


def load_cruises(min_start_date: str = "1900-01-01", fallback_days: int = 7) -> list[dict]:
    """Return cruises (name, start, end) sorted by start time."""
    cruises = pd.read_csv(CRUISES_URL)
    for column in ("start_time", "end_time"):
        cruises[column] = pd.to_datetime(cruises[column], errors="coerce", utc=True).dt.tz_localize(None)
    cruises = cruises.dropna(subset=["start_time"])
    cruises = cruises[cruises["start_time"] >= pd.Timestamp(min_start_date)].sort_values("start_time")
    cruises["name"] = cruises["name"].astype(str).str.upper()
    cruises["end_time"] = cruises["end_time"].fillna(
        cruises["start_time"] + pd.Timedelta(days=fallback_days)
    )
    return [
        {
            "name": row.name,
            "start": row.start_time.strftime("%Y-%m-%d"),
            "end": row.end_time.strftime("%Y-%m-%d"),
        }
        for row in cruises.itertuples(index=False)
    ]
