# utils/spatial.py

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

def convert_gps(raw_series, sign_series):
    signs = sign_series.map({'N': 1, 'S': -1, 'E': 1, 'W': -1})
    return signs * (np.floor(raw_series / 100) + (raw_series % 100) / 60)

DistanceMethod = Literal["haversine", "geodesic"]


class StationLocator:
    DEFAULT_STATION_REF_URL = "https://nes-lter-api.whoi.edu/api/stations/file"

    REQUIRED_COLUMNS = [
        "station",
        "startDate",
        "endDate",
        "decimalLatitude",
        "decimalLongitude",
    ]

    def __init__(
        self,
        station_reference: pd.DataFrame | str | None = None,
        max_distance_km: float | None = 2.0,
        distance_method: DistanceMethod = "haversine",
    ) -> None:
        if station_reference is None:
            st = pd.read_csv(self.DEFAULT_STATION_REF_URL)
        elif isinstance(station_reference, str):
            st = pd.read_csv(station_reference)
        else:
            st = station_reference.copy()

        st = self._prepare_station_reference(st)

        if distance_method not in ("haversine", "geodesic"):
            raise ValueError("distance_method must be 'haversine' or 'geodesic'")

        self.station_reference = st
        self.max_distance_km = max_distance_km
        self.distance_method = distance_method

    # -------------------------
    # preprocessing
    # -------------------------
    @classmethod
    def _prepare_station_reference(cls, st: pd.DataFrame) -> pd.DataFrame:
        st = st.copy()
        st.columns = st.columns.str.strip()

        missing = [c for c in cls.REQUIRED_COLUMNS if c not in st.columns]
        if missing:
            raise ValueError(f"station_reference missing required columns: {missing}")

        st["station"] = st["station"].astype(str).str.strip()

        st["startDate"] = pd.to_datetime(
            st["startDate"], errors="coerce", utc=True
        ).dt.tz_localize(None)

        st["endDate"] = st["endDate"].replace("current", pd.NA)
        st["endDate"] = pd.to_datetime(
            st["endDate"], errors="coerce", utc=True
        ).dt.tz_localize(None)

        st["endDate_filled"] = st["endDate"].fillna(pd.Timestamp("2100-12-31"))

        st["decimalLatitude"] = pd.to_numeric(st["decimalLatitude"], errors="coerce")
        st["decimalLongitude"] = pd.to_numeric(st["decimalLongitude"], errors="coerce")

        st = st.dropna(
            subset=["station", "startDate", "decimalLatitude", "decimalLongitude"]
        ).copy()

        if st.empty:
            raise ValueError("station_reference contains no valid station rows")

        return st

    @staticmethod
    def _normalize_timestamp(timestamp: Any) -> pd.Timestamp:
        ts = pd.to_datetime(timestamp, errors="coerce", utc=True)
        if pd.isna(ts):
            raise ValueError("timestamp is missing or invalid")
        return ts.tz_localize(None)

    # -------------------------
    # distance methods
    # -------------------------
    @staticmethod
    def _haversine_km(
        lat1: float | np.ndarray,
        lon1: float | np.ndarray,
        lat2: float | np.ndarray,
        lon2: float | np.ndarray,
    ) -> np.ndarray:
        r = 6371.0088

        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        return r * c

    @staticmethod
    def _geodesic_km(
        lat: float,
        lon: float,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> np.ndarray:
        """
        Uses geopy geodesic (ellipsoidal). Falls back with helpful error if missing.
        """
        try:
            from geopy.distance import geodesic
        except ImportError as e:
            raise ImportError(
                "geopy is required for distance_method='geodesic'. "
                "Install with `pip install geopy`."
            ) from e

        out = []
        for la, lo in zip(lats, lons):
            out.append(geodesic((lat, lon), (la, lo)).km)
        return np.array(out)

    # -------------------------
    # core logic
    # -------------------------
    def active_stations(self, timestamp: Any) -> pd.DataFrame:
        ts = self._normalize_timestamp(timestamp)

        active = self.station_reference[
            (self.station_reference["startDate"] <= ts)
            & (ts <= self.station_reference["endDate_filled"])
        ].copy()

        if active.empty:
            raise ValueError(f"No active stations found for timestamp {ts}")

        return active

    def station_distances(
        self,
        lat: float,
        lon: float,
        timestamp: Any,
    ) -> pd.Series:
        if pd.isna(lat) or pd.isna(lon):
            raise ValueError("lat/lon is missing or invalid")

        active = self.active_stations(timestamp)

        lats = active["decimalLatitude"].to_numpy()
        lons = active["decimalLongitude"].to_numpy()

        if self.distance_method == "haversine":
            distances = self._haversine_km(lat, lon, lats, lons)
        else:
            distances = self._geodesic_km(lat, lon, lats, lons)

        return pd.Series(distances, index=active.index, name="distance_km")

    def nearest_station(
        self,
        lat: float,
        lon: float,
        timestamp: Any,
        max_distance_km: float | None = None,
    ):
        distances = self.station_distances(lat, lon, timestamp)

        i = distances.idxmin()
        d_km = float(distances.loc[i])

        threshold = self.max_distance_km if max_distance_km is None else max_distance_km
        if threshold is not None and d_km > threshold:
            return np.nan, np.nan

        station_name = self.station_reference.loc[i, "station"]
        return station_name, d_km