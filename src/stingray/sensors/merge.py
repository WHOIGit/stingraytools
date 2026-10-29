from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stingray.config.columns import sled_columns
from stingray.config.loader import load_config
from stingray.io.csv import read_csv_parallel, write_csv_atomic
from stingray.io.indexing import load_or_build_file_index, filter_file_index
from stingray.utils.temporal import convert_timestamp
from stingray.utils.gridding import assign_time_bins
from stingray.utils.spatial import convert_gps
from stingray.sensors.ctd import density_ies80
from stingray.sensors.fluorometer import calibrate_chlorophyll, calibrate_backscatter
from stingray.sensors.par import calibrate_par
from stingray.sensors.qc import apply_qc
from stingray.sensors.suna import calibrate_nitrate
from stingray.profiles.identify import identify_profiles

logger = logging.getLogger(__name__)

SENSOR_NAMES = ["CTD", "DVL", "Fluorometer", "GPS", "Oxygen", "PAR", "SUNA"]
OUTPUT_FORMATS = ("csv", "parquet")

CTD_AGG = {
    "Temperature": "median",
    "Conductivity": "median",
    "Pressure": "median",
    "Salinity": "median",
    "Sound Velocity": "median",
    "density": "median",
    "Depth": "first",
}
GPS_AGG = {"Latitude": "first", "Longitude": "first"}
DVL_AGG = {
    "anglePitch": "median",
    "angleRoll": "median",
    "angleHeading": "median",
    "distanceAltitude": "median",
    "velocityWaterTransverse": "median",
    "velocityWaterLongitudinal": "median",
    "velocityWaterNormal": "median",
    "velocityGroundTransverse": "median",
    "velocityGroundLongitudinal": "median",
    "velocityGroundVertical": "median",
}
FLUORO_AGG = {"Chlorophyll": "median", "Backscattering": "median"}
OXYGEN_AGG = {"O2Concentration": "median", "AirSaturation": "median"}
PAR_AGG = {"Raw PAR [V]": "median"}
SUNA_AGG = {"NitrateConcentration[uM]": "median"}


def aggregate_bins(
    df: pd.DataFrame,
    spec: dict[str, str],
    count_name: str | None = None,
) -> pd.DataFrame:
    """
    Aggregate one sensor per time bin.

    A sensor with no files (or none of the requested columns) yields an empty
    float frame with the expected columns, so the later merge still works.
    """
    spec = {col: how for col, how in spec.items() if col in df.columns}
    columns = [*spec, *([count_name] if count_name else [])]
    if df.empty or "time_bin" not in df.columns or not spec:
        return pd.DataFrame({name: pd.Series(dtype="float64") for name in ["time_bin", *columns]})

    if not df["Timestamp"].is_monotonic_increasing:
        df = df.sort_values("Timestamp", kind="stable")
    grouped = df.groupby("time_bin", as_index=False)
    out = grouped.agg(spec)
    if count_name:
        out[count_name] = grouped.size()["size"].to_numpy()
    return out


def interpolate_gps_within_deployments(sled: pd.DataFrame, gps_cols: list[str]) -> int:
    """Linearly interpolate GPS gaps in time inside each deployment (in place)."""
    t = sled["timestamp"].to_numpy(np.float64)
    filled = 0
    for idx in sled.groupby("deployment").indices.values():
        for col in gps_cols:
            values = sled[col].to_numpy(np.float64)
            y = values[idx]
            valid = ~np.isnan(y)
            if valid.sum() < 2 or valid.all():
                continue
            interp = np.interp(t[idx], t[idx][valid], y[valid], left=np.nan, right=np.nan)
            new = ~valid & ~np.isnan(interp)
            if new.any():
                values = values.copy()
                values[idx[new]] = interp[new]
                sled[col] = values
                filled += int(new.sum())
    return filled


def merge_sensors(
    cruise: str,
    start: str,
    end: str,
    root: str | Path = "sensor_data",
    cal_year: str | None = None,
    time_bin_seconds: float = 5.0,
    out_dir: str | Path = "dash_data/data/stingray/",
    index_dir: str | Path = "indexes",
    overwrite_index: bool = False,
    suna_cal_file: str | Path | None = None,
    suna_cal_dir: str | Path | None = None,
    output_format: str = "csv",
    bin_counts: bool = False,
    dry_run: bool = False,
    qc: bool = True,
) -> Path | None:
    # Logging the function entry and parameters for better traceability.
    logger.info(
        "Merging sensors data | cruise=%s start=%s end=%s root=%s cal_year=%s "
        "time_bin_seconds=%s out_dir=%s index_dir=%s overwrite_index=%s "
        "suna_cal_file=%s suna_cal_dir=%s",
        cruise,
        start,
        end,
        root,
        cal_year,
        time_bin_seconds,
        out_dir,
        index_dir,
        overwrite_index,
        suna_cal_file,
        suna_cal_dir,
    )
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {OUTPUT_FORMATS}: {output_format}")

    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    root = Path(root)
    out_dir = Path(out_dir)
    index_dir = Path(index_dir)

    index_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # LOAD RAW SENSORS
    # -------------------------
    logger.info("Scanning raw sensor files...")
    sensors = {}

    for name in SENSOR_NAMES:
        idx = load_or_build_file_index(
            root / name,
            index_dir / f"{name}_index.csv",
            overwrite=overwrite_index,
        )

        files = filter_file_index(idx, start_date, end_date)
        if dry_run:
            logger.info("[dry-run] %s: %s files in range", name, len(files))
            continue
        df = read_csv_parallel(files)
        # First QC, per sensor stream and before any calibration.
        sensors[name.lower()] = apply_qc(df, name) if qc else df

        logger.info(
            "%s: %s files, %s rows",
            name,
            len(files),
            len(sensors[name.lower()]),
        )

    if dry_run:
        logger.info("[dry-run] Nothing read or written.")
        return None

    # -------------------------
    # DEFINE TIME GRID FROM CTD
    # -------------------------
    logger.info("Gridding all sensors to %s-second bins...", time_bin_seconds)
    ctd = sensors["ctd"]

    if ctd.empty:
        logger.error("[%s] CTD missing — skipping cruise.", cruise)
        return None

    t_ctd = ctd["Timestamp"].to_numpy(float)
    grid_start = np.floor(np.nanmin(t_ctd) / time_bin_seconds) * time_bin_seconds
    grid_end = np.ceil(np.nanmax(t_ctd) / time_bin_seconds) * time_bin_seconds

    # -------------------------
    # ATTACH time_bin TO ALL SENSORS
    # -------------------------
    for key, df in sensors.items():
        if df.empty:
            continue

        df["time_bin"] = assign_time_bins(
            np.asarray(df["Timestamp"], dtype=np.float64),
            time_bin_seconds,
            grid_start,
            grid_end,
        )

    # -------------------------
    # CTD BLOCK
    # -------------------------
    ctd["density"] = density_ies80(
        ctd["Salinity"].to_numpy(float),
        ctd["Temperature"].to_numpy(float),
        (ctd["Pressure"] / 10.0).to_numpy(float),
    )

    ctd_agg = aggregate_bins(ctd, CTD_AGG, "ctd_n" if bin_counts else None)

    logger.info("CTD bins: %s", len(ctd_agg))

    # -------------------------
    # GPS BLOCK
    # -------------------------
    gps = sensors["gps"]

    if not gps.empty:
        gps["Latitude"] = convert_gps(gps["Latitude"], gps["Latitude Hemisphere"])
        gps["Longitude"] = convert_gps(gps["Longitude"], gps["Longitude Hemisphere"])

    gps_agg = aggregate_bins(gps, GPS_AGG, "gps_n" if bin_counts else None)

    gps_agg = gps_agg.set_index("time_bin").reindex(ctd_agg["time_bin"]).reset_index()

    logger.info(
        "GPS bins raw: %s / %s",
        gps_agg[["Latitude", "Longitude"]].notna().all(axis=1).sum(),
        len(gps_agg),
    )

    # -------------------------
    # DVL BLOCK
    # -------------------------
    dvl = sensors["dvl"]
    dvl_agg = aggregate_bins(dvl, DVL_AGG, "dvl_n" if bin_counts else None)

    logger.info("DVL bins: %s", len(dvl_agg))

    # -------------------------
    # FLUOROMETER BLOCK
    # -------------------------
    fluoro = sensors["fluorometer"]

    if not fluoro.empty:
        fluoro["Chlorophyll"] = calibrate_chlorophyll(
            fluoro["Chlorophyll"],
            year=cal_year,
            date=start_date,
        )
        fluoro["Backscattering"] = calibrate_backscatter(
            fluoro["Backscattering"],
            year=cal_year,
            date=start_date,
        )

    fluoro_agg = aggregate_bins(fluoro, FLUORO_AGG, "fluorometer_n" if bin_counts else None)

    logger.info("Fluorometer bins: %s", len(fluoro_agg))

    # -------------------------
    # OXYGEN BLOCK
    # -------------------------
    oxygen = sensors["oxygen"]

    oxygen_agg = aggregate_bins(oxygen, OXYGEN_AGG, "oxygen_n" if bin_counts else None)

    logger.info("Oxygen bins: %s", len(oxygen_agg))

    # -------------------------
    # PAR BLOCK
    # -------------------------
    par = sensors["par"]

    if not par.empty:
        par["Raw PAR [V]"] = calibrate_par(
            par["Raw PAR [V]"],
            year=cal_year,
            date=start_date,
        )

    par_agg = aggregate_bins(par, PAR_AGG, "par_n" if bin_counts else None)

    logger.info("PAR bins: %s", len(par_agg))

    # -------------------------
    # SUNA BLOCK
    # -------------------------
    suna = sensors["suna"]

    if not suna.empty:
        if suna_cal_file is None and suna_cal_dir is None:
            # Fast path: no TSP correction.
            suna["NitrateConcentration[uM]"] = calibrate_nitrate(
                suna,
                cruise,
                cal_file=None,
                cal_dir=None,
            )

        else:
            # Slow path: TSP correction needs CTD fields and matdate.
            suna["Times"], suna["matdate"] = convert_timestamp(suna["Timestamp"])

            ctd_sorted = ctd.sort_values("Timestamp")
            t_ctd = ctd_sorted["Timestamp"].to_numpy(dtype=np.float64)
            t_suna = suna["Timestamp"].to_numpy(dtype=np.float64)

            for column in ("Salinity", "Temperature", "Pressure"):
                suna[column] = np.interp(
                    t_suna,
                    t_ctd,
                    ctd_sorted[column].to_numpy(dtype=np.float64),
                    left=np.nan,
                    right=np.nan,
                )

            suna["NitrateConcentration[uM]"] = calibrate_nitrate(
                suna,
                cruise,
                cal_file=suna_cal_file,
                cal_dir=suna_cal_dir,
            )

    suna_agg = aggregate_bins(suna, SUNA_AGG, "suna_n" if bin_counts else None)

    logger.info("SUNA bins: %s", len(suna_agg))

    # -------------------------
    # CAST / PROFILE SEGMENTATION
    # -------------------------
    logger.info("Identifying deployments and casts from CTD...")
    profile_settings = load_config("processing")["profiles"]
    cast, deployment = identify_profiles(
        depth=ctd_agg["Depth"].to_numpy(np.float64),
        time_seconds=ctd_agg["time_bin"].to_numpy(np.float64),
        bin_size=float(profile_settings["bin_size"]),
        smooth_window=int(profile_settings["smooth_window"]),
        dir_window=int(profile_settings["dir_window"]),
        max_gap_sec=float(profile_settings["max_gap_sec"]),
        min_samples=int(profile_settings["min_samples"]),
        surface_start_threshold=float(profile_settings["surface_start_threshold"]),
        min_turn_depth=float(profile_settings["min_turn_depth"]),
        min_turn_time=float(profile_settings["min_turn_time"]),
    )

    ctd_agg["cast"] = pd.Series(cast).astype("Int64")
    ctd_agg["deployment"] = pd.Series(deployment).astype("Int64")

    # -------------------------
    # MERGE MASTER TABLE
    # -------------------------
    sled = (
        ctd_agg
        .merge(gps_agg, on="time_bin", how="left")
        .merge(dvl_agg, on="time_bin", how="left")
        .merge(fluoro_agg, on="time_bin", how="left")
        .merge(par_agg, on="time_bin", how="left")
        .merge(oxygen_agg, on="time_bin", how="left")
        .merge(suna_agg, on="time_bin", how="left")
    )

    # -------------------------
    # POST-MERGE PROCESSING
    # -------------------------
    sled["timestamp"] = sled["time_bin"]
    sled["times"] = pd.to_datetime(convert_timestamp(sled["time_bin"])[0])
    sled = sled.drop(columns=["time_bin"], errors="ignore")
    sled = sled.sort_values("timestamp").reset_index(drop=True)

    sled.rename(columns=sled_columns(), inplace=True)

    # -------------------------
    # QC: GPS INTERPOLATION WITHIN DEPLOYMENTS
    # -------------------------
    gps_cols = ["latitude", "longitude"]

    if all(c in sled.columns for c in gps_cols) and "deployment" in sled.columns:
        filled = interpolate_gps_within_deployments(sled, gps_cols)
        logger.info("QC: %s GPS points interpolated within deployments.", filled)

    # -------------------------
    # SAVE OUTPUT
    # -------------------------
    out_path = out_dir / f"{start_date.strftime('%Y%m%d')}_{cruise}.{output_format}"
    if output_format == "parquet":
        sled.to_parquet(out_path, index=False)
    else:
        write_csv_atomic(sled, out_path)
    
    logger.info("Merge complete for cruise %s | %s to %s", cruise, start, end)
    logger.info("Bin size: %s seconds", time_bin_seconds)
    logger.info("Data points: %s", len(sled))
    logger.info("Number of deployments: %s", ctd_agg["deployment"].nunique())
    logger.info("Number of casts: %s", ctd_agg["cast"].nunique())
    logger.info("Output saved to: %s", out_path)

    return out_path
