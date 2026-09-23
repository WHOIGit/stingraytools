#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stingray.utils.gridding import assign_time_bins
from stingray.utils.temporal import convert_timestamp
from stingray.stats.poisson import add_poisson_ci
from stingray.logging.setup import log_command_options, setup_logging

ORIGIN = datetime(1904, 1, 1)
logger = logging.getLogger(__name__)


def require_columns(df: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"{source} is missing required columns: {missing_str}")


@dataclass(frozen=True)
class Config:
    detections_csv: Path
    class_map_csv: Path
    sensor_csv: Path
    media_csv: Path
    out_csv: Path
    score_thresh: float
    bin_width: float
    volume_per_frame: float
    add_ci: bool


def process(config: Config) -> pd.DataFrame:
    """Convert image detections into time-binned abundance merged onto sensor data."""
    logger.info("Loading detection CSV...")
    df = pd.read_csv(config.detections_csv)
    require_columns(
        df,
        [
            "media",
            "frame",
            "class_id",
            "confidence",
        ],
        config.detections_csv,
    )

    logger.info("Loading class map CSV...")
    class_map_df = pd.read_csv(config.class_map_csv)
    require_columns(class_map_df, ["class_id", "class"], config.class_map_csv)
    class_map_df = class_map_df[["class_id", "class"]].drop_duplicates()
    if class_map_df["class_id"].duplicated().any():
        duplicated_ids = sorted(
            class_map_df.loc[class_map_df["class_id"].duplicated(), "class_id"]
            .astype(int)
            .unique()
        )
        raise ValueError(f"{config.class_map_csv} has duplicate class_id values: {duplicated_ids}")

    df["frame"] = df["frame"].astype(int)
    df["class_id"] = df["class_id"].astype(int)
    df["score"] = df["confidence"]
    df = df.merge(class_map_df, on="class_id", how="left")
    if df["class"].isna().any():
        missing_ids = sorted(df.loc[df["class"].isna(), "class_id"].unique())
        raise ValueError(f"Class IDs missing from class map CSV: {missing_ids}")

    df = (
        df.loc[df["score"] >= config.score_thresh]
        .sort_values(["media", "frame", "class"])
        .reset_index(drop=True)
    )

    df_count = (
        df.groupby(["media", "frame", "class"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    data_cols = df_count.columns.difference(
        ["media", "frame", "total_abundance"]
    ).to_list()

    logger.info("Loading sensor + media CSVs...")
    sensor_df = pd.read_csv(config.sensor_csv)
    media_df = pd.read_csv(config.media_csv)
    require_columns(sensor_df, ["times"], config.sensor_csv)
    require_columns(media_df, ["times", "media", "frame"], config.media_csv)

    abundance_df = media_df[["times", "media", "frame"]].merge(
        df_count,
        on=["media", "frame"],
        how="left",
    )
    abundance_df[data_cols] = abundance_df[data_cols].fillna(0)

    sensor_df["times"] = pd.to_datetime(sensor_df["times"], errors="coerce")
    abundance_df["times"] = pd.to_datetime(abundance_df["times"], errors="coerce")

    sensor_df = sensor_df.sort_values("times")
    abundance_df = abundance_df.sort_values("times")

    sensor_df["timestamp"] = (sensor_df["times"] - ORIGIN).dt.total_seconds()
    abundance_df["timestamp"] = (abundance_df["times"] - ORIGIN).dt.total_seconds()

    abundance_df["timestamp"] = assign_time_bins(
        abundance_df["timestamp"].to_numpy(),
        bin_width=config.bin_width,
        grid_start=sensor_df["timestamp"].min(),
        grid_end=sensor_df["timestamp"].max(),
    )
    abundance_df["times"] = pd.to_datetime(
        convert_timestamp(abundance_df["timestamp"])[0]
    )

    df_bin = abundance_df.groupby("times", as_index=False).agg(
        {
            "times": "first",
            "media": lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan,
            "frame": lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan,
            **{col: "mean" for col in data_cols},
        }
    )

    frame_num = abundance_df.groupby("times").size().rename("frame_num").reset_index()
    df_bin = df_bin.merge(frame_num, on="times", how="left")

    scale_factor = 1 / config.volume_per_frame
    df_bin[data_cols] = df_bin[data_cols] * scale_factor
    df_bin["total_abundance"] = df_bin[data_cols].sum(axis=1)

    if config.add_ci:
        logger.info("Computing Poisson confidence intervals...")
        raw_counts_by_bin = (
            abundance_df.groupby("times")[data_cols]
            .sum()
            .reset_index()
            .set_index("times")
            .reindex(df_bin["times"])
            .fillna(0)
            .astype(int)
        )
        df_bin = add_poisson_ci(
            df_bin,
            raw_counts_by_bin,
            data_cols,
            scale_factor,
        )

    inherited_media_columns = [
        column for column in sensor_df.columns
        if re.fullmatch(
            r"(?:media|frame|media_path|id|link)(?:_[1-9][0-9]*)?",
            column,
        )
    ]
    # Allow an enriched CSV to be used as the input on a later run. Any
    # columns produced by the new abundance table are replaced below instead
    # of being duplicated with pandas' _x/_y suffixes.
    replace_columns = [
        column
        for column in df_bin.columns
        if column != "times" and column in sensor_df.columns
    ]
    df_merged = (
        sensor_df.drop(
            columns=set(inherited_media_columns + replace_columns),
            errors="ignore",
        ).merge(
            df_bin,
            on="times",
            how="left",
        )
        .sort_values("times")
        .reset_index(drop=True)
    )

    logger.info("Writing output: %s", config.out_csv)
    config.out_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix=f".{config.out_csv.name}.",
            dir=config.out_csv.parent,
            delete=False,
            newline="",
            encoding="utf-8",
        ) as temp_file:
            temp_path = Path(temp_file.name)
        df_merged.to_csv(temp_path, index=False)
        os.replace(temp_path, config.out_csv)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    logger.info("Done")

    return df_merged


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="stingray images abundance",
        description="Convert image detections to time-binned abundance."
    )

    required = parser.add_argument_group("required file paths")
    required.add_argument(
        "--detections-csv",
        required=True,
        help="Detection table with media, frame, class_id, and confidence columns.",
    )
    required.add_argument(
        "--class-map-csv",
        required=True,
        help="Class map table with class_id and class columns.",
    )
    required.add_argument(
        "--sensor-csv",
        required=True,
        help="Sensor CSV to merge onto.",
    )
    required.add_argument(
        "--media-csv",
        required=True,
        help="Media/frame timestamp CSV.",
    )
    required.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path.",
    )

    options = parser.add_argument_group("processing options")
    options.add_argument(
        "--work-dir",
        default=".",
        help="Workspace whose logs directory receives command logs.",
    )
    options.add_argument("--score-thresh", type=float, required=True)
    options.add_argument(
        "--bin-width",
        type=float,
        required=True,
        help="Time-bin width in seconds",
    )
    options.add_argument("--volume-per-frame", type=float, required=True)
    options.add_argument("--add-ci", action="store_true", help="Add Poisson confidence intervals")
    options.add_argument(
        "--no-file-log",
        action="store_true",
        help="Disable Stingray log files and write logs only to the console.",
    )

    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        detections_csv=Path(os.path.expanduser(args.detections_csv)),
        class_map_csv=Path(os.path.expanduser(args.class_map_csv)),
        sensor_csv=Path(os.path.expanduser(args.sensor_csv)),
        media_csv=Path(os.path.expanduser(args.media_csv)),
        out_csv=Path(os.path.expanduser(args.out_csv)),
        score_thresh=args.score_thresh,
        bin_width=args.bin_width,
        volume_per_frame=args.volume_per_frame,
        add_ci=args.add_ci,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    work_dir = Path(args.work_dir).expanduser().resolve()
    setup_logging(
        log_dir=work_dir / "logs",
        name="stingray_images_abundance",
        file=not args.no_file_log,
    )
    log_command_options(logger, args)
    config = config_from_args(args)
    process(config)


if __name__ == "__main__":
    main()
