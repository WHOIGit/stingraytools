from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stingray.io.csv import write_csv_atomic
from stingray.logging.setup import log_command_options, setup_logging
from stingray.utils.gridding import bin_to_sensor_grid, infer_bin_width
from stingray.utils.temporal import DEFAULT_ORIGIN_DATE, parse_datetimes

logger = logging.getLogger(__name__)

STREAM_NAME = re.compile(r"[^\s,/\\]+")


def add_media_to_merged(
    merged_csv: str | Path,
    media_stream: str,
    media_list: str | Path,
    out_path: str | Path | None = None,
    overwrite: bool = False,
    bin_width: float | None = None,
) -> Path:
    """
    Add one media stream's frame information to a merged sensor CSV.

    ``media_stream`` is only a label: it names the new columns
    ``media_<stream>`` and ``frame_<stream>``. ``media_list`` is the exact
    frame-list CSV. Each sensor time bin gets the first frame that falls in the
    bin, using the same binning rule as abundance. Other streams' columns are
    never touched, and no link columns are written.
    """
    started_at = time.monotonic()
    if not STREAM_NAME.fullmatch(str(media_stream)):
        raise ValueError(
            "media_stream must be non-empty and contain no whitespace, commas "
            f"or path separators: {media_stream!r}"
        )

    merged_csv = Path(merged_csv)
    media_path = Path(media_list)
    if not media_path.is_file():
        raise FileNotFoundError(f"Frame-list CSV not found: {media_path}")
    out_path = merged_csv if out_path is None else Path(out_path)
    media_column = f"media_{media_stream}"
    frame_column = f"frame_{media_stream}"

    logger.info(
        "Adding media to sensor data | sensor_csv=%s stream=%s media_list=%s "
        "out_path=%s overwrite=%s",
        merged_csv,
        media_stream,
        media_path,
        out_path,
        overwrite,
    )
    sled = pd.read_csv(merged_csv)
    logger.info("Loaded %s sensor rows with %s columns.", len(sled), len(sled.columns))

    # Frame viewers build links from media and frame; the sensor CSV carries none.
    legacy_columns = [
        column for column in sled.columns
        if re.fullmatch(r"(?:media_path|id|link)(?:_[1-9][0-9]*)?", column)
    ]
    if legacy_columns:
        sled = sled.drop(columns=legacy_columns)

    if "timestamp" not in sled.columns:
        raise ValueError("Merged sensor file must contain 'timestamp' column.")

    sled = sled.sort_values("timestamp").copy()

    sensor_times = np.sort(
        sled["timestamp"].dropna().to_numpy(dtype=np.float64)
    )

    if len(sensor_times) < 2:
        raise ValueError("Merged sensor file must contain at least two timestamps.")

    existing_cols = [c for c in (media_column, frame_column) if c in sled.columns]
    existing_with_data = [c for c in existing_cols if sled[c].notna().any()]
    if existing_with_data and not overwrite:
        raise ValueError(
            f"Media columns already contain data for stream {media_stream}: "
            f"{', '.join(existing_with_data)}. Use --overwrite to replace them."
        )

    # Frames go into the same sensor-timestamp bins that abundance counts them in.
    if bin_width is None:
        bin_width = infer_bin_width(sensor_times)
        logger.info("Inferred time-bin width from sensor timestamps: %s s", bin_width)
    grid_start = sensor_times[0]
    sensor_bin_index = np.rint((sensor_times - grid_start) / bin_width)

    media = pd.read_csv(
        media_path,
        usecols=lambda column: column in {"times", "media", "frame"},
    )
    input_rows = len(media)

    required_columns = {"times", "media", "frame"}
    missing_columns = sorted(required_columns.difference(media.columns))
    if missing_columns:
        raise ValueError(
            f"Frame-list CSV is missing {', '.join(missing_columns)}: {media_path}"
        )

    media["times"] = parse_datetimes(media["times"])
    media = media.dropna(subset=["times"]).copy()
    logger.info(
        "%s frame rows loaded; %s have valid timestamps.",
        input_rows,
        len(media),
    )
    if media.empty:
        raise ValueError(f"Frame-list CSV has no valid media times: {media_path}")

    media["timestamp"] = (media["times"] - DEFAULT_ORIGIN_DATE).dt.total_seconds()

    frame_bin = bin_to_sensor_grid(
        media["timestamp"].to_numpy(dtype=np.float64),
        sensor_times,
        bin_width,
    )
    frame_index = np.rint((frame_bin - grid_start) / bin_width)
    position = np.clip(
        np.searchsorted(sensor_bin_index, frame_index), 0, len(sensor_bin_index) - 1
    )
    in_bin = np.isfinite(frame_index) & (sensor_bin_index[position] == frame_index)
    logger.info(
        "%s: %s of %s frames fall in a sensor time bin.",
        media_stream,
        int(in_bin.sum()),
        len(media),
    )
    media = media.loc[in_bin].copy()
    if media.empty:
        raise ValueError(
            f"Frame-list CSV has no frames inside sensor time bins: {media_path}"
        )
    media["sensor_timestamp"] = sensor_times[position[in_bin]]

    # First frame of each bin is the representative one stored on the sensor row.
    media_agg = (
        media.sort_values(["times", "media", "frame"])
        .drop_duplicates("sensor_timestamp", keep="first")[
            ["sensor_timestamp", "media", "frame"]
        ]
        .rename(columns={"media": media_column, "frame": frame_column})
        .sort_values("sensor_timestamp")
        .reset_index(drop=True)
    )

    if existing_cols:
        logger.info("Replacing existing media columns for %s: %s", media_stream, existing_cols)
        sled = sled.drop(columns=existing_cols)

    sled = sled.merge(
        media_agg,
        left_on="timestamp",
        right_on="sensor_timestamp",
        how="left",
    ).drop(columns=["sensor_timestamp"])
    logger.info(
        "%s: frames stored on %s of %s sensor rows.",
        media_stream,
        len(media_agg),
        len(sled),
    )

    sled = sled.sort_values("timestamp")
    write_csv_atomic(sled, out_path)

    logger.info("Saved media-enriched file to: %s", out_path)
    logger.info(
        "Media enrichment complete | sensor_rows=%s stream=%s elapsed_seconds=%.2f",
        len(sled),
        media_stream,
        time.monotonic() - started_at,
    )

    return out_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="stingray images add-media",
        description=(
            "Add one media stream's first frame per time bin to a merged sensor "
            "CSV, as media_<stream> and frame_<stream> columns."
        ),
    )

    parser.add_argument(
        "merged_csv",
        help="Sensor CSV to add the media stream to.",
    )
    parser.add_argument(
        "--media-stream",
        required=True,
        help="Stream label used as the column suffix: media_<stream>, frame_<stream>.",
    )
    parser.add_argument(
        "--media-list",
        required=True,
        help="Exact frame-list CSV (columns times, media, frame) for this stream.",
    )
    parser.add_argument(
        "--out-path",
        default=None,
        help="Output CSV path. Default: update MERGED_CSV in place.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=None,
        help=(
            "Time-bin width in seconds. Default: inferred from the sensor CSV. "
            "Must match the width used for abundance."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace this stream's media/frame columns if they already hold data.",
    )
    parser.add_argument(
        "--work-dir",
        default=".",
        help="Workspace whose logs directory receives command logs.",
    )
    parser.add_argument(
        "--no-file-log",
        action="store_true",
        help="Disable Stingray log files and write logs only to the console.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO.",
    )

    args = parser.parse_args(argv)

    work_dir = Path(args.work_dir).expanduser().resolve()
    setup_logging(
        log_dir=work_dir / "logs",
        name="stingray_images_add_media",
        level=getattr(logging, args.log_level),
        file=not args.no_file_log,
    )
    log_command_options(logger, args)

    add_media_to_merged(
        merged_csv=args.merged_csv,
        media_stream=args.media_stream,
        media_list=args.media_list,
        out_path=args.out_path,
        overwrite=args.overwrite,
        bin_width=args.bin_width,
    )


if __name__ == "__main__":
    main()
