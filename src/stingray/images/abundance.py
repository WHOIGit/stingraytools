#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stingray.utils.gridding import bin_to_sensor_grid
from stingray.utils.temporal import convert_timestamp
from stingray.io.csv import write_csv_atomic
from stingray.stats.poisson import add_poisson_ci
from stingray.utils.temporal import parse_datetimes
from stingray.logging.setup import log_command_options, setup_logging

ORIGIN = datetime(1904, 1, 1)
logger = logging.getLogger(__name__)

DETECTION_FORMATS = (".parquet", ".csv", ".json")
# COCO images name video frames <media>_<frame>.png with a 0-based frame.
COCO_FRAME_NAME = re.compile(r"(.+)_(\d+)\.png", re.IGNORECASE)
ISSUE_FIELDS = ["issue_type", "media", "frame", "source", "detail", "reason"]


def require_columns(df: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"{source} is missing required columns: {missing_str}")


@dataclass(frozen=True)
class Config:
    detections: Path
    sensor_csv: Path
    media_csv: Path
    out_csv: Path
    score_thresh: float
    bin_width: float
    volume_per_frame: float
    add_ci: bool
    class_map_csv: Path | None = None
    issues_csv: Path | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class Detections:
    """Detections in one model-independent form.

    ``boxes`` has media, frame (0-based source-video frame), class (name) and
    score (NaN for manual boxes). ``classes`` is the full class list when the
    source provides one. ``covered`` lists the frames that were looked at when
    the source covers only some frames (COCO); None means every frame of the
    frame list was processed.
    """

    boxes: pd.DataFrame
    classes: list[str]
    covered: pd.DataFrame | None


def add_issue(
    issues: list[dict[str, str]],
    issue_type: str,
    source: Path,
    detail: str,
    reason: str,
    media: str = "",
    frame: str = "",
) -> None:
    issues.append({
        "issue_type": issue_type, "media": media, "frame": frame,
        "source": str(source), "detail": detail, "reason": reason,
    })


def read_class_map(path: Path) -> dict[int, str]:
    """Read class_id -> class from a class map CSV."""
    class_map_df = pd.read_csv(path)
    require_columns(class_map_df, ["class_id", "class"], path)
    class_map_df = class_map_df[["class_id", "class"]].drop_duplicates()
    if class_map_df["class_id"].duplicated().any():
        duplicated_ids = sorted(
            class_map_df.loc[class_map_df["class_id"].duplicated(), "class_id"]
            .astype(int)
            .unique()
        )
        raise ValueError(f"{path} has duplicate class_id values: {duplicated_ids}")
    return dict(zip(class_map_df["class_id"].astype(int), class_map_df["class"].astype(str)))


def read_table_detections(
    path: Path,
    class_map_csv: Path | None,
    issues: list[dict[str, str]],
) -> Detections:
    """Read a detection table (Parquet or CSV) using only the needed columns."""
    classes: list[str] = []
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        schema = pq.read_schema(path)
        available = list(schema.names)
        metadata = schema.metadata or {}
        if b"classes" in metadata:
            class_list = json.loads(metadata[b"classes"])
            classes = [class_list[key] for key in sorted(class_list, key=int)]

        def read(columns: list[str]) -> pd.DataFrame:
            return pd.read_parquet(path, columns=columns)
    else:
        available = list(pd.read_csv(path, nrows=0).columns)

        def read(columns: list[str]) -> pd.DataFrame:
            return pd.read_csv(path, usecols=columns)

    missing = [column for column in ("media", "frame") if column not in available]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    if "class" in available:
        class_columns = ["class"]
    elif "class_id" in available:
        if class_map_csv is None:
            raise ValueError(f"{path} has only class_id; pass --class-map-csv to name the classes")
        class_columns = ["class_id"]
    else:
        raise ValueError(f"{path} is missing required columns: class (or class_id)")
    # Older tables call the model score "confidence"; no score means manual boxes.
    score_column = next((name for name in ("score", "confidence") if name in available), None)

    df = read(["media", "frame", *class_columns] + ([score_column] if score_column else []))
    if class_columns == ["class_id"]:
        logger.info("Loading class map CSV...")
        class_map = read_class_map(class_map_csv)
        if not classes:
            classes = [class_map[class_id] for class_id in sorted(class_map)]
        df["class"] = df["class_id"].astype(int).map(class_map)
        unknown = df["class"].isna()
        for class_id, count in df.loc[unknown, "class_id"].value_counts().sort_index().items():
            add_issue(issues, "unknown_class_id", path, f"class_id {class_id}: {count} row(s)",
                      "class ID missing from the class map; rows skipped")
        df = df.loc[~unknown].copy()

    df["score"] = pd.to_numeric(df[score_column], errors="coerce") if score_column else np.nan
    df["media"] = df["media"].astype(str)
    df["frame"] = df["frame"].astype(int)
    return Detections(df[["media", "frame", "class", "score"]], classes, None)


def read_coco_detections(path: Path, issues: list[dict[str, str]]) -> Detections:
    """Read standard COCO; images name video frames <media>_<frame>.png."""
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    categories = {category["id"]: str(category["name"]) for category in data.get("categories", [])}
    frames: dict = {}
    for image in data.get("images", []):
        name = re.split(r"[\\/]", str(image.get("file_name", "")))[-1]
        match = COCO_FRAME_NAME.fullmatch(name)
        if match is None:
            add_issue(issues, "invalid_image_name", path, name,
                      "file_name is not <media>_<frame>.png; image skipped")
            continue
        frames[image.get("id")] = (match.group(1), int(match.group(2)))

    rows = []
    for annotation in data.get("annotations", []):
        # Boxes the labeler marked deleted are kept only for its history.
        if annotation.get("deleted"):
            continue
        frame_key = frames.get(annotation.get("image_id"))
        if frame_key is None:
            continue
        class_name = categories.get(annotation.get("category_id"))
        if class_name is None:
            add_issue(issues, "unknown_category", path, f"category_id {annotation.get('category_id')}",
                      "category not in categories; box skipped", *map(str, frame_key))
            continue
        rows.append((*frame_key, class_name, annotation.get("score")))

    boxes = pd.DataFrame(rows, columns=["media", "frame", "class", "score"])
    boxes["score"] = pd.to_numeric(boxes["score"], errors="coerce")
    covered = pd.DataFrame(sorted(set(frames.values())), columns=["media", "frame"])
    classes = [categories[category_id] for category_id in sorted(categories)]
    return Detections(boxes, classes, covered)


def read_detections(config: Config, issues: list[dict[str, str]]) -> Detections:
    """Read detections from Parquet, CSV, or COCO JSON, chosen by extension."""
    suffix = config.detections.suffix.lower()
    if suffix not in DETECTION_FORMATS:
        raise ValueError(
            f"{config.detections}: detections must be one of {', '.join(DETECTION_FORMATS)}"
        )
    if suffix == ".json":
        return read_coco_detections(config.detections, issues)
    return read_table_detections(config.detections, config.class_map_csv, issues)


def report_unmatched_frames(
    frames: pd.DataFrame,
    frame_keys: pd.DataFrame,
    source: Path,
    issues: list[dict[str, str]],
) -> None:
    """List detection frames that have no time in the frame list."""
    unmatched = frames.merge(frame_keys, on=["media", "frame"], how="left", indicator=True)
    unmatched = unmatched.loc[unmatched["_merge"] == "left_only"]
    for media, group in unmatched.groupby("media"):
        add_issue(issues, "frame_not_in_frame_list", source, f"{len(group)} frame(s)",
                  "no time for these frames; left out of abundance", media=str(media))


def write_issues(path: Path, issues: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ISSUE_FIELDS)
        writer.writeheader()
        writer.writerows(issues)
    os.replace(temporary, path)


def process(config: Config) -> pd.DataFrame:
    """Convert image detections into time-binned abundance merged onto sensor data.

    Detections are read from a table (Parquet or CSV) with columns ``media``,
    ``frame``, ``class`` (or ``class_id`` plus a class map) and an optional
    ``score``, or from standard COCO JSON whose images are named
    ``<media>_<frame>.png``. ``frame`` is always the 0-based index into the
    source video, the same numbering as the frame list. Boxes without a score
    are manual and always counted; the score threshold applies to the rest.
    A COCO file covers only its listed images, so other frames are left out
    rather than counted as zero. Problem rows are skipped and reported.
    """
    issues: list[dict[str, str]] = []
    logger.info("Loading detections: %s", config.detections)
    detections = read_detections(config, issues)

    df = detections.boxes
    df = (
        df.loc[df["score"].isna() | (df["score"] >= config.score_thresh)]
        .sort_values(["media", "frame", "class"])
        .reset_index(drop=True)
    )

    df_count = (
        df.groupby(["media", "frame", "class"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    # Classes the source lists but that never passed the threshold stay as zeros.
    for class_name in detections.classes:
        if class_name not in df_count.columns:
            df_count[class_name] = 0
    data_cols = df_count.columns.difference(
        ["media", "frame", "total_abundance"]
    ).to_list()

    logger.info("Loading sensor + media CSVs...")
    sensor_df = pd.read_csv(config.sensor_csv)
    media_df = pd.read_csv(
        config.media_csv,
        usecols=lambda column: column in {"times", "media", "frame"},
    )
    require_columns(sensor_df, ["times"], config.sensor_csv)
    require_columns(media_df, ["times", "media", "frame"], config.media_csv)
    media_df = media_df[["times", "media", "frame"]].copy()
    media_df["media"] = media_df["media"].astype(str)
    frame_keys = media_df[["media", "frame"]].drop_duplicates()

    if detections.covered is not None:
        report_unmatched_frames(detections.covered, frame_keys, config.detections, issues)
        media_df = media_df.merge(detections.covered, on=["media", "frame"], how="inner")
    else:
        report_unmatched_frames(df_count[["media", "frame"]], frame_keys, config.detections, issues)

    abundance_df = media_df.merge(
        df_count,
        on=["media", "frame"],
        how="left",
    )
    abundance_df[data_cols] = abundance_df[data_cols].fillna(0)

    sensor_df["times"] = parse_datetimes(sensor_df["times"])
    abundance_df["times"] = parse_datetimes(abundance_df["times"])

    sensor_df = sensor_df.sort_values("times")
    abundance_df = abundance_df.sort_values("times")

    sensor_df["timestamp"] = (sensor_df["times"] - ORIGIN).dt.total_seconds()
    abundance_df["timestamp"] = (abundance_df["times"] - ORIGIN).dt.total_seconds()

    abundance_df["timestamp"] = bin_to_sensor_grid(
        abundance_df["timestamp"].to_numpy(),
        sensor_df["timestamp"].to_numpy(),
        config.bin_width,
    )
    abundance_df["times"] = convert_timestamp(abundance_df["timestamp"])[0]

    binned = abundance_df.groupby("times", sort=True)
    df_bin = binned.agg(
        frame_num=("media", "size"),
        **{col: (col, "mean") for col in data_cols},
    ).reset_index()
    df_bin = df_bin[["times", *data_cols, "frame_num"]]

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

    # Abundance owns only the columns it produces. The media and frame columns
    # written by add-media (including extra camera streams) are left untouched.
    replace_columns = [
        column
        for column in df_bin.columns
        if column != "times" and column in sensor_df.columns
    ]
    populated = [column for column in replace_columns if sensor_df[column].notna().any()]
    if populated and not config.overwrite:
        raise ValueError(
            f"Abundance columns already contain data in {config.sensor_csv}: "
            f"{', '.join(populated)}. Use --overwrite to replace them."
        )
    df_merged = (
        sensor_df.drop(columns=replace_columns).merge(
            df_bin,
            on="times",
            how="left",
        )
        .sort_values("times")
        .reset_index(drop=True)
    )

    logger.info("Writing output: %s", config.out_csv)
    write_csv_atomic(df_merged, config.out_csv)

    for issue_type in sorted({issue["issue_type"] for issue in issues}):
        count = sum(issue["issue_type"] == issue_type for issue in issues)
        logger.warning("Skipped input (%s): %d issue(s)", issue_type, count)
    if config.issues_csv is not None:
        write_issues(config.issues_csv, issues)
        logger.info("Issues written: %s", config.issues_csv)
    logger.info("Done")

    return df_merged


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="stingray images abundance",
        description="Convert image detections to time-binned abundance."
    )

    required = parser.add_argument_group("required file paths")
    required.add_argument(
        "--detections",
        required=True,
        help=(
            "Detections as Parquet or CSV (media, frame, class or class_id, optional score) "
            "or standard COCO JSON with images named <media>_<frame>.png. "
            "frame is the 0-based source-video frame."
        ),
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
        "--class-map-csv",
        help="Class map with class_id and class columns, for tables that carry only class_id.",
    )
    options.add_argument(
        "--issues-csv",
        help="Write skipped detections and frames to this CSV.",
    )
    options.add_argument(
        "--work-dir",
        default=".",
        help="Workspace whose logs directory receives command logs.",
    )
    options.add_argument(
        "--score-thresh",
        type=float,
        required=True,
        help="Minimum model score; boxes without a score (manual) are always counted.",
    )
    options.add_argument(
        "--bin-width",
        type=float,
        required=True,
        help="Time-bin width in seconds",
    )
    options.add_argument("--volume-per-frame", type=float, required=True)
    options.add_argument("--add-ci", action="store_true", help="Add Poisson confidence intervals")
    options.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace abundance columns already present in the sensor CSV.",
    )
    options.add_argument(
        "--no-file-log",
        action="store_true",
        help="Disable Stingray log files and write logs only to the console.",
    )

    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        detections=Path(os.path.expanduser(args.detections)),
        class_map_csv=Path(os.path.expanduser(args.class_map_csv)) if args.class_map_csv else None,
        issues_csv=Path(os.path.expanduser(args.issues_csv)) if args.issues_csv else None,
        sensor_csv=Path(os.path.expanduser(args.sensor_csv)),
        media_csv=Path(os.path.expanduser(args.media_csv)),
        out_csv=Path(os.path.expanduser(args.out_csv)),
        score_thresh=args.score_thresh,
        bin_width=args.bin_width,
        volume_per_frame=args.volume_per_frame,
        add_ci=args.add_ci,
        overwrite=args.overwrite,
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
