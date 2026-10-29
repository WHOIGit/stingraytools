from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


def build_file_index(dir_root: str | Path) -> list[tuple[datetime, str]]:
    """
    Scan a directory for CSV files and return a list of
    (datetime, filepath) pairs extracted from filenames.

    Expected filename pattern:
        <prefix>-YYYY-MM-DD HH-MM-SS.ffffff.csv
    """
    dir_root = Path(dir_root)
    index: list[tuple[datetime, str]] = []

    for file_path in sorted(dir_root.glob("*.csv")):
        try:
            basename = file_path.stem
            date_str = basename.split("-", 1)[1]
            file_date = datetime.strptime(date_str, "%Y-%m-%d %H-%M-%S.%f")
            index.append((file_date, str(file_path)))
        except Exception as exc:
            logger.info("Skipped %s: %s", file_path, exc)

    return index


def filter_file_index(
    file_index_df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime | None,
) -> list[str]:
    """
    Filter an index DataFrame for files within a date range.

    Parameters
    ----------
    file_index_df : pd.DataFrame
        Must contain 'datetime' and 'filepath' columns.
    start_date : datetime
        Inclusive lower bound.
    end_date : datetime | None
        Inclusive upper bound by day. If None, no upper bound is applied.

    Returns
    -------
    list[str]
        Filepaths matching the requested range.
    """
    required_cols = {"datetime", "filepath"}
    missing = required_cols.difference(file_index_df.columns)
    if missing:
        raise ValueError(f"file_index_df is missing required columns: {sorted(missing)}")

    end_bound = datetime.max if end_date is None else end_date + timedelta(days=1)
    mask = (file_index_df["datetime"] >= start_date) & (file_index_df["datetime"] < end_bound)

    return file_index_df.loc[mask, "filepath"].tolist()


def save_file_index(dir_root: str | Path, out_file: str | Path) -> pd.DataFrame:
    """
    Build a file index and save it to CSV.

    Returns
    -------
    pd.DataFrame
        Index dataframe with columns ['datetime', 'filepath'].
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    index = build_file_index(dir_root)
    df = pd.DataFrame(index, columns=["datetime", "filepath"]).sort_values("datetime")
    df.to_csv(out_file, index=False)

    return df


def load_or_build_file_index(
    dir_root: str | Path,
    out_file: str | Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Load an existing CSV index if available, or build a new one.

    Parameters
    ----------
    dir_root : str | Path
        Directory containing CSV files to index.
    out_file : str | Path
        Path to cached index CSV.
    overwrite : bool, default False
        If True, rebuild the index even if the cache exists.

    Returns
    -------
    pd.DataFrame
        Index dataframe with parsed datetime column.
    """
    out_file = Path(out_file)
    dir_root = Path(dir_root)

    # A directory's mtime changes when files are added or removed, so a newer
    # directory means the cached index is stale.
    stale = (
        out_file.exists()
        and dir_root.is_dir()
        and dir_root.stat().st_mtime > out_file.stat().st_mtime
    )
    if stale and not overwrite:
        logger.info("Index is older than %s; rebuilding: %s", dir_root, out_file)

    if out_file.exists() and not overwrite and not stale:
        df = pd.read_csv(out_file, parse_dates=["datetime"])
    else:
        if overwrite:
            logger.info("Rebuilding index (overwrite requested): %s", out_file)
        else:
            logger.info("No cached index found. Building new index: %s", out_file)

        df = save_file_index(dir_root, out_file)

    return df.sort_values("datetime").reset_index(drop=True)