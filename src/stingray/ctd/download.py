#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import pandas as pd

from stingray.logging.setup import log_command_options, setup_logging

API_URL = "https://nes-lter-api.whoi.edu"
DEFAULT_MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)
REQUEST_TIMEOUT_SEC = 60
REQUEST_RETRIES = 3
logger = logging.getLogger(__name__)


def fetch_bytes(url: str, retries: int = REQUEST_RETRIES, timeout: float = REQUEST_TIMEOUT_SEC) -> bytes:
    """GET a URL with a timeout and exponential-backoff retries."""
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 ** (attempt - 1))


def read_csv_url(url: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(BytesIO(fetch_bytes(url)), **kwargs)


def cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="stingray ctd download",
        description=(
            "Download NES-LTER CTD cruise data, merge missing lat/lon/date "
            "from metadata when needed, and save one CSV per cruise."
        )
    )
    parser.add_argument(
        "--out-dir",
        default="dash_data/data/ctd",
        help="Output directory for cruise CSVs",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Thread pool size for cast downloads",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cruise if any *_CRUISE.csv already exists",
    )
    parser.add_argument(
        "--only-cruise",
        nargs="*",
        default=None,
        help="Optional list of cruise names to process",
    )
    parser.add_argument(
        "--work-dir",
        default=".",
        help="Workspace containing output and logs",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--no-file-log",
        action="store_true",
        help="Disable Stingray log files and write logs only to the console.",
    )
    return parser.parse_args(argv)


def cruise_file_exists(out_dir: Path, cruise: str) -> bool:
    return any(out_dir.glob(f"*_{cruise}.csv"))


def get_cast_list(cruise: str) -> tuple[pd.Series, pd.DataFrame | None]:
    try:
        metadata = read_csv_url(
            f"{API_URL}/api/ctd/metadata/{cruise}",
            usecols=["cast", "latitude", "longitude", "date"],
        )
        cast_ids = metadata["cast"].dropna().astype(str).unique()
        return cast_ids, metadata
    except Exception:
        cast_list = pd.read_json(BytesIO(fetch_bytes(f"{API_URL}/api/ctd/casts/{cruise}")))
        cast_ids = cast_list["number"].dropna().astype(str).unique()
        return cast_ids, None


def load_full_cruise_fast(
    cruise: str,
    max_workers: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    cast_ids, metadata = get_cast_list(cruise)
    base = f"{API_URL}/api/ctd/cast/{cruise}/"
    failed_casts = []

    def fetch(cast_id: str) -> pd.DataFrame | None:
        try:
            return read_csv_url(base + str(cast_id))
        except Exception as exc:
            failed_casts.append((cast_id, str(exc)))
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        dfs = list(executor.map(fetch, cast_ids))

    dfs = [df for df in dfs if df is not None]
    if not dfs:
        raise RuntimeError(f"All casts failed for {cruise}")

    df = pd.concat(dfs, ignore_index=True)
    if metadata is not None:
        meta = metadata.copy()
        meta["cast"] = meta["cast"].astype(str)
        if "cast" in df.columns:
            df["cast"] = df["cast"].astype(str)
            merge_cols = []
            for col in ["latitude", "longitude", "date"]:
                if col not in df.columns or df[col].isna().all():
                    merge_cols.append(col)
            if merge_cols:
                df = df.merge(
                    meta[["cast"] + merge_cols],
                    on="cast",
                    how="left",
                    suffixes=("", "_meta"),
                )
                for col in merge_cols:
                    meta_col = f"{col}_meta"
                    if col in df.columns and meta_col in df.columns:
                        df[col] = df[col].where(df[col].notna(), df[meta_col])
                        df.drop(columns=[meta_col], inplace=True)
                    elif meta_col in df.columns:
                        df.rename(columns={meta_col: col}, inplace=True)

    if failed_casts:
        logger.warning("[%s] %s casts failed", cruise, len(failed_casts))
        for cast_id, err in failed_casts[:10]:
            logger.warning("[%s] cast %s: %s", cruise, cast_id, err)

    return df


def get_date_from_df(df: pd.DataFrame) -> str:
    if "date" not in df.columns:
        return "nodate"
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.notna().any():
        return dates.min().strftime("%Y%m%d")
    return "nodate"


def main(argv: list[str] | None = None) -> None:
    args = cli(argv)
    work_dir = Path(args.work_dir).expanduser().resolve()
    setup_logging(
        log_dir=work_dir / "logs",
        name="stingray_ctd_download",
        level=getattr(logging, args.log_level),
        file=not args.no_file_log,
    )
    log_command_options(logger, args)

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = work_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading cruise list...")
    cruise_list = read_csv_url(f"{API_URL}/api/ctd/cruises/all")
    if args.only_cruise:
        wanted = set(args.only_cruise)
        cruise_list = cruise_list[cruise_list["name"].isin(wanted)].copy()
        logger.info("Restricted to %s cruises: %s", len(cruise_list), sorted(wanted))

    total = len(cruise_list)
    logger.info("Total cruises to consider: %s", total)
    for i, cruise in enumerate(cruise_list["name"], start=1):
        if args.skip_existing and cruise_file_exists(out_dir, cruise):
            logger.info("[%s/%s] Skipping %s (file already exists)", i, total, cruise)
            continue
        try:
            logger.info("[%s/%s] Processing %s...", i, total, cruise)
            df = load_full_cruise_fast(
                cruise,
                max_workers=args.max_workers,
                logger=logger,
            )
            date_str = get_date_from_df(df)
            out_path = out_dir / f"{date_str}_{cruise}.csv"
            df.to_csv(out_path, index=False)
            logger.info("[%s] Saved: %s", cruise, out_path)
            logger.info("[%s] Rows: %s", cruise, len(df))
        except Exception as exc:
            logger.error("[%s] Failed: %s", cruise, exc)
    logger.info("Done.")


if __name__ == "__main__":
    main()
