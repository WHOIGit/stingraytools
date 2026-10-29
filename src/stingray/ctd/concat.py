"""Concatenate Sea-Bird CNV casts with standardized time and coordinates.

Run as ``stingray ctd concat``.

Priority: per-scan recognized data columns -> header metadata -> missing value.
Original sensor columns are preserved (duplicate names receive _2, _3, ...).
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter
from io import StringIO
from pathlib import Path
import re

import numpy as np
import pandas as pd

from stingray.logging.setup import log_command_options, setup_logging

logger = logging.getLogger(__name__)

DEFAULT_BAD = -9.99e-29
COLUMN_ALIASES = {
    "latitude": ("latitude", "lat", "nmealatitude", "gpslatitude", "latdec", "latdeg"),
    "longitude": ("longitude", "lon", "long", "lontitude", "nmealongitude", "gpslongitude", "londec", "londeg"),
}
TIME_ALIASES = {  # epoch / interpretation
    "timey": "unix", "timeq": "nmea", "times": "elapsed",
    "timej": "year_day",  # Sea-Bird Julian day within year, when year is in header
}


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def make_unique(names):
    counts = Counter()
    output = []
    for name in names:
        counts[name] += 1
        output.append(name if counts[name] == 1 else f"{name}_{counts[name]}")
    return output


def find_columns(frame, aliases):
    by_normalized = {}
    for name in frame.columns:
        by_normalized.setdefault(normalize(name), []).append(name)
    return [column for alias in aliases for column in by_normalized.get(normalize(alias), [])]


def header_value(header, labels):
    """Return first matching # or * key=value header field, ignoring capitalization."""
    wanted = {normalize(label) for label in labels}
    for line in header:
        match = re.match(r"^\s*[#*]\s*([^=]+?)\s*=\s*(.*?)\s*$", line)
        if match and normalize(match.group(1)) in wanted:
            return re.split(r"\s+\[", match.group(2), maxsplit=1)[0].strip()
    return None


def parse_coord(raw, axis):
    """Accept decimal degrees or degrees/decimal minutes with optional hemisphere."""
    if raw is None or pd.isna(raw):
        return np.nan
    raw = str(raw).strip().replace("°", " ").replace("'", " ").replace('"', " ")
    match = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*(?:[, :]\s*(\d+(?:\.\d+)?))?\s*([NSEW])?$", raw, re.I)
    if not match:
        return np.nan
    deg, mins, hemi = match.groups()
    value = float(deg)
    if mins is not None:
        minutes = float(mins)
        if minutes >= 60:
            return np.nan
        value = np.copysign(abs(value) + minutes / 60, value)
    if hemi:
        hemi = hemi.upper()
        if (axis == "latitude" and hemi not in "NS") or (axis == "longitude" and hemi not in "EW"):
            return np.nan
        value = (-1 if hemi in "SW" else 1) * abs(value)
    return value if abs(value) <= (90 if axis == "latitude" else 180) else np.nan


def numeric_clean(series, bad_value):
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return values.mask(np.isclose(values, bad_value, atol=1e-35, rtol=0))


def cast_start(header):
    raw = header_value(header, ("start_time", "NMEA UTC (Time)", "System UTC", "System UpLoad Time"))
    if not raw:
        return pd.NaT
    return pd.to_datetime(raw, errors="coerce", utc=True).tz_localize(None)


def standardize_time(df, header, bad_value):
    start = cast_start(header)
    result = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    # Prefer absolute per-scan times; elapsed only needs header start time.
    for alias, kind in TIME_ALIASES.items():
        for col in find_columns(df, (alias,)):
            sec = numeric_clean(df[col], bad_value)
            if kind == "unix":
                candidate = pd.to_datetime(sec, unit="s", errors="coerce", utc=True).dt.tz_localize(None)
            elif kind == "nmea":
                candidate = pd.to_datetime(sec, unit="s", origin="2000-01-01", errors="coerce")
            elif kind == "elapsed" and pd.notna(start):
                candidate = start + pd.to_timedelta(sec, unit="s", errors="coerce")
            elif kind == "year_day" and pd.notna(start):
                candidate = pd.Timestamp(year=start.year, month=1, day=1) + pd.to_timedelta(sec - 1, unit="D", errors="coerce")
            else:
                continue
            result = result.fillna(candidate)
    # When no per-scan time can be built, cast time is the best available metadata.
    return result.fillna(start) if pd.notna(start) else result


def standardize_coords(df, header, bad_value):
    for axis, aliases in COLUMN_ALIASES.items():
        meta = header_value(header, (f"NMEA {axis}", axis, "GPS " + axis))
        fallback = parse_coord(meta, axis)
        result = pd.Series(np.nan, index=df.index, dtype=float)
        for col in find_columns(df, aliases):
            # CNV numeric column latitude/longitude is normally decimal degrees.
            values = numeric_clean(df[col], bad_value)
            values = values.where(values.abs() <= (90 if axis == "latitude" else 180))
            result = result.fillna(values)
        df[axis] = result.fillna(fallback)


def read_cnv(path):
    lines = path.read_text(encoding="latin1", errors="replace").splitlines()
    end = next((i for i, line in enumerate(lines) if line.strip() == "*END*"), None)
    if end is None:
        raise ValueError("missing *END* header terminator")
    header = lines[:end]
    matches = [re.match(r"^\s*#\s*name\s+(\d+)\s*=\s*([^:]+):", line, re.I) for line in header]
    names = {int(m.group(1)): m.group(2).strip() for m in matches if m}
    if not names or set(names) != set(range(max(names) + 1)):
        raise ValueError("missing or non-contiguous '# name N =' definitions")
    columns = make_unique([names[i] for i in range(len(names))])
    body = "\n".join(lines[end + 1:]).strip()
    if not body:
        raise ValueError("no numeric data")
    df = pd.read_csv(StringIO(body), sep=r"\s+", header=None, names=columns, comment="#")
    bad_str = header_value(header, ("bad_flag",))
    try:
        bad_value = float(bad_str) if bad_str else DEFAULT_BAD
    except ValueError:
        bad_value = DEFAULT_BAD
    df["times"] = standardize_time(df, header, bad_value)
    standardize_coords(df, header, bad_value)
    return df


def combine(root, pattern, recursive=True, strict=False):
    paths = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise ValueError(f"No files match '{pattern}' in {root}")
    frames = []
    failures = 0
    for path in paths:
        try:
            df = read_cnv(path)
            # Cast identifier retains leading zeros and any suffix.
            cast = path.stem.split("_", 1)[1] if "_" in path.stem else path.stem
            df.insert(0, "cast", cast)
            df.insert(1, "filename", path.name)
            df.insert(2, "folder", path.parent.name)
            frames.append(df)
            logger.info(
                "%s: %s rows, %s times, %s lat, %s lon",
                path, len(df), df["times"].notna().sum(),
                df["latitude"].notna().sum(), df["longitude"].notna().sum(),
            )
        except Exception as exc:
            failures += 1
            logger.error("%s: %s", path, exc)
            if strict:
                raise
    if not frames:
        raise ValueError("No CNV files successfully read")
    output = pd.concat(frames, ignore_index=True, sort=False)
    # Stable sort retains file order when timestamps are unavailable/equal.
    output = output.sort_values(["cast", "times"], kind="stable", na_position="last").reset_index(drop=True)
    logger.info("Combined %s file(s), %s rows; %s failed", len(frames), len(output), failures)
    return output


def cli(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="stingray ctd concat",
        description="Concatenate Sea-Bird .cnv casts with standardized time, latitude and longitude",
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing CNV files")
    parser.add_argument("--pattern", default="*.cnv", help="Glob pattern (default: *.cnv)")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Output CSV path (default: <first-date>_CTD.csv in input directory)",
    )
    parser.add_argument("--no-recursive", action="store_true", help="Search only the input directory")
    parser.add_argument("--strict", action="store_true", help="Stop at the first invalid CNV file")
    parser.add_argument("--work-dir", default=".", help="Workspace containing relative paths and logs")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument(
        "--no-file-log", action="store_true",
        help="Disable Stingray log files and write logs only to the console.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = cli(argv)
    work_dir = Path(args.work_dir).expanduser().resolve()
    setup_logging(
        log_dir=work_dir / "logs",
        name="stingray_ctd_concat",
        level=getattr(logging, args.log_level),
        file=not args.no_file_log,
    )
    log_command_options(logger, args)

    root = Path(args.input_dir).expanduser()
    root = (root if root.is_absolute() else work_dir / root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")
    try:
        combined = combine(root, args.pattern, not args.no_recursive, args.strict)
    except ValueError as exc:
        raise SystemExit(str(exc))
    timestamps = combined["times"].dropna()
    date_tag = timestamps.min().strftime("%Y%m%d") if not timestamps.empty else "unknown_date"
    target = args.output.expanduser() if args.output else root / f"{date_tag}_CTD.csv"
    if not target.is_absolute():
        target = work_dir / target
    target.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(target, index=False)
    logger.info("Saved: %s", target)


if __name__ == "__main__":
    main()
