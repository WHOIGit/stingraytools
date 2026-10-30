"""Batch-flow wrapper for merging one or more cruises' sensor data."""

from __future__ import annotations

import argparse
from pathlib import Path

from stingray.sensors.cruises import load_cruises
from stingray.sensors.merge import merge_sensors



def workspace_path(work_dir, value):
    value = Path(value).expanduser()
    return value if value.is_absolute() else Path(work_dir) / value


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=".")
    parser.add_argument("--root", default="sensor_data")
    parser.add_argument("--out-dir", default="dash_data/data/stingray")
    parser.add_argument("--index-dir", default="indexes")
    parser.add_argument("--time-bin-seconds", type=float, default=5.0)
    parser.add_argument("--suna-cal-dir", default="suna_calibration")
    parser.add_argument("--overwrite-index", action="store_true")
    parser.add_argument("--cruise", nargs="+", help="Process only these cruises.")
    parser.add_argument("--fallback-days", type=int, default=7)
    args = parser.parse_args(argv)

    work_dir = Path(args.work_dir).expanduser().resolve()
    cruises = load_cruises("1900-01-01", args.fallback_days)
    if args.cruise:
        wanted = {name.upper() for name in args.cruise}
        available = {cruise["name"] for cruise in cruises}
        missing = sorted(wanted - available)
        if missing:
            parser.error(f"Cruise(s) not found: {', '.join(missing)}")
        cruises = [cruise for cruise in cruises if cruise["name"] in wanted]
    for cruise in cruises:
        message = f"Processing sensor merge: {cruise['name']} {cruise['start']} through {cruise['end']}"
        print(message)
        merge_sensors(
            cruise=cruise["name"],
            start=cruise["start"],
            end=cruise["end"],
            root=workspace_path(work_dir, args.root),
            time_bin_seconds=args.time_bin_seconds,
            out_dir=workspace_path(work_dir, args.out_dir),
            index_dir=workspace_path(work_dir, args.index_dir),
            overwrite_index=args.overwrite_index,
            suna_cal_dir=workspace_path(work_dir, args.suna_cal_dir),
        )


if __name__ == "__main__":
    main()
