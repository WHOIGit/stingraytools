"""Batch-flow wrapper for adding media streams to merged sensor CSVs, one cruise at a time."""

from __future__ import annotations

import argparse
from pathlib import Path

from stingray.images.add_media import add_media_to_merged
from stingray.sensors.cruises import load_cruises


def workspace_path(work_dir, value):
    value = Path(value).expanduser()
    return value if value.is_absolute() else Path(work_dir) / value


def parse_stream(value):
    stream, separator, path = value.partition("=")
    if not separator or not stream or not path:
        raise argparse.ArgumentTypeError(f"Expected STREAM=PATH, got: {value}")
    return stream, path


def find_merged_csv(merged_dir: Path, cruise: str) -> Path:
    matches = sorted(merged_dir.glob(f"*_{cruise}.csv"))
    if not matches:
        raise FileNotFoundError(f"No merged sensor CSV found for {cruise} in {merged_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple merged sensor CSVs found for {cruise}: {matches}")
    return matches[0]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=".")
    parser.add_argument("--merged-dir", default="dash_data/data/stingray")
    parser.add_argument("--out-dir", default="dash_data/data/media_enriched")
    parser.add_argument(
        "--media",
        nargs="+",
        required=True,
        type=parse_stream,
        metavar="STREAM=PATH",
        help="Stream label and its exact frame-list CSV. {cruise} in PATH is replaced by the cruise name.",
    )
    parser.add_argument("--bin-width", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
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
    merged_dir = workspace_path(work_dir, args.merged_dir)
    out_dir = workspace_path(work_dir, args.out_dir)
    for cruise in cruises:
        merged_csv = find_merged_csv(merged_dir, cruise["name"])
        out_path = out_dir / merged_csv.name
        print(f"Adding media to {cruise['name']}: {merged_csv.name} -> {out_path}")
        source = merged_csv
        # Each stream updates the same output file in turn.
        for stream, path in args.media:
            add_media_to_merged(
                merged_csv=source,
                media_stream=stream,
                media_list=workspace_path(work_dir, path.format(cruise=cruise["name"])),
                out_path=out_path,
                overwrite=args.overwrite,
                bin_width=args.bin_width,
            )
            source = out_path


if __name__ == "__main__":
    main()
