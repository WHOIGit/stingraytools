from __future__ import annotations

import argparse
import logging
from pathlib import Path

from stingray.config.loader import set_config_dir
from stingray.logging.setup import log_command_options, setup_logging
from stingray.sensors.cruises import load_cruises
from stingray.sensors.merge import OUTPUT_FORMATS, merge_sensors

logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="stingray sensors merge",
        description="Stingray CTD-binned sensor aggregation and cast identification"
    )

    parser.add_argument(
        "--cruise",
        nargs="+",
        required=True,
        help="Cruise name(s), or 'all'. Several cruises or 'all' look up dates from the NES-LTER API.",
    )
    parser.add_argument("--start", help="YYYY-MM-DD (single cruise; looked up when omitted)")
    parser.add_argument("--end", help="YYYY-MM-DD (single cruise; looked up when omitted)")
    parser.add_argument("--fallback-days", type=int, default=7, help="Cruise length when the API has no end time.")
    parser.add_argument(
        "--work-dir",
        default=".",
        help="Workspace containing runtime inputs, outputs, indexes, and logs.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Raw sensor-data root. Default: WORK_DIR/sensor_data.",
    )
    parser.add_argument(
        "--cal-year",
        default=None,
        help=(
            "Name of a calibration entry in calibrations.yaml. "
            "Default: the entry in force on the cruise start date."
        ),
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help=(
            "Folder with calibrations.yaml, units.yaml, columns.yaml or processing.yaml "
            "overriding the bundled defaults (also STINGRAY_CONFIG_DIR). "
            "Entries a file leaves out keep their defaults."
        ),
    )
    parser.add_argument("--time-bin-seconds", type=float, default=5.0)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: WORK_DIR/dash_data/data/stingray.",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Sensor index directory. Default: WORK_DIR/indexes.",
    )
    parser.add_argument(
        "--suna-cal-file",
        default=None,
        help="Optional SUNA calibration file.",
    )
    parser.add_argument(
        "--suna-cal-dir",
        default=None,
        help="Optional SUNA calibration directory, relative to WORK_DIR if needed.",
    )
    parser.add_argument("--overwrite-index", action="store_true")
    parser.add_argument(
        "--format",
        dest="output_format",
        default="csv",
        choices=list(OUTPUT_FORMATS),
        help="Output file format. The dashboard reads CSV.",
    )
    parser.add_argument(
        "--bin-counts",
        action="store_true",
        help="Add per-sensor sample-count columns (<sensor>_n) for each time bin.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the files in range for each cruise without reading or writing data.",
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
    )

    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    # Resolve all default runtime paths from one movable workspace.
    work_dir = Path(args.work_dir).expanduser().resolve()
    root = Path(args.root).expanduser() if args.root else Path("sensor_data")
    out_dir = (
        Path(args.out_dir).expanduser()
        if args.out_dir
        else Path("dash_data") / "data" / "stingray"
    )
    index_dir = Path(args.index_dir).expanduser() if args.index_dir else Path("indexes")
    suna_cal_file = Path(args.suna_cal_file).expanduser() if args.suna_cal_file else None
    suna_cal_dir = Path(args.suna_cal_dir).expanduser() if args.suna_cal_dir else None

    # Absolute paths remain unchanged; relative paths belong to the workspace.
    root = root if root.is_absolute() else work_dir / root
    out_dir = out_dir if out_dir.is_absolute() else work_dir / out_dir
    index_dir = index_dir if index_dir.is_absolute() else work_dir / index_dir
    if suna_cal_file is not None and not suna_cal_file.is_absolute():
        suna_cal_file = work_dir / suna_cal_file
    if suna_cal_dir is not None and not suna_cal_dir.is_absolute():
        suna_cal_dir = work_dir / suna_cal_dir

    setup_logging(
        log_dir=work_dir / "logs",
        name=__name__,
        level=getattr(logging, args.log_level),
        file=not args.no_file_log,
    )
    if args.config_dir:
        config_dir = Path(args.config_dir).expanduser()
        set_config_dir(config_dir if config_dir.is_absolute() else work_dir / config_dir)
    log_command_options(logger, args)

    names = [name.upper() for name in args.cruise]
    single = len(names) == 1 and names[0] != "ALL"
    if (args.start or args.end) and not (single and args.start and args.end):
        raise SystemExit("--start and --end must be given together, for a single cruise.")

    if single and args.start:
        cruises = [{
            "name": names[0],
            "start": args.start,
            "end": args.end,
        }]
    else:
        available = load_cruises("1900-01-01", args.fallback_days)
        if names != ["ALL"]:
            missing = sorted(set(names) - {cruise["name"] for cruise in available})
            if missing:
                raise SystemExit(f"Cruise(s) not found: {', '.join(missing)}")
            available = [cruise for cruise in available if cruise["name"] in set(names)]
        cruises = available

    for cruise in cruises:
        logger.info(
            "Processing sensor merge: %s %s through %s",
            cruise["name"], cruise["start"], cruise["end"],
        )
        try:
            merge_sensors(
                cruise=cruise["name"],
                start=cruise["start"],
                end=cruise["end"],
                root=root,
                cal_year=args.cal_year,
                time_bin_seconds=args.time_bin_seconds,
                out_dir=out_dir,
                index_dir=index_dir,
                overwrite_index=args.overwrite_index,
                suna_cal_file=suna_cal_file,
                suna_cal_dir=suna_cal_dir,
                output_format=args.output_format,
                bin_counts=args.bin_counts,
                dry_run=args.dry_run,
            )
        except Exception:
            if len(cruises) == 1:
                raise
            logger.exception("[%s] Merge failed; continuing with next cruise.", cruise["name"])


if __name__ == "__main__":
    main()
