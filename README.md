# StingrayTools

StingrayTools is the reusable processing package for Stingray sensor data,
camera metadata, image abundance, and CTD reference data.

## Responsibilities

- `sensors`: merge calibrated sensor observations into cruise CSV files.
- `images`: build frame timestamps, attach camera metadata, and compute image
  abundance.
- `ctd`: download and compile CTD reference data.
- shared utilities: CSV handling, time grids, profiles, and statistics.

Video inference and job orchestration are maintained in the companion
[stingray-image-analysis](https://github.com/WHOIGit/stingray-image-analysis)
repository. The Dash application is maintained in
[stingray-dashboard](https://github.com/WHOIGit/stingray-dashboard).

## Data contract

Dashboard-ready cruise files use one dataset directory per platform/project:

```text
dashboard_data/
  data/
    <platform_project>/
      <cruise>.csv
  media_list/
    <camera_stream>/
      <cruise>_frame_list_fast.csv
```

Sensor processing creates the cruise CSV. Image abundance updates that same
CSV with abundance columns; it does not create a separate dashboard dataset.

## Installation

Install only the required dependency group:

```bash
pip install "stingraytools[sensors] @ git+https://github.com/WHOIGit/stingraytools.git"
pip install "stingraytools[images] @ git+https://github.com/WHOIGit/stingraytools.git"
pip install "stingraytools[ctd] @ git+https://github.com/WHOIGit/stingraytools.git"
pip install "stingraytools[abundance] @ git+https://github.com/WHOIGit/stingraytools.git"
```

## Commands

Merge one cruise of sensor data:

```bash
stingray sensors merge \
  --work-dir /path/to/stingray/data \
  --cruise CRUISE_ID \
  --start START_DATE \
  --end END_DATE \
  --time-bin-seconds BIN_WIDTH_SECONDS
```

`--start`/`--end` may be omitted: dates are then looked up from the
NES-LTER API. `--cal-year` is optional (see Configuration). `--cruise` accepts several names or `all`. Other options:
`--dry-run` (list files in range only), `--format parquet`, `--bin-counts`
(per-sensor `<sensor>_n` columns), and `--no-file-log`. `stingray --version`
prints the installed version.

Build camera frame timestamps:

```bash
stingray images frame-timestamp \
  --work-dir /path/to/stingray/data \
  --cruise CRUISE_ID \
  --media-dir /path/to/CAMERA_MEDIA_DIR \
  --out-dir /path/to/stingray/data/media_list/CAMERA_STREAM
```

`frame-timestamp` caches probed media (path, size, modification time) under
`WORK_DIR/indexes`, so reruns only probe new or changed files. Use `--no-cache`
to disable it, `--verify-samples N` to probe more files per group (this also
ignores the cache), and `--header-count` to use container frame counts when a
group has to be measured file by file.

Add one camera stream's frame information to a cruise CSV. The stream name is
only a label and the frame list is an exact file; nothing is searched for:

```bash
stingray images add-media \
  /path/to/CRUISE.csv \
  --media-stream CAMERA_NAME \
  --media-list /path/to/frame_list.csv
```

Each sensor row stores the first frame of its time bin in `media_<stream>` and
`frame_<stream>` (no links, which the frame viewer builds). One stream per call;
other streams' columns are never touched. Re-running needs `--overwrite` once
those columns exist. `abundance` follows the same rule for its own columns and
never modifies the media/frame columns. All steps write the one cruise CSV atomically.

`add-media` and `abundance` place frames with the same binning rule: a bin starts at its sensor
timestamp and spans the bin width. `--bin-width` is optional for `add-media`
and is inferred from the sensor CSV when omitted.

Download CTD reference data:

```bash
stingray ctd download \
  --work-dir /path/to/stingray/data \
  --skip-existing
```

Concatenate Sea-Bird `.cnv` casts into one CSV with standardized `times`,
`latitude`, and `longitude`:

```bash
stingray ctd concat /path/to/cnv_dir --output /path/to/YYYYMMDD_CTD.csv
```

Run the companion video and abundance workflow from its repository. Its
abundance step reads the sensor CSV and writes the updated product to the same
path.

`stingray images abundance --detections` accepts detections from any model in
one of these forms:

- a Parquet or CSV table with `media`, `frame`, `class` (or `class_id` with
  `--class-map-csv`) and an optional `score`; or
- standard COCO JSON whose images are named `<media>_<frame>.png`.

`frame` is the 0-based source-video frame, matching the frame list. Boxes
without a score are manual and always counted. A COCO file counts only the
frames it lists. Skipped rows and frames are logged and, with `--issues-csv`,
written to a CSV.

Use command help for complete options:

```bash
stingray --help
stingray sensors --help
stingray images --help
```

## Configuration

Values that change with instruments or platforms live in YAML files bundled in
`src/stingray/config/defaults/`:

| File | Contents |
|---|---|
| `calibrations.yaml` | Dated calibration entries per sensor (chlorophyll, backscattering, PAR) |
| `units.yaml` | Display unit of each output column |
| `columns.yaml` | Raw/merged column name to output column name |
| `processing.yaml` | Cast/profile segmentation parameters |

A cruise uses the calibration entry in force on its start date, or the entry
named by `--cal-year`. To change values, copy a file into your own folder and
pass `--config-dir` (or set `STINGRAY_CONFIG_DIR`). An override only needs the
entries it changes; anything it leaves out falls back to the bundled default,
and a missing or unreadable file is reported and ignored. A calibration list
replaces the sensor's whole list. The dashboard keeps its own unit and
column-alias tables; keep them in step with `units.yaml` and `columns.yaml`.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
```

## License and citation

StingrayTools is distributed under the MIT License. See [LICENSE](LICENSE).

Please cite [CITATION.cff](CITATION.cff).
