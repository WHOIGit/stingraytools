# stingraytools

Sensor-processing, image-metadata, image abundance, CSV I/O, profile, and
statistical tools for NES-LTER Stingray data.

## Installation

Install the sensor-processing dependencies. This also supports image abundance,
which reuses the sensor gridding and Poisson confidence-interval tools:

```bash
 pip install "stingraytools[sensors] @ git+https://github.com/WHOIGit/stingraytools.git"
```

Install the image-metadata and training-data dependencies. Use this for frame
timestamp/media CSV generation and YOLO training-data preparation:

```bash
 pip install "stingraytools[images] @ git+https://github.com/WHOIGit/stingraytools.git"
```

Install the CTD compilation dependency set:

```bash
 pip install "stingraytools[ctd] @ git+https://github.com/WHOIGit/stingraytools.git"
```

Install the abundance dependency set:

```bash
 pip install "stingraytools[abundance] @ git+https://github.com/WHOIGit/stingraytools.git"
```

Install the complete Stingray CLI dependency set:

```bash
 pip install "stingraytools[pipeline] @ git+https://github.com/WHOIGit/stingraytools.git"
```

Confirm that the command-line interface is available:

```bash
stingray --help
stingray sensors --help
stingray images --help
```

Use a leaf command's help for its complete arguments, defaults, and
descriptions, for example `stingray images add-media --help`. The singular
aliases `stingray sensor` and `stingray image` are also accepted.

## Processing workspace

The tools operate on a data workspace containing cruise data:

```bash
mkdir stingray_workspace
cd stingray_workspace
```

The default workspace layout is:

```text
stingray_workspace/
  sensor_data/          raw CTD, DVL, fluorometer, GPS, oxygen, PAR, and SUNA folders
  media_list/           optional camera-stream frame metadata
  suna_calibration/     optional cruise-specific SUNA calibration files
  indexes/              generated sensor-file indexes
  logs/                 processing logs
  dash_data/data/       dashboard-ready output
```

The processing commands write dashboard-ready products, but do not include or
serve a visualization application. The separate
 [stingray-dashboard](https://github.com/WHOIGit/stingray-dashboard) repository
consumes these products.

## Example: shipboard processing

This workflow produces dashboard-ready CSV files in a shared workspace.

```bash
# Create a dedicated virtual environment on the shipboard or server Linux host.
python3 -m venv ~/venv/stingray
source ~/venv/stingray/bin/activate

# Install only the sensor-processing dependencies for this job.
 pip install "stingraytools[sensors] @ git+https://github.com/WHOIGit/stingraytools.git"

# Mount the network share using the method appropriate for the host system.
# This example assumes the share is available at /mnt/stingray_share.
cd /mnt/stingray_share

# Confirm the workspace contains the expected runtime inputs before processing.
# sensor_data/ contains the raw instrument folders.
ls sensor_data

# Process one cruise into dash_data/data/stingray/.
stingray sensors merge \
  --work-dir . \
  --cruise CRUISE_ID \
  --start START_DATE \
  --end END_DATE \
  --cal-year CALIBRATION_YEAR \
  --time-bin-seconds BIN_WIDTH_SECONDS

# After camera timestamps finish, enrich the sensor product with one or more
# ordered camera streams.
stingray images add-media \
  dash_data/data/stingray/DATE_CRUISE.csv \
  --work-dir . \
  --cruise CRUISE_ID \
  --media-list-dirs \
    media_list/CAMERA_STREAM_1/DATE_CRUISE_frame_list_fast.csv \
    media_list/CAMERA_STREAM_2/DATE_CRUISE_frame_list_fast.csv \
  --out-path /path/to/media_enriched/DATE_CRUISE.csv

# Compile CTD reference files into dash_data/data/ctd/ when needed.
stingray ctd download \
  --work-dir . \
  --skip-existing

```

## Example: WSL2 development

This workflow is useful when editing code or batch-editing CSV outputs from a
Windows-mounted drive. It keeps the source checkout editable while using the
same workspace layout as the shipboard deployment.

```bash
# Create and activate a development environment.
python -m venv .venv
source .venv/bin/activate

# Install the full Stingray CLI environment from the local checkout so code edits are live.
pip install -e ".[dev]"

# Enter the mounted data workspace, not the source repository.
cd "/mnt/c/path/to/stingray_workspace"

# Process or reprocess the cruise data into dash_data/data/stingray/.
stingray sensors merge \
  --work-dir . \
  --cruise CRUISE_ID \
  --start START_DATE \
  --end END_DATE \
  --time-bin-seconds BIN_WIDTH_SECONDS

# Optionally attach camera streams later without delaying sensor processing.
stingray images add-media \
  dash_data/data/stingray/DATE_CRUISE.csv \
  --work-dir . \
  --cruise CRUISE_ID \
  --media-list-dirs media_list/CAMERA_STREAM/DATE_CRUISE_frame_list_fast.csv \
  --out-path /path/to/media_enriched/DATE_CRUISE.csv

```

## Process one cruise

The following command processes one cruise. Both `--start` and `--end` are
inclusive calendar dates. Sensor observations are aggregated into bins of width
\(\Delta t\) seconds.

```bash
stingray sensors merge \
  --work-dir . \
  --cruise CRUISE_ID \
  --start START_DATE \
  --end END_DATE \
  --cal-year CALIBRATION_YEAR \
  --time-bin-seconds BIN_WIDTH_SECONDS
```

### Common options

```text
--cruise CRUISE
    Cruise ID.

--start START
    Inclusive cruise start date in YYYY-MM-DD format.

--end END
    Inclusive cruise end date in YYYY-MM-DD format.

--work-dir WORK_DIR
    Workspace containing runtime inputs and outputs. Default: current directory.

--root ROOT
    Raw sensor-data directory. Default: WORK_DIR/sensor_data.

--cal-year CAL_YEAR
    Sensor calibration year. Default: 2021.

--time-bin-seconds TIME_BIN_SECONDS
    Time-bin width Delta t in seconds. Default: 5.

--out-dir OUT_DIR
    Output directory. Default: WORK_DIR/dash_data/data/stingray.

--index-dir INDEX_DIR
    Generated sensor-file index directory. Default: WORK_DIR/indexes.

--suna-cal-file SUNA_CAL_FILE
    Optional SUNA calibration file for TSP-corrected nitrate.

--suna-cal-dir SUNA_CAL_DIR
    Optional directory containing cruise-specific SUNA calibration files.

--overwrite-index
    Rebuild cached sensor-file indexes.

--log-level {DEBUG,INFO,WARNING,ERROR}
    Logging level. Default: INFO.
```

## Add camera streams after sensor processing

Sensor aggregation does not wait for image or video processing. After one or
more camera-stream frame lists are ready, create a media-enriched dashboard CSV:

```bash
stingray images add-media \
  dash_data/data/stingray/DATE_CRUISE.csv \
  --work-dir . \
  --cruise CRUISE_ID \
  --media-list-dirs \
    media_list/CAMERA_STREAM_1/DATE_CRUISE_frame_list_fast.csv \
    media_list/CAMERA_STREAM_2/DATE_CRUISE_frame_list_fast.csv \
  --out-path /path/to/media_enriched/DATE_CRUISE.csv \
  --log-level INFO
```

The ordered frame-list inputs become `media`/`frame`, `media_2`/`frame_2`,
`media_3`/`frame_3`, and so forth. Existing `media_1`/`frame_1` columns are
recognized as aliases for the first stream, while newly written first-stream
columns use the unsuffixed form. Use `--overwrite` to replace populated media
columns deliberately.

Frame-list CSVs retain their complete metadata for other workflows. Media
enrichment reads only `times`, `media`, and `frame`, because those fields are
sufficient to build dashboard frame links. Legacy `id`, `link`, and
`media_path` columns are removed from the enriched output.

The command uses the same logging system as sensor processing. By default it
records resolved inputs, row and match counts, overwrite decisions, output
location, and elapsed time in `WORK_DIR/logs`. Use `--no-file-log` for console
logging only. Run `stingray images add-media --help` for the complete current
option list.

## Batch-process cruises

Batch processing is implemented as three independent runnable scripts in the
repository root under `flows/`. Each script is self-contained and can be
downloaded or copied by itself; it requires only an installed `stingraytools`
package and the runtime data workspace. Each script retrieves and filters the
authoritative NES-LTER cruise table and stops on the first failed cruise so
partial products are visible.

```bash
# Merge sensors for every cruise. Calibration is selected from cruise date,
# and SUNA calibration files are discovered under suna_calibration/ by cruise ID.
python flows/sensor_merge_flow.py \
  --work-dir . \
  --root sensor_data \
  --time-bin-seconds 5

# Build frame timestamp lists. The media path must contain {cruise}.
python flows/media_timestamp_flow.py \
  --work-dir . \
  --media-dir-template /path/to/media/{cruise} \
  --out-dir-template media_list/CAMERA_STREAM_1/{cruise}

# Attach one or more ordered camera streams to the merged sensor CSVs.
python flows/add_media_flow.py \
  --work-dir . \
  --media-list-dirs \
    media_list/CAMERA_STREAM_1/{cruise} \
    media_list/CAMERA_STREAM_2/{cruise} \
  --out-dir dash_data/data/media_enriched
```

Use `--cruise AR88 AR95 AR99 HRS2601` to restrict a flow to named cruises. Use
`--fallback-days` when the API does not provide an end date, and use
`--overwrite-index` or `--overwrite` where supported when rebuilding existing
products. See `python flows/<flow>.py --help` for the complete options.

After copying a script outside the repository, run it directly from the
workspace, for example `python sensor_merge_flow.py --help`. Install
`stingraytools` separately in the active environment.

## Sensor and Image Modules

The distribution includes these processing and metadata modules:

```text
stingray.sensors.ctd
stingray.sensors.fluorometer
stingray.sensors.par
stingray.sensors.suna
stingray.sensors.merge
stingray.images.build_frame_timestamps
stingray.images.abundance
stingray.images.generate_yolo_training
stingray.ctd.download
```

Import the relevant functions from Python scripts or notebooks when a workflow
needs finer control than the command-line interface provides.

`stingray images abundance` depends on the sensor/statistics stack, so install
`stingraytools[abundance]` for abundance-only batch jobs. `stingray images
frame-timestamp` and `stingray images generate-training` depend on the image
stack, so install `stingraytools[images]` for those jobs.

## Output and related tools

The default merged output is written below
`WORK_DIR/dash_data/data/stingray/`. These products can be visualized with the
 separately maintained [stingray-dashboard](https://github.com/WHOIGit/stingray-dashboard).
