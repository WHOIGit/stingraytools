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
  --cal-year CALIBRATION_YEAR \
  --time-bin-seconds BIN_WIDTH_SECONDS
```

Build camera frame timestamps:

```bash
stingray images frame-timestamp \
  --work-dir /path/to/stingray/data \
  --cruise CRUISE_ID \
  --media-dir /path/to/CAMERA_MEDIA_DIR \
  --out-dir /path/to/stingray/data/media_list/CAMERA_STREAM
```

Attach camera metadata to a cruise CSV:

```bash
stingray images add-media \
  /path/to/stingray/data/dashboard_data/data/DATASET/CRUISE.csv \
  --work-dir /path/to/stingray/data \
  --cruise CRUISE_ID \
  --media-list-dirs /path/to/stingray/data/media_list/CAMERA_STREAM
```

Download CTD reference data:

```bash
stingray ctd download \
  --work-dir /path/to/stingray/data \
  --skip-existing
```

Run the companion video and abundance workflow from its repository. Its
abundance step reads the sensor CSV and writes the updated product to the same
path.

Use command help for complete options:

```bash
stingray --help
stingray sensors --help
stingray images --help
```

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest packages/stingraytools/tests
```

## License and citation

StingrayTools is distributed under the MIT License. See [LICENSE](LICENSE).

Please cite [CITATION.cff](CITATION.cff).
