# StingrayTools

StingrayTools contains processing workflows for the NES-LTER Stingray tow sled.
The workflows can run independently or as one data pipeline from raw
sensor files, image metadata, ML detections, and CTD reference data to
dashboard-ready CSV products.

## Packages

- [stingraytools](packages/stingraytools/README.md): sensor processing, image
  metadata, image abundance, CTD compilation, shared time/grid utilities, and
  command-line workflows.

## Data Workflow

The main processing path is:

```text
raw Stingray sensor files
  -> stingray sensors merge
  -> dashboard_data/data/SENSOR_DATASET/

raw image or video files
  -> stingray images frame-timestamp
  -> media_list/CAMERA_STREAM/

sensor CSV + one or more camera-stream frame lists
  -> stingray images add-media
  -> media-enriched dashboard CSV

ML detection label files
  -> stingray-image-analysis/merge_detection_labels.sh
  -> stingray images abundance
  -> dashboard_data/data/shadowgraph/

NES-LTER CTD API data
  -> stingray ctd download
  -> dashboard_data/data/ctd/
```

ML inference and post-inference processing are orchestrated by the separate
[stingray-image-analysis](https://github.com/WHOIGit/stingray-image-analysis)
workflow repository.
This repository provides the reusable timestamp and abundance commands used by
that workflow.

## Installation

Install only the dependency set needed by the job:

```bash
pip install "stingraytools[sensors] @ git+https://github.com/WHOIGit/stingraytools.git"
pip install "stingraytools[images] @ git+https://github.com/WHOIGit/stingraytools.git"
pip install "stingraytools[ctd] @ git+https://github.com/WHOIGit/stingraytools.git"
pip install "stingraytools[abundance] @ git+https://github.com/WHOIGit/stingraytools.git"
```

Install the full processing pipeline dependency set:

```bash
pip install "stingraytools[pipeline] @ git+https://github.com/WHOIGit/stingraytools.git"
```

The dashboard application is maintained in the separate
[stingray-dashboard](https://github.com/WHOIGit/stingray-dashboard) repository.

## Core Commands

Merge one cruise of Stingray sensor data:

```bash
stingray sensors merge \
  --work-dir /path/to/stingray/data \
  --cruise CRUISE_ID \
  --start START_DATE \
  --end END_DATE \
  --cal-year CALIBRATION_YEAR \
  --time-bin-seconds BIN_WIDTH_SECONDS
```

Build image/video frame timestamps:

```bash
stingray images frame-timestamp \
  --work-dir /path/to/stingray/data \
  --cruise CRUISE_ID \
  --media-dir /path/to/CAMERA_MEDIA_DIR \
  --out-dir /path/to/stingray/data/media_list/CAMERA_STREAM
```

Attach one or more camera streams after the sensor CSV is available:

```bash
stingray images add-media \
  /path/to/stingray/data/dash_data/data/stingray/DATE_CRUISE.csv \
  --work-dir /path/to/stingray/data \
  --cruise CRUISE_ID \
  --media-list-dirs \
    /path/to/stingray/data/media_list/CAMERA_STREAM_1/DATE_CRUISE_frame_list_fast.csv \
    /path/to/stingray/data/media_list/CAMERA_STREAM_2/DATE_CRUISE_frame_list_fast.csv \
  --out-path /path/to/media_enriched/DATE_CRUISE.csv
```

Sensor processing and camera timestamp generation are intentionally independent.
The sensor product can therefore update near real time, while `add-media` creates
an enriched product after slower camera processing finishes.

## Command Help

List command groups and drill down to the complete options for one command:

```bash
stingray --help
stingray sensors --help
stingray images --help
stingray images add-media --help
```

The singular aliases `stingray sensor` and `stingray image` are also accepted.

Download CTD reference files:

```bash
stingray ctd download \
  --work-dir /path/to/stingray/data \
  --skip-existing
```

Run post-inference image abundance processing:

```bash
# Clone and enter the companion workflow repository.
git clone https://github.com/WHOIGit/stingray-image-analysis.git
cd stingray-image-analysis

# Copy and edit one cruise configuration before submitting jobs.
cp configs/cruise.example.conf.sh configs/my_cruise.conf.sh

# Create the log directory before Slurm opens the job log files.
mkdir -p slogs

# Submit each required stage in workflow order after its predecessor finishes.
sbatch frame_timestamps.sbatch configs/my_cruise.conf.sh
sbatch yolo_predict.sbatch configs/my_cruise.conf.sh
sbatch image_abundance.sbatch configs/my_cruise.conf.sh
```

Workflow runner details are in the
[stingray-image-analysis repository](https://github.com/WHOIGit/stingray-image-analysis).

## Development

Install the development dependency set from a local checkout:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run package checks:

```bash
python -m pytest packages/stingraytools/tests
```

## License

StingrayTools is distributed under the MIT License. See [LICENSE](LICENSE).

## Citation

Please cite this software as:

> Pham, Anh H. *StingrayTools*, version 3.1.0. MIT License. https://github.com/WHOIGit/stingraytools

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).
