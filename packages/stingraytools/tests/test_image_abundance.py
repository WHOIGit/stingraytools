from datetime import datetime, timedelta

import pandas as pd

from stingray.images.abundance import Config, process


def test_abundance_uses_canonical_detection_table(tmp_path):
    start = datetime(2024, 1, 1, 0, 0, 0)

    sensor_csv = tmp_path / "sensor.csv"
    media_csv = tmp_path / "media.csv"
    detections_csv = tmp_path / "detections.csv"
    class_map_csv = tmp_path / "class_map.csv"
    out_csv = tmp_path / "abundance.csv"

    pd.DataFrame(
        {
            "times": [start, start + timedelta(seconds=5)],
            "temperature": [10.0, 10.1],
        }
    ).to_csv(sensor_csv, index=False)

    pd.DataFrame(
        {
            "times": [start, start + timedelta(seconds=5)],
            "media": ["media_a", "media_a"],
            "frame": [0, 1],
        }
    ).to_csv(media_csv, index=False)

    pd.DataFrame(
        {
            "media": ["media_a", "media_a", "media_a"],
            "frame": [0, 0, 1],
            "class_id": [0, 0, 1],
            "confidence": [0.9, 0.4, 0.95],
        }
    ).to_csv(detections_csv, index=False)

    pd.DataFrame(
        {
            "class_id": [0, 1],
            "class": ["copepod", "salp"],
            "source_file": ["classes.yaml", "classes.yaml"],
            "source_format": ["class_names_yaml", "class_names_yaml"],
        }
    ).to_csv(class_map_csv, index=False)

    result = process(
        Config(
            detections_csv=detections_csv,
            class_map_csv=class_map_csv,
            sensor_csv=sensor_csv,
            media_csv=media_csv,
            out_csv=out_csv,
            score_thresh=0.5,
            bin_width=5,
            volume_per_frame=0.5,
            add_ci=False,
        )
    )

    assert out_csv.exists()
    assert {"copepod", "salp", "total_abundance"}.issubset(result.columns)
    assert result["total_abundance"].fillna(0).sum() > 0


def test_abundance_can_overwrite_previous_product(tmp_path):
    start = datetime(2024, 1, 1, 0, 0, 0)
    sensor_csv = tmp_path / "sensor.csv"
    media_csv = tmp_path / "media.csv"
    detections_csv = tmp_path / "detections.csv"
    class_map_csv = tmp_path / "class_map.csv"

    pd.DataFrame(
        {
            "times": [start, start + timedelta(seconds=5)],
            "temperature": [10.0, 10.1],
        }
    ).to_csv(sensor_csv, index=False)
    pd.DataFrame(
        {
            "times": [start, start + timedelta(seconds=5)],
            "media": ["media_a", "media_a"],
            "frame": [0, 1],
        }
    ).to_csv(media_csv, index=False)
    pd.DataFrame(
        {
            "media": ["media_a", "media_a"],
            "frame": [0, 1],
            "class_id": [0, 1],
            "confidence": [0.9, 0.95],
        }
    ).to_csv(detections_csv, index=False)
    pd.DataFrame(
        {
            "class_id": [0, 1],
            "class": ["copepod", "salp"],
            "source_file": ["classes.yaml", "classes.yaml"],
            "source_format": ["class_names_yaml", "class_names_yaml"],
        }
    ).to_csv(class_map_csv, index=False)

    config = Config(
        detections_csv=detections_csv,
        class_map_csv=class_map_csv,
        sensor_csv=sensor_csv,
        media_csv=media_csv,
        out_csv=sensor_csv,
        score_thresh=0.5,
        bin_width=5,
        volume_per_frame=0.5,
        add_ci=False,
    )

    first = process(config)
    second = process(config)
    saved = pd.read_csv(sensor_csv)

    assert list(second.columns) == list(first.columns)
    assert not any(column.endswith(("_x", "_y")) for column in saved.columns)
    pd.testing.assert_series_equal(
        saved["total_abundance"],
        pd.Series(second["total_abundance"], name="total_abundance"),
        check_names=True,
    )
