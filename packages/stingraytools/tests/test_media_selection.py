from pathlib import Path

import pytest

from stingray.images import resolve_frame_list_csv


def test_resolve_frame_list_ignores_video_list(tmp_path):
    video_list = tmp_path / "20240101_EN001_video_list.csv"
    frame_list = tmp_path / "20240101_EN001_frame_list.csv"
    video_list.touch()
    frame_list.touch()

    assert resolve_frame_list_csv(tmp_path, "EN001") == frame_list


def test_resolve_frame_list_accepts_exact_csv(tmp_path):
    frame_list = tmp_path / "custom.csv"
    frame_list.touch()

    assert resolve_frame_list_csv(frame_list, "EN001") == frame_list


def test_resolve_frame_list_rejects_ambiguous_directory(tmp_path):
    (tmp_path / "20240101_EN001_frame_list.csv").touch()
    (tmp_path / "20240101_EN001_frame_list_details.csv").touch()

    with pytest.raises(ValueError, match="Pass the exact CSV path"):
        resolve_frame_list_csv(tmp_path, "EN001")
