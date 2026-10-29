from __future__ import annotations

import argparse
import logging

from stingray.config.columns import SLED_COLUMNS
from stingray.logging.setup import log_command_options


def test_log_command_options_logs_defaults_and_omits_token(caplog):
    """Parsed defaults should be reproducible without exposing API tokens."""
    args = argparse.Namespace(
        host="https://example.test",
        project_id=1,
        token="secret-token",
        work_dir=".",
    )

    with caplog.at_level(logging.INFO):
        log_command_options(logging.getLogger("test.command"), args)

    message = caplog.messages[-1]
    assert "host" in message
    assert "project_id" in message
    assert "work_dir" in message
    assert "token" not in message
    assert "secret-token" not in message


def test_log_command_options_survives_warning_log_level(caplog):
    """Reproducibility options should remain visible at restrictive log levels."""
    logger = logging.getLogger("test.warning-command")
    logger.setLevel(logging.WARNING)

    with caplog.at_level(logging.WARNING):
        log_command_options(logger, argparse.Namespace(log_level="WARNING"))

    assert "log_level" in caplog.messages[-1]


def test_dvl_velocity_columns_are_output_as_snake_case():
    """DVL velocity fields should be preserved in dashboard-ready output."""
    assert SLED_COLUMNS["velocityWaterTransverse"] == "velocity_water_transverse"
    assert SLED_COLUMNS["velocityWaterLongitudinal"] == "velocity_water_longitudinal"
    assert SLED_COLUMNS["velocityWaterNormal"] == "velocity_water_normal"
    assert SLED_COLUMNS["velocityGroundTransverse"] == "velocity_ground_transverse"
    assert SLED_COLUMNS["velocityGroundLongitudinal"] == "velocity_ground_longitudinal"
    assert SLED_COLUMNS["velocityGroundVertical"] == "velocity_ground_vertical"
