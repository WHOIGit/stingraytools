from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat
from stingray.utils.temporal import rawdate2date, rawtime2time

logger = logging.getLogger(__name__)


def parse_mat(mat_path: str | Path, name: str | None = None) -> dict[str, Any]:
    """
    Parse a MATLAB .mat struct into a Python dictionary.

    Parameters
    ----------
    mat_path : str | Path
        Path to the .mat file.
    name : str, optional
        Variable name inside the .mat file. If omitted, uses the file stem.

    Returns
    -------
    dict[str, Any]
        Parsed MATLAB struct fields as a Python dictionary.
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    if name is None:
        name = mat_path.stem

    mat = loadmat(mat_path)
    if name not in mat:
        valid_keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(
            f"Variable '{name}' not found in {mat_path}. "
            f"Available variables: {valid_keys}"
        )

    matfile = mat[name]
    if not hasattr(matfile, "dtype") or matfile.dtype.names is None:
        raise ValueError(f"Variable '{name}' in {mat_path} is not a MATLAB struct.")

    data: dict[str, Any] = {}
    for field_name in matfile.dtype.names:
        data[field_name] = matfile[field_name][0][0]

    return data


def ensure_suna_cal_lines(
    file_path: str | Path,
    initials: str = "",
    out_dir: str | Path = "modified_CAL",
) -> Path:
    """
    Ensure required metadata lines exist in a SUNA CAL file.
    If required lines are missing, write a modified CAL file to `out_dir`
    and return that new path. If the original file already contains the
    required lines, return the original file path.
    """
    file_path = Path(file_path)
    out_dir = Path(out_dir)
    out_path = out_dir / f"{file_path.stem}_{initials}_modified.CAL"

    if out_path.exists():
        logger.info("Modified CAL file already exists.")
        return out_path

    required_lines = """H,Pixel base,1,,,
H,Sensor Depth offset,0,,,
H,Br wavelength offset,210,,,
H,Min fit wavelength,,,,
H,Max fit wavelength,,,,
H,Use seawater dark current,No,,,
H,Pressure coef,0.0265,,,"""

    content = file_path.read_text()

    if required_lines in content:
        logger.info("Required lines are already present. Using original CAL file.")
        return file_path

    out_dir.mkdir(parents=True, exist_ok=True)
    insert_index = content.rfind("H,File creation time")
    if insert_index == -1:
        logger.warning(
            "Could not find insertion point in CAL file: %s. Using original file.",
            file_path,
        )
        return file_path

    content = content[:insert_index] + required_lines + "\n" + content[insert_index:]
    out_path.write_text(content)

    logger.info("Required lines not present in original file.")
    logger.info("Generated modified CAL file: %s", out_path)
    return out_path


def parse_suna_calibration(cal_file: str | Path) -> dict[str, Any]:
    """
    Parse a SUNA nitrate calibration (.CAL) file into a dictionary.
    """
    cal_file = Path(cal_file)

    cal: dict[str, Any] = {
        "type": "SUNA",
        "SN": "",
        "CalTemp": None,
        "CalSDN": None,
        "CalDateStr": "",
        "DC_flag": 1,
        "pixel_base": None,
        "depth_lag": None,
        "WL_offset": None,
        "min_fit_WL": None,
        "max_fit_WL": None,
        "pres_coef": None,
    }

    replacements = {
        "New ": "",
        "DI DC Corr": "Ref",
        "Reference": "Ref",
        "Wavelength": "WL",
        "WaveLen": "WL",
        ",NO3": ",ENO3",
        ",SWA": ",ESW",
        ",ASW,": ",ESW,",
        ",T*ASW": ",TSWA",
    }

    hdr: list[str] | None = None

    try:
        lines = cal_file.read_text().splitlines()
        data_lines: list[list[str]] = []

        for line in lines:
            tline = line.strip().split(",")
            if not tline or len(tline) < 2:
                continue

            if tline[0] == "H":
                field = tline[1]

                if field.startswith("SUNA"):
                    m = re.search(r"SUNA\s+(\d+)", field)
                    if m:
                        cal["SN"] = m.group(1)

                if field.startswith("Lamp#"):
                    cal["SN"] = " ".join(tline[1:])

                m = re.search(r"\d+/\d+/\d{4}", field)
                if m:
                    d_str = m.group(0)
                    cal["CalDateStr"] = d_str
                    cal["CalSDN"] = datetime.strptime(d_str, "%m/%d/%Y").date()

                if "creation time" in field:
                    m = re.search(r"\d{2}-\w{3}-\d{4}", field)
                    if m:
                        d_str = m.group(0)
                        cal["CalDateStr"] = d_str
                        cal["CalSDN"] = datetime.strptime(d_str, "%d-%b-%Y").date()

                if "CalTemp" in field:
                    cal["CalTemp"] = (
                        float(tline[2]) if len(tline) > 2 and tline[2] else None
                    )

                if re.search(r"T_CAL", field) and cal["CalTemp"] is None:
                    parts = field.split(" ")
                    if len(parts) > 1:
                        try:
                            cal["CalTemp"] = float(parts[1])
                        except ValueError:
                            pass

                if "Pixel base" in field:
                    cal["pixel_base"] = (
                        int(tline[2]) if len(tline) > 2 and tline[2] else None
                    )

                if "Sensor Depth" in field:
                    vals = tline[2:4]
                    cal["depth_lag"] = [float(x) if x else None for x in vals]

                if "Br wavelength" in field:
                    cal["WL_offset"] = (
                        float(tline[2]) if len(tline) > 2 and tline[2] else None
                    )

                if "Min fit" in field:
                    cal["min_fit_WL"] = (
                        float(tline[3]) if len(tline) > 3 and tline[3] else None
                    )

                if "Max fit" in field:
                    cal["max_fit_WL"] = (
                        float(tline[3]) if len(tline) > 3 and tline[3] else None
                    )

                if "Use seawater dark" in field:
                    cal["DC_flag"] = 0 if "yes" in field.lower() else 1

                if "Pressure coef" in field:
                    cal["pres_coef"] = (
                        float(tline[2]) if len(tline) > 2 and tline[2] else None
                    )

                if "Wavelength" in tline and "NO3" in tline:
                    hdr_line = ",".join(tline)
                    for old, new in replacements.items():
                        hdr_line = hdr_line.replace(old, new)
                    hdr = hdr_line.split(",")[1:]

            elif tline[0] == "E":
                data_lines.append(tline[1:])

        if len(data_lines) == 0:
            logger.info("File exists but no data lines found for: %s", cal_file)
            return cal

        if hdr is None:
            logger.warning(
                "No wavelength/data header found in calibration file: %s",
                cal_file,
            )
            return cal

        data = np.array(data_lines, dtype=float)
        if data.shape[1] != len(hdr):
            logger.warning(
                "Header/data width mismatch in %s: %s data columns vs %s header columns",
                cal_file,
                data.shape[1],
                len(hdr),
            )
            n = min(data.shape[1], len(hdr))
            data = data[:, :n]
            hdr = hdr[:n]

        for i, field_name in enumerate(hdr):
            cal[field_name] = data[:, i]

    except FileNotFoundError:
        logger.info("Cannot find calibration file at: %s", cal_file)
        return cal

    return cal


def find_suna_calibration_file(
    cruise: str,
    cal_dir: str | Path = "suna_calibration",
) -> Path | None:
    """
    Find the first SUNA calibration file whose filename contains the cruise id.
    """
    cal_dir = Path(cal_dir)
    if not cal_dir.exists():
        return None

    cruise_upper = cruise.upper()
    matches = sorted(
        file_path
        for file_path in cal_dir.iterdir()
        if file_path.is_file() and cruise_upper in file_path.name.upper()
    )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple SUNA calibration files match cruise {cruise}: {matches}"
        )
    if matches:
        return matches[0]

    return None


def load_suna_header(header_file: str | Path | None = None) -> list[str]:
    """
    Load the first line of the SUNA header reference file.

    If `header_file` is None, loads the bundled `data_reference/suna_hdr.txt`.
    """
    if header_file is None:
        header_file = Path(__file__).resolve().parents[1] / "data_reference" / "suna_hdr.txt"
    else:
        header_file = Path(header_file)

    return header_file.read_text().splitlines()[0].strip().split(",")


def read_suna_file(
    filename: str | Path,
    hdr: bool | list[str] | None = None,
    skiprows: int = 14,
) -> pd.DataFrame:
    """
    Read a raw SUNA data file.

    Parameters
    ----------
    filename : str or Path
        SUNA file path.
    hdr : bool, list[str], or None
        If True, use bundled SUNA header reference.
        If list, use that list as column names.
        If None or False, leave columns as integer labels.
    skiprows : int
        Number of metadata/header rows to skip before data.

    Returns
    -------
    pandas.DataFrame
        Parsed SUNA data.
    """
    filename = Path(filename)

    try:
        if hdr is True:
            hdr = load_suna_header()

        temp = pd.read_csv(filename, skiprows=skiprows, header=None)

        lhs = pd.DataFrame(
            {
                0: pd.Series([np.nan] * len(temp), dtype="float64"),
                1: rawdate2date(temp.iloc[:, 1]),
                2: rawtime2time(temp.iloc[:, 2]),
            }
        )

        rhs = temp.iloc[:, 3:-6].copy()
        df = pd.concat([lhs, rhs], axis=1)

        if isinstance(hdr, list):
            df.columns = hdr[:-8]

        return df

    except Exception as e:
        logger.info("Failed to read %s: %s", filename, e)
        return pd.DataFrame()
