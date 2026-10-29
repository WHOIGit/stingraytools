from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stingray.io.suna import (
    ensure_suna_cal_lines,
    find_suna_calibration_file,
    load_suna_header,
    parse_mat,
    parse_suna_calibration,
)

logger = logging.getLogger(__name__)

def tsp_correction(
    spec: dict[str, Any],
    ncal: dict[str, Any],
    pcor_flag: int,
) -> dict[str, Any]:
    """
    Apply SUNA TSP correction algorithm.
    """
    wl_offset = ncal["WL_offset"]
    d = spec.copy()

    d["P"] = d["STP"][2]
    d["T"] = d["STP"][1]
    d["S"] = d["STP"][0]
    del d["STP"]

    if ncal["pixel_base"] == 0:
        d["spectra_pix_range"] += 1
        d["pix_fit_win"] += 1
        logger.info("Pixel registration offset by +1")

    dc = d["DC"] if ncal["DC_flag"] != 0 else d["SWDC"]

    if ncal["min_fit_WL"]:
        d["pix_fit_win"][0] = np.where(ncal["WL"] >= ncal["min_fit_WL"])[0][0]
        d["WL_fit_win"][0] = ncal["WL"][d["pix_fit_win"][0]]

    if ncal["max_fit_WL"]:
        d["pix_fit_win"][1] = np.where(ncal["WL"] <= ncal["max_fit_WL"])[0][-1]
        d["WL_fit_win"][1] = ncal["WL"][d["pix_fit_win"][1]]

    if d["spectra_WL_range"][0][0] > d["WL_fit_win"][0][0]:
        logger.info(
            "Min WL of returned spectra (%s) is greater than Min WL of fit window (%s).",
            d["spectra_WL_range"][0][0],
            d["WL_fit_win"][0][0],
        )
        d["WL_fit_win"][0][0] = d["spectra_WL_range"][0][0]
        logger.info(
            "Fit window adjusted [%s %s] and NO3 estimate may be compromised.",
            d["WL_fit_win"][0][0],
            d["WL_fit_win"][0][1],
        )

    if d["spectra_WL_range"][0][1] < d["WL_fit_win"][0][1]:
        logger.info(
            "Max WL of returned spectra (%s) is less than Max WL of fit window (%s).",
            d["spectra_WL_range"][0][1],
            d["WL_fit_win"][0][1],
        )
        d["WL_fit_win"][0][1] = d["spectra_WL_range"][0][1]
        logger.info(
            "Fit window adjusted [%s %s] and NO3 estimate may be compromised.",
            d["WL_fit_win"][0][0],
            d["WL_fit_win"][0][1],
        )

    if np.isnan(np.sum(d["pix_fit_win"])) and "WL_fit_win" in d:
        pfit_low = np.where(ncal["WL"] >= d["WL_fit_win"][0][0])[0][0]
        pfit_hi = np.where(ncal["WL"] <= d["WL_fit_win"][0][1])[0][-1]
        d["pix_fit_win"] = [pfit_low, pfit_hi]

    ind1 = np.where(ncal["WL"] >= d["spectra_WL_range"][0][0])[0][0]
    ind2 = np.where(ncal["WL"] <= d["spectra_WL_range"][0][1])[0][-1]
    d["spectra_pix_range"] = [ind1, ind2]

    saved_pixels = np.arange(d["spectra_pix_range"][0], d["spectra_pix_range"][1] + 1)
    rows, cols = d["UV_INTEN"].shape

    ref = np.tile(ncal["Ref"][saved_pixels], (rows, 1))
    wl = np.tile(ncal["WL"][saved_pixels], (rows, 1))
    esw = np.tile(ncal["ESW"][saved_pixels], (rows, 1))
    eno3 = ncal["ENO3"][saved_pixels]

    uv_inten_sw = d["UV_INTEN"] - dc
    if np.count_nonzero(uv_inten_sw <= 0):
        uv_inten_sw[uv_inten_sw <= 0] = np.nan
        logger.info("UV intensities <= dark current found. Setting to NaN.")

    ratio = uv_inten_sw / ref
    bad_ratio = (~np.isfinite(ratio)) | (ratio <= 0)
    if np.any(bad_ratio):
        ratio[bad_ratio] = np.nan
        logger.info("Non-positive or invalid UV/REF ratios found. Setting to NaN.")

    abs_sw = -np.log10(ratio)
    ctd_temp = np.tile(d["T"], (cols, 1)).T
    ctd_sal = np.tile(d["S"], (cols, 1)).T
    cal_temp = np.tile(ncal["CalTemp"], (rows, cols))

    tcorr_coef = [1.27353e-07, -7.56395e-06, 2.91898e-05, 1.67660e-03, 1.46380e-02]
    tcorr = np.polyval(tcorr_coef, (wl - wl_offset)) * (ctd_temp - cal_temp)

    esw_in_situ = esw * np.exp(tcorr)
    if pcor_flag == 1:
        pres_term = (1 - d["P"] / 1000 * ncal["pres_coef"]).reshape(-1, 1)
        esw_in_situ *= pres_term

    abs_br_tcor = esw_in_situ * ctd_sal
    abs_cor = abs_sw - abs_br_tcor

    t_fit = (saved_pixels >= d["pix_fit_win"][0]) & (saved_pixels <= d["pix_fit_win"][1])
    fit_wl = wl[0, t_fit]
    fit_eno3 = eno3[t_fit]

    ones = np.ones_like(fit_eno3)
    m = np.column_stack((fit_eno3, ones / 100, fit_wl / 1000))
    m_inv = np.linalg.pinv(m)

    cols_m = m.shape[1]
    no3 = np.full((rows, cols_m + 3), np.nan)

    for i in range(rows):
        samp_abs_cor = abs_cor[i, t_fit]
        if np.all(np.isnan(samp_abs_cor)):
            continue

        no3[i, :3] = m_inv @ samp_abs_cor
        no3[i, 1] /= 100
        no3[i, 2] /= 1000

        abs_bl = wl[i, :] * no3[i, 2] + no3[i, 1]
        abs_no3_exp = eno3 * no3[i, 0]
        fit_dif = abs_cor[i, :] - abs_bl - abs_no3_exp

        rms_error = np.sqrt(np.nansum(fit_dif[t_fit] ** 2) / np.sum(t_fit))
        ind_240 = np.where(fit_wl <= 240)[0][-1]
        abs_240 = [fit_wl[ind_240], samp_abs_cor[ind_240]]
        no3[i, cols_m:cols_m + 3] = [rms_error, *abs_240]

    out = {
        "hdr": [
            "SDN",
            "AVG DC",
            "Pres",
            "Temp",
            "Sal",
            "SBE_NO3, uM/L",
            "NO3, uM/L",
            "BL_B",
            "BL_M",
            "RMS ERROR",
            "WL~240",
            "ABS~240",
        ],
        "info": {
            "WL_fit_window": d["WL_fit_win"],
            "spectra_WL_range": d["spectra_WL_range"],
            "WL_offset": wl_offset,
        },
    }

    avg_dc = np.nanmean(dc, axis=1)
    no3 = np.column_stack((d["SDN"], avg_dc, d["P"], d["T"], d["S"], d["NO3"], no3))
    tnan = np.isnan(no3[:, 5])
    out["data"] = no3[~tnan, :]

    return out


def calibrate_nitrate(
    df: pd.DataFrame,
    cruise: str,
    nitrate_col: str = "NitrateConcentration[uM]",
    dark_col: str = "DarkValueUsedForFit",
    cal_file: str | Path | None = None,
    cal_dir: str | Path | None = None,
) -> np.ndarray:
    """
    Calibrate nitrate concentration from SUNA sensor data.

    Behavior:
    - If cal_file is provided, use that calibration file.
    - Else if cal_dir is provided, scan that directory for a cruise-matching CAL file.
    - Else skip TSP correction and return raw nitrate values.

    Preserves alignment with the original dataframe rows and avoids mutating
    the caller's dataframe in place.
    """

    # Fast path: no calibration requested
    if cal_file is None and cal_dir is None:
        raw = df[nitrate_col].to_numpy(dtype=float, copy=True)

        if dark_col in df.columns:
            dark = df[dark_col].to_numpy(dtype=float, copy=False)
            raw[dark == 0] = np.nan

        return raw

    # Optional calibration lookup
    if cal_file is not None:
        cal_path = Path(cal_file)
    else:
        cal_path = find_suna_calibration_file(cruise=cruise, cal_dir=cal_dir)

    # Fast fallback: calibration requested, but not found
    if cal_path is None:
        logger.info(
            "No calibration file found for cruise %s, using raw nitrate values.",
            cruise,
        )

        raw = df[nitrate_col].to_numpy(dtype=float, copy=True)

        if dark_col in df.columns:
            dark = df[dark_col].to_numpy(dtype=float, copy=False)
            raw[dark == 0] = np.nan

        return raw

    if not cal_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {cal_path}")

    # Slow path starts here only when TSP correction is actually needed
    out_full = np.full(len(df), np.nan, dtype=float)
    df = df.copy()

    df[nitrate_col] = pd.to_numeric(df[nitrate_col], errors="coerce")
    if dark_col in df.columns:
        df[dark_col] = pd.to_numeric(df[dark_col], errors="coerce")

    pkg_dir = Path(__file__).resolve().parent.parent
    data_ref_dir = pkg_dir / "data_reference"
    spec_path = data_ref_dir / "spec.mat"
    hdr_file = data_ref_dir / "suna_hdr.txt"

    cal_path = ensure_suna_cal_lines(cal_path, "AP")
    spec = parse_mat(spec_path)
    ncal = parse_suna_calibration(cal_path)
    nitrate_hdr = load_suna_header(hdr_file)

    uv_cols = [col for col in df.columns if re.search(r"SpectrumCh", col)]
    uv_list = [col for col in nitrate_hdr if re.search(r"UV\(\d+(\.\d+)?\)", col)]

    if len(uv_cols) != len(uv_list):
        raise ValueError(
            f"Mismatch between dataframe spectrum columns ({len(uv_cols)}) "
            f"and SUNA header UV columns ({len(uv_list)})."
        )

    rename_map = dict(zip(uv_cols, uv_list))
    df = df.rename(columns=rename_map)

    required_cols = [
        nitrate_col,
        "Salinity",
        "Temperature",
        "Pressure",
        "matdate",
        dark_col,
        *uv_list,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for nitrate calibration: {missing}")

    df[uv_list] = df[uv_list].apply(pd.to_numeric, errors="coerce")

    valid = df[nitrate_col].notna()
    if not valid.any():
        return out_full

    work = df.loc[valid].copy()
    work_idx = work.index.to_numpy()

    spec["STP"] = [
        work["Salinity"].to_numpy(dtype=float),
        work["Temperature"].to_numpy(dtype=float),
        work["Pressure"].to_numpy(dtype=float),
    ]
    spec["UV_INTEN"] = work[uv_list].to_numpy(dtype=float)
    spec["DC"] = np.outer(work[dark_col].to_numpy(dtype=float), np.ones(len(uv_list)))
    spec["NO3"] = work[nitrate_col].to_numpy(dtype=float)
    spec["SDN"] = work["matdate"].to_numpy()
    spec["spectra_WL_range"][0][0] = ncal["WL"][0]
    spec["spectra_WL_range"][0][1] = ncal["WL"][-1]

    result = tsp_correction(spec, ncal, pcor_flag=1)
    corrected = result["data"][:, 6]

    if len(corrected) != len(work_idx):
        logger.warning(
            "Corrected nitrate length (%s) does not match valid input rows (%s). "
            "Attempting positional assignment to available rows only.",
            len(corrected),
            len(work_idx),
        )
        n = min(len(corrected), len(work_idx))
        out_full[work_idx[:n]] = corrected[:n]
    else:
        out_full[work_idx] = corrected

    dark_zero = work[dark_col].to_numpy(dtype=float) == 0
    out_full[work_idx[dark_zero]] = np.nan

    logger.info("Calibration file found for cruise %s, TSP correction applied.", cruise)
    return out_full