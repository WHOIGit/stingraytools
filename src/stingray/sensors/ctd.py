# sensors/ctd.py

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


@njit
def _ies80_numba(s, t, p):
    """
    Numba kernel for EOS-80 / UNESCO 1980 seawater density.

    Parameters
    ----------
    s : 1D float64 ndarray
        Practical salinity.
    t : 1D float64 ndarray
        Temperature in degrees C.
    p : 1D float64 ndarray
        Pressure.

    Returns
    -------
    rho : 1D float64 ndarray
        Density.
    """
    n = s.shape[0]
    rho = np.empty(n, dtype=np.float64)

    for i in range(n):
        si = s[i]
        ti = t[i]
        pi = p[i]

        if np.isnan(si) or np.isnan(ti) or np.isnan(pi):
            rho[i] = np.nan
            continue

        r0 = (
            999.842594
            + 6.793952e-2 * ti
            - 9.09529e-3 * ti**2
            + 1.001685e-4 * ti**3
            - 1.120083e-6 * ti**4
            + 6.536332e-9 * ti**5
            + (
                8.24493e-1
                - 4.0899e-3 * ti
                + 7.6438e-5 * ti**2
                - 8.2467e-7 * ti**3
                + 5.3875e-9 * ti**4
            ) * si
            + (
                -5.72466e-3
                + 1.0227e-4 * ti
                - 1.6546e-6 * ti**2
            ) * si**1.5
            + 4.8314e-4 * si**2
        )

        if pi != 0.0:
            k = (
                19652.21
                + 148.4206 * ti
                - 2.327105 * ti**2
                + 1.360447e-2 * ti**3
                - 5.155288e-5 * ti**4
                + (
                    3.239908
                    + 1.43713e-3 * ti
                    + 1.16092e-4 * ti**2
                    - 5.77905e-7 * ti**3
                ) * pi
                + (
                    8.50935e-5
                    - 6.12293e-6 * ti
                    + 5.2787e-8 * ti**2
                ) * pi**2
                + (
                    54.6746
                    - 0.603459 * ti
                    + 1.09987e-2 * ti**2
                    - 6.1670e-5 * ti**3
                ) * si
                + (
                    7.944e-2
                    + 1.6483e-2 * ti
                    - 5.3009e-4 * ti**2
                ) * si**1.5
                + (
                    2.2838e-3
                    - 1.0981e-5 * ti
                    - 1.6078e-6 * ti**2
                ) * pi * si
                + 1.91075e-4 * pi * si**1.5
                + (
                    9.9348e-7
                    + 2.0816e-8 * ti
                    + 9.1697e-10 * ti**2
                ) * pi**2 * si
            )
            rho[i] = r0 / (1.0 - pi / k)
        else:
            rho[i] = r0

    return rho


def density_ies80(s, t, p):
    """
    Compute seawater density using the EOS-80 / UNESCO 1980 polynomial.

    Parameters
    ----------
    s : array-like
        Practical salinity.
    t : array-like
        Temperature in degrees C.
    p : array-like
        Pressure.
        Use pressure units consistent with the original implementation.

    Returns
    -------
    numpy.ndarray
        Density values.
    """
    s = np.asarray(s, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    if s.ndim != 1 or t.ndim != 1 or p.ndim != 1:
        raise ValueError("s, t, and p must be 1D array-like inputs")
    if not (s.shape[0] == t.shape[0] == p.shape[0]):
        raise ValueError("s, t, and p must have the same length")

    return _ies80_numba(s, t, p)


def add_density(
    df: pd.DataFrame,
    salinity_col: str = "salinity",
    temperature_col: str = "temperature",
    pressure_col: str = "pressure",
    output_col: str = "density",
    copy: bool = True,
) -> pd.DataFrame:
    """
    Add EOS-80 density to a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input CTD data.
    salinity_col : str
        Column name for salinity.
    temperature_col : str
        Column name for temperature.
    pressure_col : str
        Column name for pressure.
    output_col : str
        Output density column name.
    copy : bool
        If True, return a copy. If False, modify input DataFrame in place.

    Returns
    -------
    pandas.DataFrame
        DataFrame with density column added.
    """
    required = [salinity_col, temperature_col, pressure_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"df missing required columns: {missing}")

    out = df.copy() if copy else df
    out[output_col] = density_ies80(
        out[salinity_col].to_numpy(),
        out[temperature_col].to_numpy(),
        out[pressure_col].to_numpy(),
    )
    return out