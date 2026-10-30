# utils/gridding.py

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numba import njit

logger = logging.getLogger(__name__)

def bin_data(
    df: pd.DataFrame,
    cols: list[str],
    steps: list[int | float],
    suffix: str = "_bin",
) -> pd.DataFrame:
    """
    Bin numeric or datetime columns by given step sizes.

    NaN and NaT values are preserved.
    """
    if len(cols) != len(steps):
        raise ValueError("cols and steps must have the same length")

    out = df.copy()

    for col, step in zip(cols, steps):
        if col not in out.columns:
            raise ValueError(f"df missing required column: {col}")

        if np.issubdtype(out[col].dtype, np.datetime64):
            binned = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
            valid = out[col].notna()

            raw_ns = out.loc[valid, col].astype("int64")
            binned.loc[valid] = pd.to_datetime(
                ((raw_ns + step // 2) // step) * step
            )

            out[f"{col}{suffix}"] = binned
        else:
            out[f"{col}{suffix}"] = (
                np.floor((out[col] + step / 2.0) / step) * step
            )

    return out


def bin_vectors(
    vectors: list[np.ndarray],
    steps: list[int | float],
) -> list[np.ndarray]:
    """
    Bin numeric or datetime vectors by given step sizes, with simple hysteresis.
    For each input vector in `vectors` and its corresponding step in `steps`:
    1. We first compute a "naive" bin using standard quantization:
           numeric:   floor((x + step/2) / step) * step
           datetime:  round to nearest bin in ns using the same idea on int64
    2. Then we apply a 1D hysteresis rule along the original element order:
           - We keep track of:
               last_raw : the last *committed* raw value
               last_bin : the bin assigned to that committed value
           - For each new sample raw[i]:
               if |raw[i] - last_raw| < step:
                   treat as noise  → assign bin[i] = last_bin
               else:
                   treat as real move → accept bin[i] as computed,
                                        and update last_raw, last_bin
       This means:
         - Tiny fluctuations less than `step` around the last committed value
           will NOT cause a bin change.
         - A change of at least `step` (or more) in the raw value will create
           a new bin.
         - This works for any numeric vector and for datetime64 vectors (in ns).
    3. NaNs / NaT:
         - Datetime NaT values are preserved and never binned.
         - Numeric NaNs keep their quantized value, but they do NOT update
           hysteresis state (they are effectively skipped in the hysteresis
           comparison).
    The function returns a list of binned vectors, one per input vector.
    """
    """
    Prepare a list to collect the binned output vectors.
    """
    outputs = []
    """
    Loop over each input vector and its corresponding step size.
    """
    for vec, step in zip(vectors, steps):
        # Convert to NumPy array (do not copy unless needed)
        raw = np.asarray(vec)
        # ------------------------------------------------------------------ #
        # Datetime64 case
        # ------------------------------------------------------------------ #
        if np.issubdtype(raw.dtype, np.datetime64):
            """
            For datetime vectors:
            1. Work only on valid (non-NaT) entries.
            2. Convert datetimes to int64 nanoseconds for arithmetic.
            3. Quantize (round) using the same midpoint rule:
                   q = round_to_nearest_step(raw_ns, step_ns)
            4. Apply hysteresis on the raw_ns values:
                   compare current raw_ns to last committed raw_ns.
            """
            # Initialize output array with NaT so missing values are preserved
            out = np.full(raw.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
            # Boolean mask for non-NaT values
            valid = ~np.isnat(raw)
            # Raw datetime values as int64 nanoseconds (only where valid)
            raw_ns = raw[valid].astype("int64")
            """
            Step 1: naive binning (quantization).
            We do midpoint rounding by adding step//2 before integer division:
              - (raw_ns + step//2) // step   gives the bin index
              - * step                      rescales to ns units
            """
            q_ns = ((raw_ns + step // 2) // step) * step
            """
            Step 2: hysteresis on actual sample difference.
            We walk through raw_ns in original order and maintain:
              last_raw : last committed raw ns value
              last_bin : bin assigned at that commitment
            For each index i:
                if this is the first valid sample:
                    commit it directly (seed hysteresis)
                else:
                    if |raw_ns[i] - last_raw| < step:
                        → noise → override q_ns[i] with last_bin
                    else:
                        → real move → accept q_ns[i] and update last_raw, last_bin
            """
            last_raw = None
            last_bin = None
            for i in range(raw_ns.size):
                current_raw = raw_ns[i]
                if last_raw is None:
                    # First valid sample: seed hysteresis state
                    last_raw = current_raw
                    last_bin = q_ns[i]
                else:
                    # Compare actual change in ns to step
                    if abs(current_raw - last_raw) < step:
                        # Too small: treat as noise, keep previous bin
                        q_ns[i] = last_bin
                    else:
                        # Big enough: commit a new bin and update hysteresis state
                        last_raw = current_raw
                        last_bin = q_ns[i]
            # Convert back to datetime64[ns] and fill the output array
            out[valid] = q_ns.astype("datetime64[ns]")
            outputs.append(out)
        # ------------------------------------------------------------------ #
        # Numeric case
        # ------------------------------------------------------------------ #
        else:
            """
            For numeric vectors:
            1. We compute the naive quantized bin:
                   q = floor((x + step/2) / step) * step
               which is midpoint rounding to the nearest step.
            2. We apply the same hysteresis idea as for datetimes, but now
               using raw numeric values instead of ns.
            """
            raw = raw.astype(float, copy=False)
            """
            Step 1: naive numeric binning.
            - Add step/2 before division to implement midpoint rounding.
            - floor(...) then multiply by step gives the quantized bin center.
            """
            q = np.floor((raw + step / 2.0) / step) * step
            """
            Step 2: hysteresis on actual sample difference.
            We iterate in element order and keep:
              last_raw : last committed raw numeric value
              last_bin : bin assigned at that commitment
            For each index i:
                - If raw[i] is NaN:
                    * skip it for hysteresis (do not update last_raw/last_bin)
                    * leave q[i] as is (NaN remains NaN)
                - Else:
                    * If this is the first non-NaN:
                          seed hysteresis with this raw/bin.
                    * Else:
                          if |raw[i] - last_raw| < step:
                              → noise → set q[i] = last_bin
                          else:
                              → real move → accept q[i] and update
                                            last_raw, last_bin
            """
            last_raw = None
            last_bin = None
            for i in range(raw.size):
                current_raw = raw[i]
                # Treat NaN as a gap: do not update hysteresis state
                if np.isnan(current_raw):
                    continue
                if last_raw is None:
                    # First non-NaN: seed hysteresis
                    last_raw = current_raw
                    last_bin = q[i]
                else:
                    if abs(current_raw - last_raw) < step:
                        # Small change → noise → stay in the same bin
                        q[i] = last_bin
                    else:
                        # Large enough → commit new bin and update hysteresis
                        last_raw = current_raw
                        last_bin = q[i]
            outputs.append(q)
    """
    Return the list of binned vectors, one per input vector.
    """
    return outputs

def mid2range(midpoints) -> np.ndarray:
    """
    Convert bin midpoints to bin edge ranges.

    Examples
    --------
    [1, 2, 3] -> [0.5, 1.5, 2.5, 3.5]
    """
    midpoints = np.asarray(midpoints, dtype=float)

    if midpoints.ndim != 1:
        raise ValueError("midpoints must be 1D")

    if midpoints.size < 2:
        raise ValueError("midpoints must contain at least two values")

    edges = np.empty(midpoints.size + 1, dtype=float)

    edges[0] = midpoints[0] - (midpoints[1] - midpoints[0]) / 2.0

    for i in range(midpoints.size - 1):
        edges[i + 1] = (midpoints[i] + midpoints[i + 1]) / 2.0

    edges[-1] = midpoints[-1] + (midpoints[-1] - midpoints[-2]) / 2.0

    return edges

@njit
def assign_time_bins(
    timestamps,
    bin_width,
    grid_start,
    grid_end,
):
    n = timestamps.shape[0]
    out = np.empty(n, dtype=np.float64)

    n_bins = int(np.ceil((grid_end - grid_start) / bin_width))

    for i in range(n):
        ti = timestamps[i]

        if not np.isfinite(ti):
            out[i] = np.nan
            continue

        idx = int(np.floor((ti - grid_start) / bin_width))

        if idx < 0 or idx >= n_bins:
            out[i] = np.nan
        else:
            out[i] = grid_start + idx * bin_width

    return out


def add_time_bin(
    df: pd.DataFrame,
    ts_col: str,
    grid_start: float,
    grid_end: float,
    bin_width: float,
    output_col: str = "time_bin",
    copy: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return df.copy() if copy else df

    if ts_col not in df.columns:
        raise ValueError(f"df missing required column: {ts_col}")

    out = df.copy() if copy else df

    ts = np.asarray(out[ts_col], dtype=np.float64)
    out[output_col] = assign_time_bins(ts, bin_width, grid_start, grid_end)

    return out


def warn_unmatched(
    name: str,
    df: pd.DataFrame,
    time_bin_col: str = "time_bin",
) -> None:
    if df.empty:
        return

    if time_bin_col not in df.columns:
        raise ValueError(f"df missing required column: {time_bin_col}")

    n_bad = np.isnan(df[time_bin_col]).sum()

    if n_bad > 0:
        logger.error(f"{name}: {n_bad} rows FAILED to map to CTD time bins")


def infer_bin_width(sensor_times) -> float:
    """Bin width of a sensor table: the smallest gap between its bin timestamps."""
    times = np.unique(np.asarray(sensor_times, dtype=np.float64))
    times = times[np.isfinite(times)]
    gaps = np.diff(times)
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        raise ValueError("Cannot infer bin width: need at least two distinct sensor timestamps.")
    return round(float(gaps.min()), 6)


def bin_to_sensor_grid(timestamps, sensor_times, bin_width: float) -> np.ndarray:
    """
    Assign timestamps to the sensor time-bin grid.

    A bin starts at its sensor timestamp and spans ``bin_width`` seconds. The
    grid runs from the first to the last sensor timestamp; anything outside it
    is NaN. This is the single binning rule shared by media linking and
    abundance, so both place a frame in the same bin.
    """
    sensor_times = np.asarray(sensor_times, dtype=np.float64)
    return assign_time_bins(
        np.asarray(timestamps, dtype=np.float64),
        bin_width,
        np.nanmin(sensor_times),
        np.nanmax(sensor_times),
    )
