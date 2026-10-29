# profiles/identify.py

from __future__ import annotations

import numpy as np
from numba import njit


@njit
def _nanmedian_window(arr, lo, hi):
    """
    Median of arr[lo:hi], ignoring NaNs.
    Returns NaN if all values are NaN.
    """
    count = 0
    for j in range(lo, hi):
        if not np.isnan(arr[j]):
            count += 1

    if count == 0:
        return np.nan

    buf = np.empty(count, dtype=np.float64)
    k = 0
    for j in range(lo, hi):
        v = arr[j]
        if not np.isnan(v):
            buf[k] = v
            k += 1

    buf.sort()
    return buf[count // 2]


@njit
def identify_profiles(
    depth,
    time_seconds,
    bin_size=1.0,
    smooth_window=7,
    dir_window=10,
    max_gap_sec=3600.0,
    min_samples=5,
    surface_start_threshold=20.0,
    min_turn_depth=5.0,
    min_turn_time=30.0,
):
    """
    Identify CTD profiles (casts) from depth and time.

    This function segments a continuous depth time series into separate
    physical profiles (downcasts, upcasts, and separate deployments).

    The algorithm is designed to be robust to:
      - wire jitter and ship heave
      - short winch pauses
      - noisy depth measurements
      - small yo-yo motion near turning points

    It uses quantization + median smoothing BEFORE estimating direction,
    then detects real turnarounds with hysteresis, and enforces hard breaks
    on time gaps.

    Parameters
    ----------
    depth : 1D array of float
        Raw depth measurements (positive downward).
    time_seconds : 1D array of float
        Time in seconds (monotonic increasing).
    bin_size : float
        Depth bin size (meters) used for midpoint quantization.
        Acts like a deadband: depth changes smaller than this are ignored.
    smooth_window : int
        Window size (samples) for rolling median smoothing of quantized depth.
        Suppresses spikes and high-frequency jitter.
    dir_window : int
        Window size (samples) used to estimate direction of motion.
        Larger = smoother direction but more lag near turning points.
    max_gap_sec : float
        Time gap (seconds) that forces a new profile (e.g., recovery/redeploy).
    min_samples : int
        Minimum number of unique depth bins required for a profile to be valid.
        Short micro-profiles are treated as noise.
    surface_start_threshold : float
        A profile is considered valid only if it reaches a shallow depth
        less than or equal to this threshold at least once.
        This rejects segments that begin too deep in the water column,
        such as mid-cast fragments or incomplete deployments.
    min_turn_depth : float
        Minimum depth reversal (meters) required to confirm a turnaround.
        Suppresses tiny yo-yo motion at the bottom.
    min_turn_time : float
        Minimum time (seconds) after the last extremum before confirming
        a turnaround. Prevents rapid oscillations from splitting profiles.

    Returns
    -------
    profile_out : 1D float array
        Profile labels for each sample.
        Labels are contiguous integers starting at 0.
        NaN indicates samples that belong to invalid profiles with no valid
        neighbor to merge into.
    deployment_id : 1D int array
        Deployment labels for each sample.
        Increments only across hard time gaps.
        Not affected by turnaround logic or merging.
    """
    n = depth.size

    if time_seconds.size != n:
        raise ValueError("depth and time_seconds must have the same length")

    if n == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    depth_q = np.empty(n, dtype=np.float64)
    half_bin = bin_size / 2.0

    for i in range(n):
        d = depth[i]
        if np.isnan(d):
            depth_q[i] = np.nan
        else:
            depth_q[i] = np.floor((d + half_bin) / bin_size) * bin_size

    smooth = np.empty(n, dtype=np.float64)
    hw = smooth_window // 2

    for i in range(n):
        lo = 0 if i < hw else i - hw
        hi = n if i + hw + 1 > n else i + hw + 1
        smooth[i] = _nanmedian_window(depth_q, lo, hi)

    direction = np.empty(n, dtype=np.float64)
    direction[:] = np.nan

    half = dir_window // 2
    for i in range(half, n - half):
        prev_med = smooth[i - half]
        next_med = smooth[i + half]

        if np.isnan(prev_med) or np.isnan(next_med):
            continue

        d = next_med - prev_med
        if d > 0:
            direction[i] = 1.0
        elif d < 0:
            direction[i] = -1.0

    profile = np.zeros(n, dtype=np.int64)
    hard_profile = np.zeros(n + 1, dtype=np.bool_)

    deployment_id = np.zeros(n, dtype=np.int64)
    current_deployment = 0
    deployment_id[0] = 0

    last_dir = 0.0
    last_extreme_depth = smooth[0]
    last_extreme_time = time_seconds[0]
    last_extreme_idx = 0

    for i in range(1, n):
        gap = (time_seconds[i] - time_seconds[i - 1]) > max_gap_sec

        if gap:
            current_deployment += 1
            deployment_id[i] = current_deployment

            new_pid = profile[i - 1] + 1
            profile[i] = new_pid
            hard_profile[new_pid] = True

            last_dir = 0.0
            last_extreme_depth = smooth[i]
            last_extreme_time = time_seconds[i]
            last_extreme_idx = i
            continue

        deployment_id[i] = current_deployment
        profile[i] = profile[i - 1]

        d = direction[i]
        s = smooth[i]

        if np.isnan(d) or np.isnan(s):
            continue

        dz = s - last_extreme_depth
        dt = time_seconds[i] - last_extreme_time

        if last_dir == 0.0:
            last_dir = d
            last_extreme_depth = s
            last_extreme_time = time_seconds[i]
            last_extreme_idx = i
            continue

        if d != last_dir and abs(dz) >= min_turn_depth and dt >= min_turn_time:
            new_pid = profile[i - 1] + 1
            for k in range(last_extreme_idx + 1, i + 1):
                profile[k] = new_pid

            last_dir = d
            last_extreme_depth = s
            last_extreme_time = time_seconds[i]
            last_extreme_idx = i
        else:
            if (last_dir > 0.0 and s > last_extreme_depth) or (
                last_dir < 0.0 and s < last_extreme_depth
            ):
                last_extreme_depth = s
                last_extreme_time = time_seconds[i]
                last_extreme_idx = i

    profile_out = profile.astype(np.float64)

    max_pid = profile[n - 1] + 1
    is_valid = np.zeros(max_pid, dtype=np.bool_)
    profile_deployment = np.full(max_pid, -1, dtype=np.int64)

    for pid in range(max_pid):
        seen = 0
        last = np.nan
        min_depth = 1e20

        for i in range(n):
            if profile[i] == pid:
                if profile_deployment[pid] == -1:
                    profile_deployment[pid] = deployment_id[i]

                d = depth_q[i]
                if not np.isnan(d):
                    if d < min_depth:
                        min_depth = d
                    if np.isnan(last) or d != last:
                        seen += 1
                        last = d

        is_valid[pid] = (seen >= min_samples) and (min_depth <= surface_start_threshold)

    for pid in range(max_pid):
        if is_valid[pid]:
            continue

        prev_valid = -1
        this_dep = profile_deployment[pid]

        for p in range(pid - 1, -1, -1):
            if hard_profile[p + 1]:
                break
            if profile_deployment[p] != this_dep:
                break
            if is_valid[p]:
                prev_valid = p
                break

        if prev_valid >= 0:
            for i in range(n):
                if profile[i] == pid:
                    profile_out[i] = float(prev_valid)
        else:
            for i in range(n):
                if profile[i] == pid:
                    profile_out[i] = np.nan

    old_to_new = np.full(max_pid, -1, dtype=np.int64)
    new_id = 0

    for i in range(n):
        pid = profile_out[i]
        if not np.isnan(pid):
            old_pid = int(pid)
            if old_to_new[old_pid] == -1:
                old_to_new[old_pid] = new_id
                new_id += 1

    for i in range(n):
        pid = profile_out[i]
        if not np.isnan(pid):
            profile_out[i] = float(old_to_new[int(pid)])

    return profile_out, deployment_id