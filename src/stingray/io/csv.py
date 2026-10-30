# io/csv.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import concurrent.futures
import logging
import os
import tempfile

import pandas as pd

logger = logging.getLogger(__name__)


def _pyarrow_available() -> bool:
    try:
        import pyarrow.csv  # noqa: F401
    except ImportError:
        return False
    return True


def read_csv_parallel(
    file_list: Iterable[str | Path],
    max_workers: int | None = None,
    strict: bool = False,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Read multiple CSV files in parallel and concatenate into a single DataFrame.

    Uses the pyarrow parser when installed and no extra read_csv options are
    given, falling back to the pandas C parser per file on any error.

    Parameters
    ----------
    file_list : iterable of str or Path
        Files to read.
    max_workers : int, optional
        Number of threads. Defaults to min(32, cpu_count()).
    strict : bool
        If True, raise on first failure instead of skipping files.
    **read_csv_kwargs
        Passed directly to pandas.read_csv.

    Returns
    -------
    DataFrame
    """

    files = [Path(f) for f in file_list]

    if len(files) == 0:
        return pd.DataFrame()

    use_arrow = _pyarrow_available() and not read_csv_kwargs

    def safe_read_csv(file: Path) -> pd.DataFrame:
        try:
            if use_arrow:
                try:
                    return pd.read_csv(file, engine="pyarrow")
                except Exception:
                    pass  # fall back to the default parser below
            return pd.read_csv(file, low_memory=False, **read_csv_kwargs)
        except Exception as e:
            if strict:
                raise
            logger.warning(f"Failed to read {file}: {e}")
            return pd.DataFrame()

    # Better default: bounded threads
    if max_workers is None:
        cpu = os.cpu_count() or 1
        max_workers = min(32, cpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        dfs = list(executor.map(safe_read_csv, files))

    # Filter empty frames
    dfs = [df for df in dfs if not df.empty]

    if not dfs:
        logger.warning("No CSV files were successfully read.")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def write_csv_atomic(df: pd.DataFrame, path: str | Path, **to_csv_kwargs) -> Path:
    """
    Write a CSV through a temporary file and replace the target in one step.

    Several pipeline steps rewrite the same cruise CSV, so an interrupted write
    must never leave a half-written file for the next step to read.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_csv_kwargs.setdefault("index", False)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temp_path = Path(temp_name)
    try:
        df.to_csv(temp_path, **to_csv_kwargs)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return path
