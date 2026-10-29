from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import poisson

def add_poisson_ci(df_bin, raw_counts_by_bin, data_cols, scale_factor, ci=0.95):
    ci_results = {}

    for col in data_cols:
        lower = []
        upper = []

        for count in raw_counts_by_bin[col]:
            lo, hi = poisson.interval(ci, int(count))
            lower.append(lo * scale_factor)
            upper.append(hi * scale_factor)

        ci_results[f"{col}_ci_lower"] = lower
        ci_results[f"{col}_ci_upper"] = upper

    return pd.concat([df_bin, pd.DataFrame(ci_results, index=df_bin.index)], axis=1)