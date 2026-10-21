"""Targets and labels. Row t predicts the return earned from t over the next h periods."""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(returns: pd.Series, horizon: int = 1) -> pd.Series:
    """y_t = r_t + r_{t+1} + ... + r_{t+h-1} (sum of simple returns; NaN where the window runs past the end).

    A trailing rolling sum at row i+h-1 covers r_i..r_{i+h-1}; shifting it back by h-1 rows puts it on row i.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return returns.rolling(horizon).sum().shift(-(horizon - 1))


def label_end_times(index: pd.DatetimeIndex, horizon: int) -> pd.Series:
    """For each row t, the date of the last return included in y_t: index[min(i + horizon - 1, n - 1)].

    Returned as a Series (index = `index`, values = end dates). PurgedKFold uses it as `t1`.
    """
    n = len(index)
    end_pos = np.minimum(np.arange(n) + horizon - 1, n - 1)
    return pd.Series(index[end_pos], index=index)


def sample_uniqueness(t0: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """Average uniqueness of each label: for every label, the mean of 1 / (number of labels alive) over its life.

    Labels that overlap a lot get a low weight, labels that overlap nothing get weight 1.
    Reference: Advances in Financial Machine Learning, ch 4.

    TODO (Week 3, Thursday).
    Done when tests/test_labels.py::test_uniqueness passes.
    """
    raise NotImplementedError


def triple_barrier(prices: pd.Series, vol: pd.Series, pt_sl=(1.0, 1.0), horizon: int = 10) -> pd.DataFrame:
    """Triple-barrier labels. Returns a DataFrame with columns ['label', 't1'].

    label: +1 if the upper barrier is hit first, -1 if the lower, 0 (or sign of return) at the vertical barrier.
    TODO (Week 4). The test file arrives at the start of Week 4.
    """
    raise NotImplementedError
