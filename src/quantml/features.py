"""Feature construction. Rule: row t may only use data up to t-1 (see docs/CONVENTIONS.md)."""
from __future__ import annotations

import pandas as pd


def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """WORKED EXAMPLE of the timing rule: standard deviation of the previous `window` returns.

    The `shift(1)` is what keeps r_t out of the feature at row t. Delete it and the
    look-ahead test in tests/test_features.py will fail: read that test to see why.
    """
    return returns.shift(1).rolling(window).std()


def lagged_returns(returns: pd.Series, lags) -> pd.DataFrame:
    """Lagged-return features. Column `lag_k` at row t equals r_{t-k}; every k must be >= 1.
    """
    if not all(k >= 1 for k in lags):
        raise ValueError("all lags must be >= 1")

        # 2) build one column per lag
    cols = {f"lag_{k}": returns.shift(k) for k in lags}

    # 3) wrap it in a DataFrame
    return pd.DataFrame(cols, index=returns.index)
    


def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Rank each column within each date, scaled to [-1, 1] (a standard preprocessing step in
    the Gu-Kelly-Xiu study). Input has a MultiIndex ['date', 'asset'].

    TODO (Week 2).
    """
    raise NotImplementedError
