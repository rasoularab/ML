import numpy as np
import pandas as pd

from quantml.features import lagged_returns, rolling_volatility


def test_lagged_returns_uses_past_only(returns):
    f = lagged_returns(returns, lags=[1, 2, 5])
    assert list(f.columns) == ["lag_1", "lag_2", "lag_5"]
    assert f["lag_1"].iloc[5] == returns.iloc[4]
    assert f["lag_5"].iloc[5] == returns.iloc[0]
    assert np.isnan(f["lag_1"].iloc[0])


def test_lagged_returns_rejects_lag_zero(returns):
    import pytest
    with pytest.raises(ValueError):
        lagged_returns(returns, lags=[0, 1])


def test_rolling_volatility_does_not_use_current_return(returns):
    base = rolling_volatility(returns, 20)
    shocked = returns.copy()
    shocked.iloc[100] = 10.0  # a huge shock at t=100
    after = rolling_volatility(shocked, 20)
    assert base.iloc[100] == after.iloc[100]  # row 100 must not see r_100
    assert base.iloc[101] != after.iloc[101]  # row 101 may
