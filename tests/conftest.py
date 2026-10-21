import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def returns(rng):
    idx = pd.bdate_range("2020-01-01", periods=300)
    return pd.Series(rng.normal(0, 0.01, len(idx)), index=idx, name="r")
