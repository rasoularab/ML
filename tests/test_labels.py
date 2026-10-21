import numpy as np
import pandas as pd

from quantml.labels import forward_return, label_end_times, sample_uniqueness


def test_forward_return():
    r = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=pd.bdate_range("2021-01-04", periods=5))
    y = forward_return(r, horizon=3)
    assert abs(y.iloc[0] - 0.6) < 1e-12  # r0 + r1 + r2
    assert abs(y.iloc[2] - 1.2) < 1e-12  # r2 + r3 + r4
    assert y.iloc[3:].isna().all()


def test_label_end_times():
    idx = pd.bdate_range("2021-01-04", periods=6)
    t1 = label_end_times(idx, horizon=3)
    assert t1.iloc[0] == idx[2] and t1.iloc[5] == idx[5]


def test_uniqueness():
    idx = pd.bdate_range("2021-01-04", periods=3)
    # label 0 spans days 0-1, label 1 spans days 1-2, label 2 spans day 2 only
    t1 = pd.Series([idx[1], idx[2], idx[2]], index=idx)
    u = sample_uniqueness(idx, t1)
    # concurrency by day: d0=1, d1=2, d2=2  -> label0 = (1 + 1/2)/2, label1 = (1/2 + 1/2)/2, label2 = 1/2
    np.testing.assert_allclose(u.values, [0.75, 0.5, 0.5])
