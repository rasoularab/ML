import numpy as np
import pandas as pd

from quantml.cv import KFoldScratch, PurgedKFold, WalkForward
from quantml.labels import label_end_times


def test_kfold_partitions():
    X = np.zeros((103, 2))
    seen = []
    for tr, te in KFoldScratch(n_splits=5).split(X):
        assert len(set(tr) & set(te)) == 0
        assert len(tr) + len(te) == 103
        seen.extend(te)
    assert sorted(seen) == list(range(103))  # every row tested exactly once


def test_walk_forward_no_future():
    X = np.zeros((200, 1))
    splits = list(WalkForward(train_size=100, test_size=20).split(X))
    assert len(splits) == 5
    for tr, te in splits:
        assert tr.max() < te.min()
    assert splits[0][0].min() == splits[-1][0].min() == 0  # expanding keeps the start fixed


def test_walk_forward_rolling_window_size():
    X = np.zeros((200, 1))
    for tr, te in WalkForward(train_size=100, test_size=20, expanding=False).split(X):
        assert len(tr) == 100 and tr.max() < te.min()


def _setup(n=200, h=5):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return idx, label_end_times(idx, h)


def test_purged_kfold_no_overlap():
    idx, t1 = _setup()
    X = np.zeros((len(idx), 1))
    for tr, te in PurgedKFold(n_splits=5, t1=t1).split(X):
        t_start, t_end = idx[te.min()], t1.iloc[te].max()
        for i in tr:
            # training label [idx[i], t1[i]] must not intersect the test window
            assert t1.iloc[i] < t_start or idx[i] > t_end


def test_purged_kfold_embargo():
    idx, t1 = _setup()
    n = len(idx)
    X = np.zeros((n, 1))
    pct = 0.05
    k = int(n * pct)
    for tr, te in PurgedKFold(n_splits=5, t1=t1, embargo_pct=pct).split(X):
        after = np.arange(te.max() + 1, min(te.max() + 1 + k, n))
        assert len(set(tr) & set(after)) == 0
