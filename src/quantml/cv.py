"""Cross-validation splitters. All yield (train_idx, test_idx) as integer position arrays."""
from __future__ import annotations

import numpy as np
import pandas as pd


class KFoldScratch:
    """Plain k-fold. Every row appears in exactly one test fold. Use ONLY for iid data, never for returns.

    TODO (Week 1, Wednesday). Parameters: n_splits, shuffle=False, seed=0.
    Done when tests/test_cv.py::test_kfold_partitions passes.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = False, seed: int = 0):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed

    def split(self, X):
        raise NotImplementedError


class WalkForward:
    """Expanding or rolling walk-forward splits: train on the past, test on the next block.

    TODO (Week 1, Wednesday).
    Parameters: train_size (int, minimum training rows), test_size (int), step (int),
                expanding (bool; True keeps the training start fixed, False rolls it).
    Guarantee: max(train_idx) < min(test_idx) in every split.
    Done when tests/test_cv.py::test_walk_forward_no_future passes.
    """

    def __init__(self, train_size: int, test_size: int, step: int | None = None, expanding: bool = True):
        self.train_size = train_size
        self.test_size = test_size
        self.step = step or test_size
        self.expanding = expanding

    def split(self, X):
        raise NotImplementedError


class PurgedKFold:
    """K-fold for labels that span time (de Prado, AFML ch 7).

    Parameters
    ----------
    n_splits : number of contiguous test folds
    t1 : pd.Series, index = start date of each label, values = end date of its label (from labels.label_end_times)
    embargo_pct : fraction of rows to drop from training right after each test fold

    For each test fold, remove from the training set (1) every sample whose label interval overlaps the
    test window and (2) the first int(n * embargo_pct) rows after the test fold.

    TODO (Week 1, Friday).
    Done when tests/test_cv.py::test_purged_kfold_no_overlap and ::test_purged_kfold_embargo pass.
    """

    def __init__(self, n_splits: int, t1: pd.Series, embargo_pct: float = 0.0):
        self.n_splits = n_splits
        self.t1 = t1
        self.embargo_pct = embargo_pct

    def split(self, X):
        raise NotImplementedError
