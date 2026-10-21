"""Trees and boosting from scratch (Week 3)."""
from __future__ import annotations

import numpy as np


def best_split(x, y):
    """Best split of ONE feature for squared-error regression.

    Returns (threshold, gain) where gain = SSE(parent) - SSE(left) - SSE(right).
    Efficient version: sort by x, use cumulative sums of y and y^2 so each candidate split costs O(1).
    Only consider thresholds between distinct consecutive x values.

    TODO (Week 3, Tuesday).
    Done when tests/test_trees.py::test_best_split_finds_step passes.
    """
    raise NotImplementedError


class RegressionTree:
    """CART regression tree with max_depth and min_samples_leaf.

    fit(X, y) -> self; predict(X) -> np.ndarray. A depth-1 tree must equal sklearn's DecisionTreeRegressor(max_depth=1).

    TODO (Week 3, Tuesday).
    Done when tests/test_trees.py::test_tree_depth1_matches_sklearn passes.
    """

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class GradientBoostingStumps:
    """Gradient boosting for squared loss with depth-1 trees.

    Start from the mean; at each round fit a stump to the residuals and add learning_rate * stump.
    Keep `train_loss_` (a list with the training MSE after each round).

    TODO (Week 3, Wednesday).
    Done when tests/test_trees.py::test_boosting_reduces_train_loss passes.
    """

    def __init__(self, n_rounds: int = 100, learning_rate: float = 0.1):
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
