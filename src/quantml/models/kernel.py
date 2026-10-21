"""Kernel methods (Week 2)."""
from __future__ import annotations

import numpy as np


def rbf_kernel(A, B, gamma: float):
    """K[i, j] = exp(-gamma * ||A_i - B_j||^2).  TODO (Week 2, Friday)."""
    raise NotImplementedError


class KernelRidgeScratch:
    """Kernel ridge regression with an RBF kernel.

    Solves (K + alpha I) a = y on the training data; predicts with K(X_new, X_train) a.
    Matches sklearn.kernel_ridge.KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma) (no intercept).

    TODO (Week 2, Friday).
    Done when tests/test_models_vs_sklearn.py::test_kernel_ridge_matches_sklearn passes.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 1.0):
        self.alpha = alpha
        self.gamma = gamma

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
