"""Linear models from scratch (NumPy only). Match scikit-learn's conventions so the tests can compare."""
from __future__ import annotations

import numpy as np


class RidgeScratch:
    """Ridge regression.

    Minimizes ||y - Xw - b||^2 + alpha * ||w||^2 with the intercept b NOT penalized
    (same objective as sklearn.linear_model.Ridge).

    Attributes after fit: coef_ (n_features,), intercept_ (float).

    TODO (Week 1, Thursday).
    Hint: center X and y, solve (Xc^T Xc + alpha I) w = Xc^T yc (or use the SVD), then b = ybar - xbar . w.
    Done when tests/test_models_vs_sklearn.py::test_ridge_matches_sklearn passes.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class LassoScratch:
    """Lasso by cyclic coordinate descent.

    Minimizes (1 / (2 n)) * ||y - Xw - b||^2 + alpha * ||w||_1 (same as sklearn.linear_model.Lasso).
    Use soft-thresholding for each coordinate update and an unpenalized intercept.

    Parameters: alpha, n_iter (maximum sweeps), tol (stop when the largest coefficient change < tol).
    Attributes after fit: coef_, intercept_.

    TODO (Week 1, Thursday).
    Done when tests/test_models_vs_sklearn.py::test_lasso_matches_sklearn passes.
    """

    def __init__(self, alpha: float = 0.1, n_iter: int = 5000, tol: float = 1e-10):
        self.alpha = alpha
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
