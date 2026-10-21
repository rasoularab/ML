"""Dimension reduction (Week 2)."""
from __future__ import annotations

import numpy as np


class PCAScratch:
    """PCA via the SVD of the centered data.

    Parameters: n_components.
    Attributes after fit: mean_ (n_features,), components_ (n_components, n_features),
                          explained_variance_ (n_components,) = singular_values^2 / (n - 1).
    Methods: fit(X), transform(X) -> (n, n_components).

    Fit on the TRAINING fold only. Signs of components are arbitrary (the tests compare up to sign).

    TODO (Week 2, Tuesday).
    Done when tests/test_models_vs_sklearn.py::test_pca_matches_sklearn passes.
    """

    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, X):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError


class PCRegression:
    """Principal component regression: PCAScratch followed by least squares (or ridge) on the scores.

    fit(X, y) -> self, predict(X). Choose n_components inside the CV folds.

    TODO (Week 2, Tuesday).
    """

    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
