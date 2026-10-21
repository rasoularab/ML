"""Model interface and baselines. Every model: fit(X, y) -> self, predict(X) -> np.ndarray."""
from __future__ import annotations

import numpy as np


class ZeroForecast:
    """WORKED EXAMPLE of the model interface and of the simplest baseline: always predict 0."""

    def fit(self, X, y):
        self.n_features_ = np.asarray(X).shape[1] if np.ndim(X) > 1 else 1
        return self

    def predict(self, X):
        return np.zeros(len(X))


class PrevailingMean:
    """Baseline: predict the mean of the training targets seen so far.

    TODO (Week 1, Thursday). In a walk-forward loop this is the expanding mean of the past.
    """

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
