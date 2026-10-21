import numpy as np
import pytest

from quantml.models.decomp import PCAScratch
from quantml.models.kernel import KernelRidgeScratch
from quantml.models.linear import LassoScratch, RidgeScratch


@pytest.fixture
def data(rng):
    X = rng.normal(size=(120, 6)) + rng.normal(size=6)
    w = np.array([1.5, 0.0, -2.0, 0.0, 0.5, 0.0])
    y = X @ w + 0.7 + rng.normal(scale=0.5, size=120)
    return X, y


def test_ridge_matches_sklearn(data):
    from sklearn.linear_model import Ridge
    X, y = data
    mine = RidgeScratch(alpha=3.0).fit(X, y)
    ref = Ridge(alpha=3.0).fit(X, y)
    np.testing.assert_allclose(mine.coef_, ref.coef_, atol=1e-8)
    assert abs(mine.intercept_ - ref.intercept_) < 1e-8
    np.testing.assert_allclose(mine.predict(X), ref.predict(X), atol=1e-8)


def test_lasso_matches_sklearn(data):
    from sklearn.linear_model import Lasso
    X, y = data
    mine = LassoScratch(alpha=0.1).fit(X, y)
    ref = Lasso(alpha=0.1, tol=1e-12, max_iter=100000).fit(X, y)
    np.testing.assert_allclose(mine.coef_, ref.coef_, atol=1e-5)
    assert abs(mine.intercept_ - ref.intercept_) < 1e-5


def test_pca_matches_sklearn(data):
    from sklearn.decomposition import PCA
    X, _ = data
    mine = PCAScratch(n_components=3).fit(X)
    ref = PCA(n_components=3).fit(X)
    np.testing.assert_allclose(mine.explained_variance_, ref.explained_variance_, atol=1e-8)
    for a, b in zip(mine.components_, ref.components_):
        assert min(np.abs(a - b).max(), np.abs(a + b).max()) < 1e-8  # equal up to sign


def test_kernel_ridge_matches_sklearn(data):
    from sklearn.kernel_ridge import KernelRidge
    X, y = data
    mine = KernelRidgeScratch(alpha=0.5, gamma=0.1).fit(X, y)
    ref = KernelRidge(kernel="rbf", alpha=0.5, gamma=0.1).fit(X, y)
    np.testing.assert_allclose(mine.predict(X[:20]), ref.predict(X[:20]), atol=1e-8)
