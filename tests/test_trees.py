import numpy as np

from quantml.models.trees import GradientBoostingStumps, RegressionTree, best_split


def test_best_split_finds_step():
    x = np.arange(20, dtype=float)
    y = np.where(x < 12, 0.0, 5.0)
    thr, gain = best_split(x, y)
    assert 11 <= thr <= 12
    assert gain > 0


def test_tree_depth1_matches_sklearn(rng):
    from sklearn.tree import DecisionTreeRegressor
    X = rng.normal(size=(200, 3))
    y = np.where(X[:, 1] > 0.2, 1.0, -1.0) + rng.normal(scale=0.3, size=200)
    mine = RegressionTree(max_depth=1).fit(X, y)
    ref = DecisionTreeRegressor(max_depth=1).fit(X, y)
    np.testing.assert_allclose(mine.predict(X), ref.predict(X), atol=1e-10)


def test_boosting_reduces_train_loss(rng):
    X = rng.normal(size=(300, 4))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + rng.normal(scale=0.1, size=300)
    m = GradientBoostingStumps(n_rounds=50, learning_rate=0.2).fit(X, y)
    assert m.train_loss_[-1] < m.train_loss_[0]
    assert all(b <= a + 1e-12 for a, b in zip(m.train_loss_, m.train_loss_[1:]))
