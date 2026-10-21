"""The seven-step recipe as one function. Fill it in as you learn each piece (Weeks 1-4)."""
from __future__ import annotations

import yaml

from quantml import cv, data, evaluate, features, labels


def run_experiment(config_path: str):
    """Skeleton of every experiment. Read a YAML config and run the recipe end to end.

    Steps (see README.md):
      1. Question and timing         -> config['target'], docs/CONVENTIONS.md
      2. Data                        -> data.load_prices, data.to_returns
      3. Features and target         -> features.*, labels.forward_return (lags >= 1 only)
      4. Split                       -> cv.WalkForward or cv.PurgedKFold
      5. Baseline first, then models -> models.base.ZeroForecast, then OLS, then yours
      6. Evaluate out of sample      -> evaluate.oos_r2, information_coefficient, sharpe
      7. Log                         -> evaluate.ExperimentLog.record(project=..., model=..., oos_r2=...)

    TODO (Week 1 Saturday for the single-asset version; Week 4 for the panel version).
    """
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    raise NotImplementedError(f"Fill in the recipe for project {cfg.get('project')}")


if __name__ == "__main__":
    import sys

    run_experiment(sys.argv[1])
