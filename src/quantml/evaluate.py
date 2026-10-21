"""Evaluation metrics and the trial log."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def oos_r2(y_true, y_pred, y_bench=None) -> float:
    """Out-of-sample R-squared: 1 - sum((y - yhat)^2) / sum((y - b)^2).

    `y_bench` is the benchmark forecast; if None the benchmark is zero (the Gu-Kelly-Xiu convention for
    individual stock returns). For aggregate series pass the prevailing-mean forecast instead.

    TODO (Week 1, Friday).
    Done when tests/test_evaluate.py passes.
    """
    raise NotImplementedError


def information_coefficient(pred: pd.Series, realized: pd.Series) -> pd.Series:
    """Per-date Spearman correlation between prediction and realized return.

    Inputs have a MultiIndex ['date', 'asset']; returns a Series indexed by date.
    TODO (Week 2).
    """
    raise NotImplementedError


def sharpe(returns, periods_per_year: int) -> float:
    """Annualized Sharpe ratio of a return series (mean / std * sqrt(periods_per_year)); zero risk-free rate.

    TODO (Week 2).
    """
    raise NotImplementedError


def probabilistic_sharpe(returns, benchmark_sr: float = 0.0) -> float:
    """Probabilistic Sharpe ratio (AFML ch 14). TODO (Week 10)."""
    raise NotImplementedError


def deflated_sharpe(returns, n_trials: int) -> float:
    """Deflated Sharpe ratio: adjusts for the number of trials, skewness, kurtosis and sample length (AFML ch 14).
    TODO (Week 10). Use ExperimentLog.n_trials() for n_trials.
    """
    raise NotImplementedError


class ExperimentLog:
    """Append-only JSON-lines log of every configuration you evaluate.

    This is infrastructure, not an exercise. Call `record(...)` every time you evaluate a model or
    parameter setting out of sample. `n_trials()` feeds the deflated Sharpe ratio in Week 10.
    """

    def __init__(self, path="experiments/log.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, **fields) -> None:
        row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
        with self.path.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def n_trials(self, project: str | None = None) -> int:
        if not self.path.exists():
            return 0
        n = 0
        with self.path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if project is None or row.get("project") == project:
                    n += 1
        return n
