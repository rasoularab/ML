"""Portfolio construction and costs (Week 8). Stubs only for now."""
from __future__ import annotations


def decile_long_short(pred, n_groups: int = 10):
    """Equal-weight long the top group and short the bottom group each date. TODO (Week 8)."""
    raise NotImplementedError


def turnover(weights):
    """Average one-way turnover per period from a weights DataFrame (dates x assets). TODO (Week 8)."""
    raise NotImplementedError


def apply_costs(gross_returns, turnover_series, cost_bps: float):
    """Net returns = gross returns - turnover * cost_bps / 10000. TODO (Week 8)."""
    raise NotImplementedError
