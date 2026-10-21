"""Week 1, Day 1: SPY returns, 5-day forward label, and the two baseline forecasts.

Run from the repo root:  python experiments/day1_spy_baselines.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from quantml.data import load_prices, to_returns
from quantml.labels import forward_return

TICKER, START, END, H = "SPY", "2010-01-01", "2025-12-31", 5

prices = load_prices(TICKER, START, END)
r = to_returns(prices)[TICKER]
print(f"{TICKER}: {len(r)} daily returns, {r.index[0].date()} to {r.index[-1].date()}")

fig, ax = plt.subplots(figsize=(10, 3))
r.plot(ax=ax, lw=0.5)
ax.set_title(f"{TICKER} daily returns")
fig.tight_layout()
fig.savefig("experiments/spy_returns.png", dpi=120)

# Target: y_t = r_t + ... + r_{t+H-1}
y = forward_return(r, H)

# Baseline 1: zero forecast.
f_zero = y * 0.0
# Baseline 2: prevailing mean. The mean uses returns up to t-1 only (shift(1)), scaled to the horizon.
f_prev = r.expanding().mean().shift(1) * H

ok = y.notna() & f_prev.notna()  # same rows for both baselines
mse_zero = float(np.mean((y[ok] - f_zero[ok]) ** 2))
mse_prev = float(np.mean((y[ok] - f_prev[ok]) ** 2))
print(f"rows scored: {int(ok.sum())}")
print(f"MSE zero forecast    : {mse_zero:.6e}")
print(f"MSE prevailing mean  : {mse_prev:.6e}")
print(f"prevailing mean beats zero: {mse_prev < mse_zero}")
