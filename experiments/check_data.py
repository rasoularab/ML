import sys; sys.path.insert(0, "src")
from quantml.data import load_prices
import pandas as pd
import numpy as np

p = load_prices("SPY", "2010-01-01", "2025-12-31")
print(p.shape)
print(p.columns)
print(p.index[0], p.index[-1])
print(p.isna().sum())
print(p.head())

from quantml.data import load_prices, to_returns
r = to_returns(p)["SPY"]
print(len(r), len(p) - 1)
print(r.describe())
print(r.abs().max())