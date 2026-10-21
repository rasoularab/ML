"""Data loading. All I/O lives here so the rest of the code works on in-memory pandas objects."""
from __future__ import annotations
from pathlib import Path 
import pandas as pd
import numpy as np

  # add at the top of the file with the other imports

def load_prices(tickers, start, end, cache_dir="data/raw") -> pd.DataFrame:
    """...daily adjusted close prices, wide format..."""

    # 1. make tickers a list
    tickers = [tickers] if isinstance(tickers, str) else list(tickers)                  # hint: [tickers] if isinstance(tickers, str) else list(tickers)

    # 2. cache file path (changes when tickers or dates change)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"prices_{'-'.join(tickers)}_{start}_{end}.csv"

    # 3. read from cache, or download
    if cache_file.exists():
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)   # fill in: the file
    else:
        import yfinance as yf
        raw = yf.download(tickers, start=start, end=pd.Timestamp(end) + pd.Timedelta(days=1), auto_adjust=True, progress=False)
        #   end is exclusive in yfinance: pass end plus one day (pd.Timestamp(end) + pd.Timedelta(days=1))
        prices = raw['Close']              # which column holds the adjusted price once auto_adjust=True?
        prices.to_csv(cache_file)             # save to the cache file

    # 4. clean
    prices = prices.sort_index().dropna(how="all")

    # 5. drop tickers with too many missing days, and say which
    missing = prices.isna().mean()          # share of missing days per ticker
    dropped = missing[missing > 0.05]
    if len(dropped) > 0:
        print("Dropped:", list(dropped.index))
    prices = prices.drop(columns=dropped.index)

    return prices



def to_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """Daily simple returns r_t = P_t / P_{t-1} - 1 (or log returns). The first row is dropped."""
    if log:
        out = np.log(prices).diff()
    else:
        out = prices.pct_change(fill_method=None)
    return out.iloc[1:]


# def load_panel(path) -> pd.DataFrame:
#     """Load a cross-sectional panel with a MultiIndex named ['date', 'asset'], sorted.

#     TODO (Week 2): needed for the cross-sectional projects (stock characteristics + returns).
#     """
#     raise NotImplementedError
