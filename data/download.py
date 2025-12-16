from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _ensure_datetime_index(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    out = obj.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out


def download_yf_prices(
    tickers: Iterable[str],
    start: str,
    end: str,
    prefer_adj_close: bool = True,
) -> pd.DataFrame:
    """
    Download close prices for multiple tickers via yfinance.

    Returns a wide DataFrame:
      - index: trading dates
      - columns: tickers
      - values: Adj Close if available else Close
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("yfinance not installed; add it to requirements.txt to use Yahoo Finance downloads.") from e

    tickers = list(tickers)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)

    # Multiple tickers -> MultiIndex columns: (field, ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        fields = set(raw.columns.get_level_values(0))
        field = "Adj Close" if (prefer_adj_close and "Adj Close" in fields) else "Close"
        prices = raw[field].copy()
    else:
        # Single ticker -> flat columns
        if prefer_adj_close and "Adj Close" in raw.columns:
            prices = raw[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        else:
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = _ensure_datetime_index(prices)
    prices = prices.dropna(how="all")
    return prices


def download_vix(start: str, end: str) -> pd.Series:
    """
    Download VIX as a Series (prefers Close).
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("yfinance not installed; add it to requirements.txt to use Yahoo Finance downloads.") from e

    vix_df = yf.download("^VIX", start=start, end=end, progress=False)
    vix_df = _ensure_datetime_index(vix_df)

    if "Close" in vix_df.columns:
        vix = vix_df["Close"].copy()
    elif "Adj Close" in vix_df.columns:
        vix = vix_df["Adj Close"].copy()
    else:
        vix = vix_df.iloc[:, 0].copy()

    vix.name = "VIX"
    return vix.dropna()


def align_on_common_dates(*objs: pd.DataFrame | pd.Series):
    """
    Align DataFrames/Series to intersection of datetime indices.
    """
    idx = None
    for o in objs:
        o_idx = pd.to_datetime(o.index)
        idx = o_idx if idx is None else idx.intersection(o_idx)

    aligned = []
    for o in objs:
        aligned.append(_ensure_datetime_index(o).loc[idx].copy())
    return aligned


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = _ensure_datetime_index(prices)
    rets = prices.pct_change()
    rets = rets.replace([np.inf, -np.inf], np.nan)
    return rets.dropna(how="all")


def load_fred_series(series_id: str, start: str, end: str, api_key: Optional[str] = None) -> pd.Series:
    """
    Load a FRED series using fredapi.

    Requires:
      - `fredapi` installed
      - env var FRED_API_KEY or api_key passed
    """
    try:
        from fredapi import Fred
    except ImportError as e:
        raise ImportError("fredapi not installed; add it to requirements.txt to use FRED features.") from e

    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise ValueError("Missing FRED API key. Set FRED_API_KEY or pass api_key=...")

    fred = Fred(api_key=key)
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    s = pd.Series(s)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s.name = series_id
    return s


def forward_fill_to_calendar(series: pd.Series, calendar_index: pd.DatetimeIndex) -> pd.Series:
    series = _ensure_datetime_index(series)
    out = series.reindex(calendar_index, method="ffill")
    out.name = series.name
    return out


