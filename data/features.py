from __future__ import annotations

import numpy as np
import pandas as pd


def compute_avg_pairwise_corr(sector_returns: pd.DataFrame, window: int) -> pd.Series:
    """
    Average of upper-triangle correlations for each date using trailing window.
    Returns a Series indexed by date where enough history exists.
    """
    if window < 2:
        raise ValueError("window must be >= 2")

    sector_returns = sector_returns.copy()
    sector_returns.index = pd.to_datetime(sector_returns.index)
    sector_returns = sector_returns.sort_index()
    sector_returns = sector_returns.dropna(how="any")

    dates = sector_returns.index
    out = pd.Series(index=dates, dtype=float, name=f"avg_corr_{window}")

    for i in range(window, len(dates)):
        chunk = sector_returns.iloc[i - window : i]
        corr = chunk.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        out.iloc[i] = upper.stack().mean()

    return out.dropna()


def build_features(
    returns: pd.DataFrame,
    vix: pd.Series,
    market_ticker: str,
    sector_tickers: tuple[str, ...],
    fed_funds: pd.Series | None = None,
    corr_window: int = 20,
    vol_window: int = 20,
    vol_long_window: int = 60,
    skew_window: int = 60,
) -> pd.DataFrame:
    """
    Build the MVP feature set as defined in SPEC.md.

    Inputs must already be aligned to the same trading calendar (same index).
    """
    idx = pd.to_datetime(returns.index)
    r = returns.copy()
    r.index = idx
    r = r.sort_index()

    vix = vix.copy()
    vix.index = pd.to_datetime(vix.index)
    vix = vix.reindex(idx).ffill()

    f = pd.DataFrame(index=idx)

    mkt = r[market_ticker]

    f["vol_20d"] = mkt.rolling(vol_window).std() * np.sqrt(252)
    f["vol_60d"] = mkt.rolling(vol_long_window).std() * np.sqrt(252)
    f["ret_20d"] = mkt.rolling(vol_window).mean() * 252
    f["ret_skew_60d"] = mkt.rolling(skew_window).skew()
    f["vix_level"] = vix

    sec = r[list(sector_tickers)].copy()
    f["avg_correlation"] = compute_avg_pairwise_corr(sec, window=corr_window).reindex(idx)

    if fed_funds is not None:
        ff = fed_funds.copy()
        ff.index = pd.to_datetime(ff.index)
        f["fed_funds"] = ff.reindex(idx).ffill()

    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    return f


