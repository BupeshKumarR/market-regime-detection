import numpy as np
import pandas as pd

from data.features import build_features, compute_avg_pairwise_corr


def test_compute_avg_pairwise_corr_basic():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    df = pd.DataFrame(
        {
            "A": np.linspace(0, 1, 10),
            "B": np.linspace(0, 1, 10),
            "C": np.linspace(1, 0, 10),
        },
        index=idx,
    )
    s = compute_avg_pairwise_corr(df.pct_change().dropna(), window=3)
    assert isinstance(s, pd.Series)
    assert s.index.is_monotonic_increasing
    assert s.notna().all()


def test_build_features_alignment_and_no_nans():
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    tickers = ["SPY", "XLF", "XLE", "XLK", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB"]

    # Create synthetic returns with small noise
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(len(idx), len(tickers))), index=idx, columns=tickers)

    vix = pd.Series(rng.normal(20, 2, size=len(idx)), index=idx, name="VIX")

    features = build_features(
        returns=returns,
        vix=vix,
        market_ticker="SPY",
        sector_tickers=tuple(tickers[1:]),
        fed_funds=None,
        corr_window=20,
        vol_window=20,
        vol_long_window=60,
        skew_window=60,
    )

    assert isinstance(features, pd.DataFrame)
    assert features.index.is_monotonic_increasing
    assert features.notna().all().all()
    # Must contain required columns
    for col in ["vol_20d", "vol_60d", "ret_20d", "ret_skew_60d", "vix_level", "avg_correlation"]:
        assert col in features.columns


