from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Date range
    start_date: str = "2014-01-01"
    end_date: str = "2024-12-01"

    # Universe
    market_ticker: str = "SPY"
    sector_tickers: tuple[str, ...] = ("XLF", "XLE", "XLK", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB")
    vix_ticker: str = "^VIX"

    # Macro (optional)
    fred_fed_funds_series: str = "DFF"

    # Feature windows
    corr_window: int = 20
    vol_window: int = 20
    vol_long_window: int = 60
    skew_window: int = 60

    # Regimes
    n_regimes: int = 3  # Risk-On / Risk-Off / Stress

    # Supervised horizon
    horizon_days: int = 5

    # Walk-forward evaluation
    n_splits: int = 5
    test_size: int = 252  # ~1 year

    # Model hyperparameters
    hmm_n_iter: int = 1000
    hmm_random_state: int = 42
    rf_n_estimators: int = 200
    rf_max_depth: int = 8
    rf_random_state: int = 42

    # Crisis events (start, end)
    crisis_events: dict[str, tuple[str, str]] = None


CFG = Config(
    crisis_events={
        "COVID Crash": ("2020-02-19", "2020-03-23"),
        "2022 Bear Market": ("2022-01-03", "2022-06-16"),
        "Dec 2018 Selloff": ("2018-10-03", "2018-12-24"),
        "2015 China Devaluation": ("2015-08-10", "2015-08-25"),
    }
)


