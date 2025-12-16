from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import CFG
from data.download import (
    align_on_common_dates,
    compute_returns,
    download_vix,
    download_yf_prices,
    forward_fill_to_calendar,
    load_fred_series,
)
from data.features import build_features
from models.evaluation import evaluate_crisis_events
from models.hmm_regime_discovery import summarize_regimes, walk_forward_hmm_labels
from models.supervised_classifier import concat_oos_predictions, fold_metrics_table, walk_forward_rf


@dataclass(frozen=True)
class PipelineOutputs:
    features: pd.DataFrame
    rf_y_true: pd.Series
    rf_y_pred: pd.Series
    rf_y_proba: pd.DataFrame
    rf_fold_metrics: pd.DataFrame
    crisis_metrics: pd.DataFrame


def run_pipeline(save_dir: str | Path = "results") -> PipelineOutputs:
    """
    End-to-end run:
      - download data
      - compute features
      - walk-forward HMM labels
      - walk-forward RF (t -> t+5)
      - event-based crisis evaluation

    Writes CSV outputs to results/metrics/.
    """
    save_dir = Path(save_dir)
    metrics_dir = save_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    tickers = [CFG.market_ticker, *CFG.sector_tickers]
    prices = download_yf_prices(tickers, start=CFG.start_date, end=CFG.end_date, prefer_adj_close=True)
    vix = download_vix(start=CFG.start_date, end=CFG.end_date)

    # Macro is optional
    fed_funds = None
    try:
        ff = load_fred_series(CFG.fred_fed_funds_series, start=CFG.start_date, end=CFG.end_date)
        fed_funds = ff
    except Exception:
        fed_funds = None

    prices, vix = align_on_common_dates(prices, vix)
    rets = compute_returns(prices)
    # Align returns and VIX again (returns drops first row)
    rets, vix = align_on_common_dates(rets, vix)

    if fed_funds is not None:
        fed_funds = forward_fill_to_calendar(fed_funds, rets.index)

    features = build_features(
        returns=rets,
        vix=vix,
        market_ticker=CFG.market_ticker,
        sector_tickers=CFG.sector_tickers,
        fed_funds=fed_funds,
        corr_window=CFG.corr_window,
        vol_window=CFG.vol_window,
        vol_long_window=CFG.vol_long_window,
        skew_window=CFG.skew_window,
    )

    hmm_folds = walk_forward_hmm_labels(
        features=features,
        n_splits=CFG.n_splits,
        test_size=CFG.test_size,
        n_regimes=CFG.n_regimes,
        n_iter=CFG.hmm_n_iter,
        random_state=CFG.hmm_random_state,
    )

    rf_folds = walk_forward_rf(
        features=features,
        hmm_folds=hmm_folds,
        horizon=CFG.horizon_days,
        n_estimators=CFG.rf_n_estimators,
        max_depth=CFG.rf_max_depth,
        random_state=CFG.rf_random_state,
    )

    rf_y_true, rf_y_pred, rf_y_proba = concat_oos_predictions(rf_folds)
    rf_fold_metrics = fold_metrics_table(rf_folds)

    crisis_metrics = evaluate_crisis_events(
        predicted_regimes=rf_y_pred,
        predicted_probs=rf_y_proba,
        crisis_events=CFG.crisis_events,
        flag_label=2,
    )

    # Save metrics
    rf_fold_metrics.to_csv(metrics_dir / "rf_fold_metrics.csv", index=False)
    crisis_metrics.to_csv(metrics_dir / "crisis_metrics.csv", index=False)

    # Save a time series table for dashboard plotting (OOS only)
    spy_price = prices[CFG.market_ticker].reindex(rf_y_pred.index)
    ts = pd.DataFrame(
        {
            "spy_price": spy_price,
            "rf_pred_regime": rf_y_pred,
            "p_risk_on": rf_y_proba["p_0"],
            "p_risk_off": rf_y_proba["p_1"],
            "p_stress": rf_y_proba["p_2"],
        },
        index=rf_y_pred.index,
    )
    ts.to_csv(metrics_dir / "oos_timeseries.csv")

    # Helpful sanity table for the last fold's train regime stats
    last_fold = hmm_folds[-1]
    regime_summary = summarize_regimes(features.loc[last_fold.train_index], last_fold.train_regimes)
    regime_summary.to_csv(metrics_dir / "hmm_regime_summary_last_train_fold.csv")

    return PipelineOutputs(
        features=features,
        rf_y_true=rf_y_true,
        rf_y_pred=rf_y_pred,
        rf_y_proba=rf_y_proba,
        rf_fold_metrics=rf_fold_metrics,
        crisis_metrics=crisis_metrics,
    )


if __name__ == "__main__":
    run_pipeline()


