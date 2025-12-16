from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix


def time_to_flag(
    predicted_regimes: pd.Series,
    event_start: str,
    flag_label: int = 2,  # Stress
    max_lookahead: int = 20,
) -> int | None:
    """
    Trading-day-based lag:
    - Find first prediction date >= event_start
    - Scan forward by position up to max_lookahead for flag_label
    """
    s = predicted_regimes.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    start = pd.Timestamp(event_start)

    pos0 = s.index.searchsorted(start)
    if pos0 >= len(s):
        return None

    end_pos = min(len(s), pos0 + max_lookahead + 1)
    window = s.iloc[pos0:end_pos]
    hits = np.where(window.values == flag_label)[0]
    return int(hits[0]) if len(hits) else None


def false_alarms_pre_event(
    predicted_regimes: pd.Series,
    event_start: str,
    flag_label: int = 2,
    lookback_trading_days: int = 30,
) -> int | None:
    """
    Count Stress predictions in the lookback_trading_days immediately preceding t0,
    where t0 is the first trading date >= event_start.
    """
    s = predicted_regimes.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    start = pd.Timestamp(event_start)

    pos0 = s.index.searchsorted(start)
    if pos0 >= len(s):
        return None

    start_pos = max(0, pos0 - lookback_trading_days)
    window = s.iloc[start_pos:pos0]
    return int((window.values == flag_label).sum())


def evaluate_crisis_events(
    predicted_regimes: pd.Series,
    crisis_events: Dict[str, Tuple[str, str]],
    predicted_probs: pd.DataFrame | None = None,
    flag_label: int = 2,
) -> pd.DataFrame:
    """
    For each crisis event:
      - detection_lag (trading days after start)
      - false_alarms_30d (count of Stress in prior 30 trading days)
      - confidence_at_flag (optional) = p_flag_label on first flag day
    """
    s = predicted_regimes.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()

    probs = None
    if predicted_probs is not None:
        probs = predicted_probs.copy()
        probs.index = pd.to_datetime(probs.index)
        probs = probs.sort_index()

    rows = []
    for name, (start, _end) in crisis_events.items():
        lag = time_to_flag(s, start, flag_label=flag_label)
        fa = false_alarms_pre_event(s, start, flag_label=flag_label, lookback_trading_days=30)

        conf = None
        if lag is not None and probs is not None:
            pos0 = s.index.searchsorted(pd.Timestamp(start))
            flag_pos = pos0 + lag
            if 0 <= flag_pos < len(s):
                flag_date = s.index[flag_pos]
                col = f"p_{flag_label}"
                if col in probs.columns and flag_date in probs.index:
                    conf = float(probs.loc[flag_date, col])

        rows.append(
            {
                "event": name,
                "start": start,
                "detection_lag_days": lag,
                "false_alarms_30d": fa,
                "confidence_at_flag": conf,
            }
        )

    return pd.DataFrame(rows)


def evaluate_classifier(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict:
    """
    Standard classification metrics (canonical labels 0/1/2).
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, output_dict=True),
    }
    return metrics


