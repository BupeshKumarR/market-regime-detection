import pandas as pd

from models.evaluation import false_alarms_pre_event, time_to_flag


def test_time_to_flag_trading_day_offsets():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    pred = pd.Series([0, 0, 2, 2], index=idx)  # Stress at 2020-01-06
    # event starts on a weekend -> first trading day after is 2020-01-06 (lag 0)
    assert time_to_flag(pred, "2020-01-04", flag_label=2, max_lookahead=5) == 0
    # event start at 2020-01-03 -> Stress at 2020-01-06 is 1 trading day after in this index slice
    assert time_to_flag(pred, "2020-01-03", flag_label=2, max_lookahead=5) == 1


def test_false_alarms_pre_event_counts_prior_window():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    pred = pd.Series([0, 2, 0, 2, 0, 0, 0, 0, 0, 0], index=idx)
    # event_start is idx[6]; look back 5 trading days => positions 1..5 (exclusive of idx[6])
    fa = false_alarms_pre_event(pred, str(idx[6].date()), flag_label=2, lookback_trading_days=5)
    assert fa == 2


