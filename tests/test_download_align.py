import pandas as pd

from data.download import align_on_common_dates


def test_align_on_common_dates_intersection():
    idx1 = pd.date_range("2020-01-01", periods=5, freq="B")
    idx2 = pd.date_range("2020-01-03", periods=5, freq="B")

    s1 = pd.Series(range(len(idx1)), index=idx1, name="s1")
    s2 = pd.Series(range(len(idx2)), index=idx2, name="s2")

    a1, a2 = align_on_common_dates(s1, s2)
    assert (a1.index == a2.index).all()
    assert a1.index.min() == pd.Timestamp("2020-01-03")


