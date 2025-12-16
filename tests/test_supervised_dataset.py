import pandas as pd

from models.supervised_classifier import create_supervised_dataset


def test_create_supervised_dataset_alignment_and_drop_tail():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    X = pd.DataFrame({"a": range(10)}, index=idx)
    y = pd.Series([0, 0, 1, 1, 2, 2, 0, 0, 1, 1], index=idx)

    X2, y2 = create_supervised_dataset(X, y, horizon=3)
    assert (X2.index == y2.index).all()
    assert len(X2) == 7  # last 3 dropped
    # y2[0] should equal y[3]
    assert y2.iloc[0] == y.iloc[3]


