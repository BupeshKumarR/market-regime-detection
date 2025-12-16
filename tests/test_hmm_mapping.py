import numpy as np
import pandas as pd

from models.hmm_regime_discovery import canonical_state_mapping_by_vol, regimes_to_names


def test_canonical_state_mapping_by_vol_orders_by_vol():
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    # Create features where states have different mean vol_20d
    feats = pd.DataFrame(
        {
            "vol_20d": [0.1, 0.1, 0.3, 0.3, 0.9, 0.9],
            "x": [1, 2, 3, 4, 5, 6],
        },
        index=idx,
    )
    states = np.array([2, 2, 0, 0, 1, 1])  # raw state ids
    mapping = canonical_state_mapping_by_vol(feats, states, vol_col="vol_20d")
    # raw state 2 has lowest vol -> canonical 0
    assert mapping[2] == 0
    # raw state 0 is middle -> canonical 1
    assert mapping[0] == 1
    # raw state 1 is highest -> canonical 2
    assert mapping[1] == 2


def test_regimes_to_names():
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    regimes = pd.Series([0, 1, 2], index=idx)
    names = regimes_to_names(regimes)
    assert list(names.astype(str)) == ["Risk-On", "Risk-Off", "Stress"]


