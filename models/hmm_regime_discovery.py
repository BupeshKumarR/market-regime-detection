from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


CANONICAL_LABELS = {0: "Risk-On", 1: "Risk-Off", 2: "Stress"}


@dataclass(frozen=True)
class HMMFoldResult:
    fold: int
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex
    train_regimes: pd.Series  # canonical ints (0/1/2)
    test_regimes: pd.Series   # canonical ints (0/1/2)
    test_regime_probs: pd.DataFrame  # columns: p_0, p_1, p_2 (canonical order)


def _require_hmmlearn():
    try:
        from hmmlearn import hmm  # type: ignore
    except ImportError as e:
        raise ImportError("hmmlearn not installed; add it to requirements.txt to use HMM regime discovery.") from e
    return hmm


def canonical_state_mapping_by_vol(
    train_features: pd.DataFrame,
    train_states: np.ndarray,
    vol_col: str = "vol_20d",
) -> Dict[int, int]:
    """
    Map raw HMM state ids -> canonical labels (0 Risk-On, 1 Risk-Off, 2 Stress)
    using ONLY train-window statistics (leakage-safe).
    """
    if vol_col not in train_features.columns:
        raise ValueError(f"Missing required column {vol_col!r} for regime naming.")

    s = pd.Series(train_states, index=train_features.index, name="state")
    means = train_features.groupby(s)[vol_col].mean().sort_values()
    ordered_states = list(means.index)
    if len(ordered_states) != 3:
        raise ValueError("This project MVP assumes exactly 3 regimes.")

    return {int(ordered_states[0]): 0, int(ordered_states[1]): 1, int(ordered_states[2]): 2}


def _map_states(states: np.ndarray, mapping: Dict[int, int], index: pd.DatetimeIndex) -> pd.Series:
    mapped = np.array([mapping[int(x)] for x in states], dtype=int)
    return pd.Series(mapped, index=index, name="regime")


def _reorder_probs_to_canonical(raw_probs: np.ndarray, mapping: Dict[int, int], index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    raw_probs columns are raw-state order. We reorder to canonical columns: p_0, p_1, p_2.
    """
    n = raw_probs.shape[1]
    if n != 3:
        raise ValueError("This project MVP assumes exactly 3 regimes.")

    # mapping: raw_state -> canonical_label
    # we need inverse: canonical_label -> raw_state
    inv = {canon: raw for raw, canon in mapping.items()}
    probs_canon = np.column_stack([raw_probs[:, inv[0]], raw_probs[:, inv[1]], raw_probs[:, inv[2]]])
    return pd.DataFrame(probs_canon, index=index, columns=["p_0", "p_1", "p_2"])


def fit_hmm_on_train_predict_train_test(
    features: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_regimes: int = 3,
    n_iter: int = 1000,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Leakage-safe fold procedure:
    - fit scaler + HMM on train only
    - predict states and probs for train and test
    - map raw states to canonical labels using train-vol ordering
    - return (train_regimes, test_regimes, test_probs_canonical)
    """
    hmm = _require_hmmlearn()

    X_train = features.iloc[train_idx]
    X_test = features.iloc[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X_train_s)

    train_states = model.predict(X_train_s)
    test_states = model.predict(X_test_s)
    test_probs = model.predict_proba(X_test_s)

    mapping = canonical_state_mapping_by_vol(X_train, train_states, vol_col="vol_20d")
    train_regimes = _map_states(train_states, mapping, index=X_train.index)
    test_regimes = _map_states(test_states, mapping, index=X_test.index)
    test_probs_canon = _reorder_probs_to_canonical(test_probs, mapping, index=X_test.index)

    return train_regimes, test_regimes, test_probs_canon


def walk_forward_hmm_labels(
    features: pd.DataFrame,
    n_splits: int = 5,
    test_size: int = 252,
    n_regimes: int = 3,
    n_iter: int = 1000,
    random_state: int = 42,
) -> List[HMMFoldResult]:
    """
    Produce fold-by-fold canonical regime labels for train/test windows.
    This does NOT attempt to stitch a single 'global' regime series; consumers should
    use test outputs for out-of-sample evaluation.
    """
    features = features.copy()
    features.index = pd.to_datetime(features.index)
    features = features.sort_index()

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    out: List[HMMFoldResult] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(features)):
        train_reg, test_reg, test_probs = fit_hmm_on_train_predict_train_test(
            features=features,
            train_idx=train_idx,
            test_idx=test_idx,
            n_regimes=n_regimes,
            n_iter=n_iter,
            random_state=random_state,
        )

        out.append(
            HMMFoldResult(
                fold=fold,
                train_index=train_reg.index,
                test_index=test_reg.index,
                train_regimes=train_reg,
                test_regimes=test_reg,
                test_regime_probs=test_probs,
            )
        )

    return out


def regimes_to_names(regimes: pd.Series) -> pd.Series:
    s = regimes.map(CANONICAL_LABELS).astype("category")
    s.name = "regime_name"
    return s


def summarize_regimes(features: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    """
    Simple sanity-check table: per-regime means of each feature.
    Useful for validating that Stress has high vol/corr/VIX, etc.
    """
    df = features.join(regimes.rename("regime")).dropna()
    summary = df.groupby("regime").mean(numeric_only=True)
    summary.index = summary.index.map(lambda x: CANONICAL_LABELS.get(int(x), str(x)))
    return summary


