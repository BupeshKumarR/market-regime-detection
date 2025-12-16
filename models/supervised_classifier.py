from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from models.hmm_regime_discovery import HMMFoldResult


@dataclass(frozen=True)
class RFFoldResult:
    fold: int
    test_index: pd.DatetimeIndex
    y_true: pd.Series
    y_pred: pd.Series
    y_pred_proba: pd.DataFrame  # columns: p_0, p_1, p_2
    metrics: dict


def create_supervised_dataset(
    features: pd.DataFrame,
    regimes: pd.Series,
    horizon: int = 5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create X_t -> y_{t+h} dataset with strict alignment.
    """
    y_future = regimes.shift(-horizon)
    valid = ~y_future.isna()
    X = features.loc[valid].copy()
    y = y_future.loc[valid].astype(int).copy()
    if not (X.index == y.index).all():
        raise ValueError("Index mismatch between features and target.")
    return X, y


def _fit_predict_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_estimators: int = 200,
    max_depth: int = 8,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.DataFrame, RandomForestClassifier]:
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    pred = pd.Series(clf.predict(X_test), index=X_test.index, name="y_pred").astype(int)
    proba = clf.predict_proba(X_test)

    # Ensure columns are in class order 0/1/2. sklearn uses clf.classes_.
    cols = list(clf.classes_)
    proba_df = pd.DataFrame(proba, index=X_test.index, columns=[f"p_{c}" for c in cols])
    for c in [0, 1, 2]:
        if f"p_{c}" not in proba_df.columns:
            proba_df[f"p_{c}"] = 0.0
    proba_df = proba_df[["p_0", "p_1", "p_2"]]

    return pred, proba_df, clf


def walk_forward_rf(
    features: pd.DataFrame,
    hmm_folds: List[HMMFoldResult],
    horizon: int = 5,
    n_estimators: int = 200,
    max_depth: int = 8,
    random_state: int = 42,
) -> List[RFFoldResult]:
    """
    For each fold:
      - Build train dataset from (features[train], regimes_train) predicting t+5 within train
      - Evaluate on test dataset predicting t+5 within test
    Returns fold results; concatenate test predictions for OOS series.
    """
    results: List[RFFoldResult] = []
    features = features.copy()
    features.index = pd.to_datetime(features.index)
    features = features.sort_index()

    for fold_res in hmm_folds:
        X_train_full = features.loc[fold_res.train_index]
        r_train_full = fold_res.train_regimes.loc[fold_res.train_index]
        X_train, y_train = create_supervised_dataset(X_train_full, r_train_full, horizon=horizon)

        X_test_full = features.loc[fold_res.test_index]
        r_test_full = fold_res.test_regimes.loc[fold_res.test_index]
        X_test, y_test = create_supervised_dataset(X_test_full, r_test_full, horizon=horizon)

        y_pred, y_pred_proba, _clf = _fit_predict_rf(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "n_test": int(len(y_test)),
        }

        results.append(
            RFFoldResult(
                fold=fold_res.fold,
                test_index=y_test.index,
                y_true=y_test,
                y_pred=y_pred,
                y_pred_proba=y_pred_proba,
                metrics=metrics,
            )
        )

    return results


def concat_oos_predictions(rf_folds: List[RFFoldResult]) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Concatenate out-of-sample predictions across folds.
    Returns (y_true, y_pred, y_pred_proba) indexed by date.
    """
    y_true = pd.concat([r.y_true for r in rf_folds]).sort_index()
    y_pred = pd.concat([r.y_pred for r in rf_folds]).sort_index()
    y_proba = pd.concat([r.y_pred_proba for r in rf_folds]).sort_index()
    return y_true, y_pred, y_proba


def fold_metrics_table(rf_folds: List[RFFoldResult]) -> pd.DataFrame:
    """
    Convenience: one-row-per-fold metrics table.
    """
    rows = []
    for r in rf_folds:
        row = {"fold": r.fold}
        row.update(r.metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fold")


