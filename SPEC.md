## Market Regime Detection — MVP Spec

### Goals
- **Universe**: US equities only using **SPY + 9 sector ETFs + VIX + Fed Funds (DFF)**, daily from **2014-01-01 to 2024-12-01**.
- **Regimes**: exactly **3 economically interpretable regimes**:
  - **Risk-On**: low volatility, positive returns, lower cross-sector correlation
  - **Risk-Off**: moderate volatility, weaker/negative returns, elevated cross-sector correlation
  - **Stress**: highest volatility, drawdowns, cross-sector correlation tends toward 1
- **Offline discovery**: 3-state **Gaussian HMM** fit on features (train window only).
- **Online prediction**: predict **regime at t+5 trading days** from features at **t** using **Random Forest**.
- **Evaluation** (primary): **event-based time-to-flag Stress** on curated crisis events.
- **Demo**: Streamlit app showing current regime prediction, confidence, feature drivers, and historical overlay.

### Non-Goals (MVP)
- Multi-asset regimes, alternative data.
- Ensembles (GMM/clustering), drift detection/retraining automation.
- Real-time streaming feeds or brokerage integrations.

### Data Sources
- **Prices**: Yahoo Finance via `yfinance` (SPY, sector ETFs, ^VIX).
- **Macro**: FRED via `fredapi` for **DFF** (optional if no API key; pipeline should still run without macro).

### Feature Set (minimal, defensible)
Computed on a common trading calendar (intersection of available price dates):
- `vol_20d`: annualized rolling std of SPY returns (20 trading days)
- `vol_60d`: annualized rolling std of SPY returns (60 trading days)
- `ret_20d`: annualized rolling mean of SPY returns (20 trading days)
- `ret_skew_60d`: rolling skew of SPY returns (60 trading days)
- `vix_level`: VIX close level aligned to trading calendar
- `avg_correlation`: average pairwise correlation (upper triangle) of sector ETF returns over a rolling window (20d)
- `fed_funds`: DFF forward-filled to trading calendar (if available)

### Labeling (leakage-safe)
- For any walk-forward split:
  - Fit scaler + HMM **only on train** features.
  - Predict regimes on train and test using the fitted model.
  - Map HMM states to canonical labels **using train stats only**:
    - lowest `vol_20d` → Risk-On (0)
    - middle `vol_20d` → Risk-Off (1)
    - highest `vol_20d` → Stress (2)

### Supervised Target (t → t+5)
- Define `y[t] = regime[t+5]` using the (fold-specific) regime series.
- For a given evaluation window, exclude the final 5 rows (no target available).

### Metrics (exact definitions)
#### Primary: Event-based time-to-flag (trading days)
For each event with **start date S**:
- Let `t0` be the first trading date **on/after S** present in the prediction index.
- **Time-to-flag** = smallest non-negative integer `k` such that `predicted_regime[t0+k] == Stress`, searching up to `k <= 20`.
- If no flag within 20 trading days, return `None`.

#### False alarms (pre-event)
- Count the number of days predicted as Stress in the **30 trading days prior** to `t0` (inclusive of the start boundary as implemented).

#### Secondary: Classification metrics
- Accuracy, balanced accuracy, confusion matrix, per-class precision/recall for {Risk-On, Risk-Off, Stress}.

#### Tertiary (optional): Backtest metrics
- Compare buy-and-hold SPY vs a regime-conditioned risk toggle (scaled exposure) with basic transaction cost assumptions.

### Reproducibility Requirements
- Fixed random seeds for HMM and RF.
- Deterministic feature computation and index alignment.
- A single pipeline run produces saved metrics tables and figures.


