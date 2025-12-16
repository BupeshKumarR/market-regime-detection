# Market Regime Detection

Specs-first build (see `SPEC.md`). MVP: 3 regimes (Risk-On/Risk-Off/Stress) via walk-forward HMM labels + supervised t→t+5 prediction, evaluated by event-based time-to-flag on known crises.

## Setup
- Create a virtualenv and install dependencies from `requirements.txt`.
- Optional: set `FRED_API_KEY` to enable the Fed Funds feature.

## Project Structure
- `data/`: download + feature engineering
- `models/`: regime discovery + supervised prediction + evaluation
- `dashboard/`: Streamlit demo app (consumes library APIs)
- `tests/`: pytest unit tests for alignment and feature computations
