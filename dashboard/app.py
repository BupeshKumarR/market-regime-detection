from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline import run_pipeline


RESULTS_DIR = Path("results/metrics")


def _load_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_timeseries() -> pd.DataFrame:
    path = RESULTS_DIR / "oos_timeseries.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


st.set_page_config(page_title="Market Regime Detection", layout="wide")
st.title("Market Regime Detection (MVP)")
st.caption("Specs-first build. 3 regimes: Risk-On / Risk-Off / Stress. Prediction horizon: t+5 trading days.")

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Run / Refresh")
    st.write("If metrics files are missing, run the pipeline to generate them.")
    if st.button("Run pipeline (downloads data, trains models, writes results)"):
        try:
            run_pipeline(save_dir="results")
            st.success("Pipeline completed. Results written to results/metrics/.")
        except Exception as e:
            st.error(f"Pipeline failed: {e}")

with colB:
    st.subheader("Artifacts")
    st.write(f"Looking for metrics in `{RESULTS_DIR}`")

crisis_df = _load_csv("crisis_metrics.csv")
fold_df = _load_csv("rf_fold_metrics.csv")
ts = _load_timeseries()

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Crisis event metrics (OOS predictions)")
    if crisis_df.empty:
        st.info("No `crisis_metrics.csv` found yet. Run pipeline.")
    else:
        st.dataframe(crisis_df, use_container_width=True)

with right:
    st.subheader("RF walk-forward metrics")
    if fold_df.empty:
        st.info("No `rf_fold_metrics.csv` found yet. Run pipeline.")
    else:
        st.dataframe(fold_df, use_container_width=True)

st.divider()
st.subheader("Historical overlay (out-of-sample only)")

if ts.empty:
    st.info("No `oos_timeseries.csv` found yet. Run pipeline.")
else:
    last = ts.iloc[-1]
    st.write(
        f"Latest prediction date: **{ts.index[-1].date()}** | "
        f"Predicted regime(t+5): **{int(last['rf_pred_regime'])}** | "
        f"p(Stress): **{float(last['p_stress']):.2f}**"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts["spy_price"], name="SPY", line=dict(width=2)))

    # Overlay stress points
    stress_mask = ts["rf_pred_regime"] == 2
    fig.add_trace(
        go.Scatter(
            x=ts.index[stress_mask],
            y=ts.loc[stress_mask, "spy_price"],
            mode="markers",
            name="Predicted Stress (t+5)",
            marker=dict(size=5),
        )
    )

    fig.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ts.index, y=ts["p_risk_on"], name="p(Risk-On)"))
    fig2.add_trace(go.Scatter(x=ts.index, y=ts["p_risk_off"], name="p(Risk-Off)"))
    fig2.add_trace(go.Scatter(x=ts.index, y=ts["p_stress"], name="p(Stress)"))
    fig2.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig2, use_container_width=True)


