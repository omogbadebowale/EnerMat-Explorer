# pages/03_Validation.py

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from backend.validate import validate

st.set_page_config(page_title="Validation", page_icon="✅", layout="wide")
st.markdown("## ✅ Validation – Experimental Band-Gap Benchmark")

# ─── Require user upload ───────────────────────────────────────────────────
st.error(
    "⚠️ Please upload a CSV benchmark file. "
    "The built-in 27-point dataset has been disabled."
)
uploaded = st.file_uploader(
    "📥 Upload a CSV benchmark file",
    type=["csv"],
    key="only_csv"
)

# Don’t proceed until they’ve given us a file
if not uploaded:
    st.stop()

# Read their CSV (we no longer accept ODS or default data)
df_exp = pd.read_csv(uploaded)

# ─── Bowing slider + optimise ─────────────────────────────────────────────
if "b_value" not in st.session_state:
    st.session_state.b_value = 0.30

col1, col2 = st.columns([4, 1])
with col1:
    b = st.slider(
        "Bowing parameter *b* (eV)",
        min_value=0.00,
        max_value=1.00,
        value=st.session_state.b_value,
        step=0.01,
    )
    if b != st.session_state.b_value:
        st.session_state.b_value = b
with col2:
    if st.button("🔍 optimise"):
        grid = np.linspace(0.00, 1.00, 51)
        best_b = float(min(grid, key=lambda bb: validate(bb, df_exp)[0]["MAE"]))
        st.session_state.b_value = best_b
        st.success(f"Optimal b ≃ {best_b:.2f} eV")

b = st.session_state.b_value

# ─── Run validation ────────────────────────────────────────────────────────
metrics, resid, skipped = validate(b, df_exp)

# ─── KPI display + CI ─────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("N points", metrics["N"])
c2.metric("MAE (eV)", f"{metrics['MAE']:.3f}")
c3.metric("RMSE (eV)", f"{metrics['RMSE']:.3f}")
c4.metric("R²", f"{metrics['R2']:.3f}")

def bootstrap_ci(y_true, y_pred, n=2000):
    idx = np.random.randint(0, len(y_true), (n, len(y_true)))
    maes = np.abs(y_true[idx] - y_pred[idx]).mean(axis=1)
    return np.percentile(maes, [2.5, 97.5])

lo, hi = bootstrap_ci(
    resid.Eg_eV.to_numpy(dtype=float),
    resid.Eg_pred.to_numpy(dtype=float),
)
st.caption(f"95% CI on MAE: **{lo:.3f}–{hi:.3f} eV**")

# ─── Parity plot ───────────────────────────────────────────────────────────
fig = px.scatter(
    resid, x="Eg_eV", y="Eg_pred",
    hover_data=["Composition", "abs_err"],
    labels={"Eg_eV": "Experimental E₉ (eV)", "Eg_pred": "Predicted E₉ (eV)"},
    height=500,
)
x0, x1 = resid.Eg_eV.min(), resid.Eg_eV.max()
fig.add_shape(type="line", x0=x0, y0=x0, x1=x1, y1=x1,
              line=dict(dash="dash"))
st.plotly_chart(fig, use_container_width=True)

# ─── Residuals table ─────────────────────────────────────────────────────
resid = resid.assign(Outlier=lambda df: df.abs_err > 0.15)
st.markdown("#### Residuals (|error| > 0.15 eV flagged)")
st.dataframe(resid, hide_index=True, height=300)
st.download_button(
    "💾 Download residuals CSV",
    resid.to_csv(index=False).encode(),
    "residuals.csv",
    "text/csv",
)

# ─── Skipped compositions ─────────────────────────────────────────────────
if not skipped.empty:
    with st.expander(f"⚠️ {len(skipped)} skipped rows"):
        st.dataframe(skipped, hide_index=True)
