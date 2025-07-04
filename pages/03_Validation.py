# pages/03_Validation.py

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from backend.validate import validate, load_default_dataset

st.set_page_config(page_title="Validation", page_icon="✅", layout="wide")
st.markdown("## ✅ Validation – Experimental Band-Gap Benchmark")

# ─── 0) Upload or default ───────────────────────────────────────────────
up = st.file_uploader(
    "📥  Upload a CSV or ODS benchmark file "
    "(leave empty to use the built-in 27-point dataset)",
    type=["csv", "ods"],
)
if up:
    df_exp = (
        pd.read_csv(up)
        if up.name.endswith(".csv")
        else pd.read_excel(up, engine="odf")
    )
else:
    df_exp = load_default_dataset()

# ─── 1) Bowing slider + optimise button ─────────────────────────────────
lc, rc = st.columns([4, 1])
with lc:
    b = st.slider("Bowing parameter *b* (eV)", 0.00, 1.00, 0.30, 0.01)
with rc:
    if st.button("🔍 optimise"):
        grid = np.arange(0.00, 1.01, 0.02)
        b_best = float(min(grid, key=lambda bb: validate(bb, df_exp)[0]["MAE"]))
        st.success(f"Optimal b = {b_best:.2f} eV")
        st.experimental_rerun()

# ─── 2) Run validation ─────────────────────────────────────────────────
metrics, resid, skipped = validate(b, df_exp)

# ─── 3) KPI display + bootstrap CI ────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("N points", metrics["N"])
c2.metric("MAE (eV)", f"{metrics['MAE']:.3f}")
c3.metric("RMSE (eV)", f"{metrics['RMSE']:.3f}")
c4.metric("R²", f"{metrics['R2']:.3f}")

def _boot_ci(y_true, y_pred, n=2000):
    idx = np.random.randint(0, len(y_true), (n, len(y_true)))
    maes = np.abs(y_pred[idx] - y_true[idx]).mean(axis=1)
    return np.percentile(maes, [2.5, 97.5])

lo, hi = _boot_ci(resid.Eg_eV.to_numpy(), resid.Eg_pred.to_numpy())
st.caption(f"95 % CI on MAE: **{lo:.3f} – {hi:.3f} eV**")

# ─── 4) Parity plot ─────────────────────────────────────────────────────
fig = px.scatter(
    resid,
    x="Eg_eV",
    y="Eg_pred",
    hover_data=["Composition"],
    labels={"Eg_eV": "Experimental E₉ (eV)", "Eg_pred": "Predicted E₉ (eV)"},
    height=500,
)
x0, x1 = resid.Eg_eV.min(), resid.Eg_eV.max()
fig.add_shape(type="line", x0=x0, y0=x0, x1=x1, y1=x1, line=dict(dash="dash"))
st.plotly_chart(fig, use_container_width=True)

# ─── 5) Residual table (highlight |err|>0.15) ─────────────────────────
styled = resid.style.apply(
    lambda r: ["background:#fee" if v > 0.15 else "" for v in r.abs_err],
    axis=1,
)
st.markdown("#### Residuals")
st.dataframe(styled, hide_index=True, height=300)
st.download_button(
    "💾 Download residuals CSV",
    resid.to_csv(index=False).encode(),
    "validation_residuals.csv",
    "text/csv",
)

# ─── 6) Show skipped compositions ───────────────────────────────────────
if not skipped.empty:
    with st.expander(f"⚠️ {skipped.shape[0]} skipped compositions"):
        st.dataframe(skipped, hide_index=True)
