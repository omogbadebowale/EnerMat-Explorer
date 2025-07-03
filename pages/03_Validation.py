# pages/03_Validation.py  ── GUI for experimental band-gap benchmark
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from backend.validate import validate as run_validation

# ───────────────────────────────────────────────────────────────────────────
st.title("✔ Validation – Experimental Band-Gap Benchmark")

st.markdown(
    """
This page benchmarks the **Vegard + bowing model** against  
27 experimentally measured perovskites (1.1 – 1.4 eV).  
Use the slider to adjust the bowing parameter *b*.
"""
)

# ── User control ──────────────────────────────────────────────────────────
b = st.slider("Bowing parameter b (eV)", 0.00, 1.00, 0.30, 0.01)

# ── Run validation ────────────────────────────────────────────────────────
metrics, resid = run_validation(b=b)

# ── KPI metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("N points",  metrics["N"])
c2.metric("MAE",       f"{metrics['MAE']:.03f} eV")
c3.metric("RMSE",      f"{metrics['RMSE']:.03f} eV")
c4.metric("R²",        f"{metrics['R2']:.3f}")

# ── Parity plot ───────────────────────────────────────────────────────────
fig = px.scatter(
    resid,
    x="Eg_eV", y="Eg_pred",
    hover_data=["Composition"],
    labels={"Eg_eV": "Experimental E₉ (eV)",
            "Eg_pred": "Predicted E₉ (eV)"},
    height=500, width=620
)

# 1 : 1 reference
fig.add_shape(type="line", x0=1.1, y0=1.1, x1=1.4, y1=1.4,
              line=dict(dash="dash"), name="Ideal")

# ── Robust NumPy OLS (only if safe) ───────────────────────────────────────
def add_numpy_fit(df):
    df_num = df.copy()
    df_num["Eg_eV"]   = pd.to_numeric(df_num["Eg_eV"],   errors="coerce")
    df_num["Eg_pred"] = pd.to_numeric(df_num["Eg_pred"], errors="coerce")
    df_num = df_num.dropna(subset=["Eg_eV", "Eg_pred"])
    if len(df_num) < 2:
        return                                          # not enough points
    try:
        x = df_num["Eg_eV"].astype(float).to_numpy()
        y = df_num["Eg_pred"].astype(float).to_numpy()
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            return                                      # guard against inf/nan
        m, c = np.polyfit(x, y, 1)
        x_fit = np.array([1.1, 1.4])
        fig.add_scatter(
            x=x_fit, y=m * x_fit + c,
            mode="lines",
            line=dict(dash="dot"),
            name=f"Fit: y = {m:.2f}x + {c:.2f}"
        )
    except (TypeError, np.linalg.LinAlgError):
        # even in pathological cases, do nothing instead of crashing
        pass

add_numpy_fit(resid)

st.plotly_chart(fig, use_container_width=True)

# ── Residual table & download ─────────────────────────────────────────────
st.markdown("### Residuals")
st.dataframe(resid, hide_index=True, height=350)

dl = resid.to_csv(index=False).encode()
st.download_button("📥 Download residuals CSV", dl,
                   "validation_residuals.csv", "text/csv")
