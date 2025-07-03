# pages/03_Validation.py  ── GUI for experimental band-gap benchmark
import streamlit as st
import plotly.express as px
import numpy as np
from backend.validate import validate as run_validation

# ── Title & intro ─────────────────────────────────────────────────────────
st.title("✔ Validation – Experimental Band-Gap Benchmark")

st.markdown(
    """
This page benchmarks the **Vegard + bowing model** against  
27 experimentally reported narrow-band-gap perovskites (1.1 – 1.4 eV).  
Move the slider to adjust the bowing parameter *b*.
"""
)

# ── Bowing-parameter slider ───────────────────────────────────────────────
b = st.slider("Bowing parameter b (eV)", 0.00, 1.00, 0.30, 0.01)

# ── Run validation ────────────────────────────────────────────────────────
metrics, resid = run_validation(b=b)

# ── KPI metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("N points",  metrics["N"])
c2.metric("MAE",       f"{metrics['MAE']:.03f} eV")
c3.metric("RMSE",      f"{metrics['RMSE']:.03f} eV")
c4.metric("R²",        f"{metrics['R2']:.3f}")

# ── Parity plot (custom OLS fit with NumPy) ───────────────────────────────
fig = px.scatter(
    resid,
    x="Eg_eV", y="Eg_pred",
    hover_data=["Composition"],
    labels={"Eg_eV": "Experimental E₉ (eV)",
            "Eg_pred": "Predicted E₉ (eV)"},
    height=500, width=620
)

# 1 : 1 diagonal
fig.add_shape(
    type="line", x0=1.1, y0=1.1, x1=1.4, y1=1.4,
    line=dict(dash="dash"), name="Ideal"
)

# simple linear fit using NumPy (no statsmodels required)
m, c = np.polyfit(resid["Eg_eV"], resid["Eg_pred"], 1)
x_fit = np.array([1.1, 1.4])
fig.add_scatter(
    x=x_fit, y=m * x_fit + c,
    mode="lines",
    line=dict(dash="dot"),
    name=f"Fit: y = {m:.2f}x + {c:.2f}"
)

st.plotly_chart(fig, use_container_width=True)

# ── Residual table & CSV download ─────────────────────────────────────────
st.markdown("### Residuals")
st.dataframe(resid, hide_index=True, height=350)

csv_bytes = resid.to_csv(index=False).encode()
st.download_button(
    "📥 Download residuals CSV",
    csv_bytes, "validation_residuals.csv", "text/csv"
)
