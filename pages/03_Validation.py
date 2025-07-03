# pages/03_Validation.py  ── GUI for experimental band-gap benchmark
import streamlit as st
import plotly.express as px
from backend.validate import validate as run_validation

st.title("✔ Validation – Experimental Band-Gap Benchmark")

st.markdown(
    """
This page benchmarks the **Vegard + bowing model** against  
27 experimentally reported narrow-band-gap perovskites
(1.1 – 1.4 eV window).  
Use the slider to adjust the bowing parameter *b*.
"""
)

# ── User control ──────────────────────────────────────────────────────────
b = st.slider("Bowing parameter b (eV)", 0.0, 1.0, 0.30, 0.01)

# ── Run validation ────────────────────────────────────────────────────────
metrics, resid = run_validation(b=b)

# ── KPI metrics ───────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("N points",  metrics["N"])
k2.metric("MAE",       f"{metrics['MAE']:.03f} eV")
k3.metric("RMSE",      f"{metrics['RMSE']:.03f} eV")
k4.metric("R²",        f"{metrics['R2']:.3f}")

# ── Parity plot ───────────────────────────────────────────────────────────
fig = px.scatter(
    resid, x="Eg_eV", y="Eg_pred",
    hover_data=["Composition"],
    labels={"Eg_eV": "Experimental E₉ (eV)",
            "Eg_pred": "Predicted E₉ (eV)"},
    trendline="ols", height=500, width=600
)
fig.add_shape(type="line", x0=1.1, y0=1.1, x1=1.4, y1=1.4,
              line=dict(dash="dash"))
st.plotly_chart(fig, use_container_width=True)

# ── Residual table & download ─────────────────────────────────────────────
st.markdown("### Residuals")
st.dataframe(resid, hide_index=True, height=350)

csv_bytes = resid.to_csv(index=False).encode()
st.download_button("📥 Download residuals CSV",
                   csv_bytes, "validation_residuals.csv", "text/csv")
