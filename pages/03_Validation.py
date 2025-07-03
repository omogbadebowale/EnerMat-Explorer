# pages/03_Validation.py  ── Streamlit GUI for experimental band-gap benchmark
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from backend.validate import validate as run_validation

# ──────────────────────────────────────────────────────────────────────────
st.title("✔ Validation – Experimental Band-Gap Benchmark")

st.markdown(
    """
Benchmarks the **Vegard + bowing** model against 27 experimentally
measured narrow-band-gap perovskites (1.1 – 1.4 eV).

*Adjust the slider to change the bowing parameter *b**.
"""
)

# ── User control ──────────────────────────────────────────────────────────
b = st.slider("Bowing parameter b (eV)", min_value=0.00, max_value=1.00,
              value=0.30, step=0.01)

# ── Run validation ────────────────────────────────────────────────────────
metrics, resid = run_validation(b=b)

# ── KPI display ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("N points", metrics["N"])
c2.metric("MAE",  f'{metrics["MAE"]:.03f} eV')
c3.metric("RMSE", f'{metrics["RMSE"]:.03f} eV')
c4.metric("R²",   f'{metrics["R2"]:.3f}')

# ── Parity scatter --------------------------------------------------------
fig = px.scatter(
    resid,
    x="Eg_eV", y="Eg_pred",
    hover_data=["Composition"],
    labels={"Eg_eV":   "Experimental E₉ (eV)",
            "Eg_pred": "Predicted E₉ (eV)"},
    height=520, width=640
)

# 1 : 1 diagonal
fig.add_shape(type="line", x0=1.1, y0=1.1, x1=1.4, y1=1.4,
              line=dict(dash="dash"), name="Ideal")

# ── Helper: turn "1.27–1.38" or "1.27-1.38" into 1.325 -------------------
def midpoint(val: str | float) -> float | str:
    if isinstance(val, str) and ("–" in val or "-" in val):
        sep = "–" if "–" in val else "-"
        try:
            lo, hi = [float(x) for x in val.split(sep)]
            return (lo + hi) / 2
        except ValueError:
            return np.nan
    return val

# ── Prepare clean numeric arrays for OLS fit -----------------------------
df = resid.copy()
df["Eg_eV"] = df["Eg_eV"].apply(midpoint)
df["Eg_eV"]   = pd.to_numeric(df["Eg_eV"],   errors="coerce")
df["Eg_pred"] = pd.to_numeric(df["Eg_pred"], errors="coerce")
df = df.dropna(subset=["Eg_eV", "Eg_pred"])

if len(df) >= 2 and np.isfinite(df["Eg_eV"]).all() and np.isfinite(df["Eg_pred"]).all():
    x = df["Eg_eV"].to_numpy(dtype=float)
    y = df["Eg_pred"].to_numpy(dtype=float)
    m, c = np.polyfit(x, y, 1)
    x_fit = np.array([1.1, 1.4])
    fig.add_scatter(
        x=x_fit, y=m * x_fit + c,
        mode="lines",
        line=dict(dash="dot"),
        name=f"Fit: y = {m:.2f}x + {c:.2f}"
    )

st.plotly_chart(fig, use_container_width=True)

# ── Residual table & download --------------------------------------------
st.markdown("### Residuals")
st.dataframe(resid, hide_index=True, height=330)

st.download_button(
    "📥 Download residuals CSV",
    resid.to_csv(index=False).encode(),
    file_name="validation_residuals.csv",
    mime="text/csv"
)
