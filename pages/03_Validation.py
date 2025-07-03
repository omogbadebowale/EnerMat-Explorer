# pages/03_Validation.py  – Streamlit validation page
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
measured perovskites (1.1 – 1.4 eV).  
Move the slider to adjust the bowing parameter *b*.
"""
)

b = st.slider("Bowing parameter b (eV)", 0.00, 1.00, 0.30, 0.01)

metrics, resid = run_validation(b=b)

c1, c2, c3, c4 = st.columns(4)
c1.metric("N points", metrics["N"])
c2.metric("MAE",  f'{metrics["MAE"]:.03f} eV')
c3.metric("RMSE", f'{metrics["RMSE"]:.03f} eV')
c4.metric("R²",   f'{metrics["R2"]:.3f}')

fig = px.scatter(
    resid,
    x="Eg_eV", y="Eg_pred",
    hover_data=["Composition"],
    labels={"Eg_eV": "Experimental E₉ (eV)",
            "Eg_pred": "Predicted E₉ (eV)"},
    height=520, width=640
)

fig.add_shape(type="line", x0=1.1, y0=1.1, x1=1.4, y1=1.4,
              line=dict(dash="dash"), name="Ideal")

# ── helper: convert ranges like "1.27–1.38" or "1.27-1.38" to mid-point --
def midpoint(val):
    if isinstance(val, str) and ("–" in val or "-" in val):
        sep = "–" if "–" in val else "-"
        try:
            lo, hi = [float(x) for x in val.split(sep)]
            return (lo + hi) / 2
        except ValueError:
            return np.nan
    return val

# ── clean numeric columns -----------------------------------------------
df = resid.copy()
df["Eg_eV"] = df["Eg_eV"].apply(midpoint)                # <-- correct key
df["Eg_eV"]   = pd.to_numeric(df["Eg_eV"],   errors="coerce")  # <-- correct key
df["Eg_pred"] = pd.to_numeric(df["Eg_pred"], errors="coerce")
df = df.dropna(subset=["Eg_eV", "Eg_pred"])

if len(df) >= 2 and np.isfinite(df["Eg_eV"]).all() and np.isfinite(df["Eg_pred"]).all():
    x = df["Eg_eV"].to_numpy(dtype=float)
    y = df["Eg_pred"].to_numpy(dtype=float)
    m, c = np.polyfit(x, y, 1)
    x_fit = np.array([1.1, 1.4])
    fig.add_scatter(
        x=x_fit, y=m * x_fit + c,
        mode="lines", line=dict(dash="dot"),
        name=f"Fit: y = {m:.2f}x + {c:.2f}"
    )

st.plotly_chart(fig, use_container_width=True)

st.markdown("### Residuals")
st.dataframe(resid, hide_index=True, height=330)

st.download_button(
    "📥 Download residuals CSV",
    resid.to_csv(index=False).encode(),
    "validation_residuals.csv",
    "text/csv"
)
