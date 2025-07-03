import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from backend.validate import validate as run_validation

st.title("✔ Validation – Experimental Band-Gap Benchmark")
st.markdown(
    """
Benchmarks the **Vegard + bowing** model against 27 experimental
perovskites (1.1 – 1.4 eV).  
Use the slider to adjust the bowing parameter *b*.
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
    height=500, width=620
)

# 1 : 1 diagonal
fig.add_shape(type="line", x0=1.1, y0=1.1, x1=1.4, y1=1.4,
              line=dict(dash="dash"), name="Ideal")

# ---- safe NumPy fit ------------------------------------------------------
df = resid.copy()
df["Eg_eV"]   = pd.to_numeric(df["Eg_eV"],   errors="coerce")
df["Eg_pred"] = pd.to_numeric(df["Eg_pred"], errors="coerce")
df = df.dropna(subset=["Eg_eV", "Eg_pred"])

if len(df) >= 2 and np.isfinite(df["Eg_eV"]).all() and np.isfinite(df["Eg_pred"]).all():
    x = df["Eg_eV"].to_numpy(dtype=float)
    y = df["Eg_pred"].to_numpy(dtype=float)
    m, c = np.polyfit(x, y, 1)
    xfit = np.array([1.1, 1.4])
    fig.add_scatter(x=xfit, y=m * xfit + c,
                    mode="lines", line=dict(dash="dot"),
                    name=f"Fit: y = {m:.2f}x + {c:.2f}")

st.plotly_chart(fig, use_container_width=True)

st.markdown("### Residuals")
st.dataframe(resid, hide_index=True, height=350)

st.download_button(
    "📥 Download residuals CSV",
    resid.to_csv(index=False).encode(),
    "validation_residuals.csv",
    "text/csv"
)
