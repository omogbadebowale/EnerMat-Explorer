import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from backend.perovskite_utils import get_Eg_endmember
from pathlib import Path

"""
📊 **Model Validation – calibrated end‑members**
================================================
* **No more Materials‑Project lookup** → instant load, no API key needed.
* End‑member band gaps `Eg_A` (CsPbBr₃) and `Eg_B` (CsPbI₃) are
  **automatically calibrated from your dataset**: we take the average
  `Eg_exp` of points with *x* < 0.05 for `Eg_A` and *x* > 0.95 for
  `Eg_B`.  If a dataset lacks either end, we fall back to hard‑coded
  literature values 2.30 eV (Br) and 1.73 eV (I).
* Prediction uses Vegard + bowing:
  `Eg_pred = Eg_A·(1‑x) + Eg_B·x – bow·x(1‑x)`
* **Auto‑fit** finds the bowing that minimises MAE.

Required columns (same as before):
`formula_A, formula_B, x, rh, temp, Eg_exp` – only `x` and `Eg_exp` are
used by this simple Vegard model.
"""

st.set_page_config(page_title="Model Validation", page_icon="✅")
st.title("📊 Model Validation (calibrated)")

# ── 1. Load CSV ────────────────────────────────────────────────
DEFAULT_PATH = Path("data/benchmark_eg.csv")

df_upload = st.file_uploader("⬆️ Upload experimental CSV (optional)", type=["csv"])
df = pd.read_csv(df_upload) if df_upload else (pd.read_csv(DEFAULT_PATH) if DEFAULT_PATH.exists() else None)

if df is None:
    st.info("Upload a CSV or add data/benchmark_eg.csv")
    st.stop()

for col in ["x", "Eg_exp"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["x", "Eg_exp"])

# ── 2. Determine end‑member gaps Eg_A and Eg_B ────────────────
DEFAULT_EG_A = 2.30  # CsPbBr3 literature (eV)
DEFAULT_EG_B = 1.73  # CsPbI3 literature (eV)

eA = df.loc[df["x"] < 0.05, "Eg_exp"].mean()
eB = df.loc[df["x"] > 0.95, "Eg_exp"].mean()
Eg_A = float(eA) if not np.isnan(eA) else DEFAULT_EG_A
Eg_B = float(eB) if not np.isnan(eB) else DEFAULT_EG_B

# ── 3. Bowing slider + auto‑fit ────────────────────────────────
col_sl, col_bt = st.columns([3,1])
bow = col_sl.slider("Bowing parameter", 0.00, 1.00, 0.30, 0.01)
if col_bt.button("🔧 Auto‑fit"):
    grid = np.linspace(0,1,101)
    maes = []
    for b in grid:
        preds = Eg_A*(1-df["x"]) + Eg_B*df["x"] - b*df["x"]*(1-df["x"])
        maes.append(mean_absolute_error(df["Eg_exp"], preds))
    bow = float(grid[int(np.argmin(maes))])
    st.success(f"Best bowing ≈ {bow:.2f}")

# ── 4. Predictions & metrics ──────────────────────────────────
Eg_pred = Eg_A*(1-df["x"]) + Eg_B*df["x"] - bow*df["x"]*(1-df["x"])
df["Eg_pred"] = Eg_pred

mae = mean_absolute_error(df["Eg_exp"], df["Eg_pred"])
r2  = r2_score        (df["Eg_exp"], df["Eg_pred"])

col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error (eV)", f"{mae:.3f}")
col2.metric("R²", f"{r2:.2f}")

# ── 5. Parity plot ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(df["Eg_exp"], df["Eg_pred"], alpha=0.6)
lims=[df[["Eg_exp","Eg_pred"]].min().min(), df[["Eg_exp","Eg_pred"]].max().max()]
ax.plot(lims, lims, "k--")
ax.set_xlabel("Experimental Eg (eV)")
ax.set_ylabel("Predicted Eg (eV)")
ax.set_aspect("equal","box")
st.pyplot(fig)

# ── 6. Download button ────────────────────────────────────────
st.download_button("Download results (CSV)", df.to_csv(index=False).encode(),
                   "validation_results.csv", "text/csv")
