import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

"""
📊 **Model Validation – calibrated end‑members (no external API)**
=================================================================
* Uses only the uploaded CSV (or `data/benchmark_eg.csv`) → **instant**
  evaluation, no Materials‑Project key needed.
* End‑member gaps `Eg_A` (CsPbBr₃) & `Eg_B` (CsPbI₃) are **learned from
  your dataset** when possible; otherwise fallback to literature values
  2.30 eV (Br) and 1.73 eV (I).
* Vegard + bowing model:
  `Eg_pred = Eg_A·(1‑x) + Eg_B·x – bow·x(1‑x)`
* **Auto‑fit** scans bowing 0→1 and picks the one that minimises MAE.

Required CSV columns (order doesn’t matter):
`formula_A, formula_B, x, rh, temp, Eg_exp`  
Only `x` and `Eg_exp` are used by this simple model.
"""

st.set_page_config(page_title="Model Validation", page_icon="✅")
st.title("📊 Model Validation (calibrated)")

# ── 1  Load CSV ────────────────────────────────────────────────
DEFAULT_PATH = Path("data/benchmark_eg.csv")

df_upload = st.file_uploader("⬆️ Upload experimental CSV (optional)", type=["csv"])
df = pd.read_csv(df_upload) if df_upload else (pd.read_csv(DEFAULT_PATH) if DEFAULT_PATH.exists() else None)

if df is None:
    st.info("Upload a CSV or add data/benchmark_eg.csv (6 columns)")
    st.stop()

for col in ("x", "Eg_exp"):
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["x", "Eg_exp"]).copy()

# ── 2  Calibrate end‑member gaps ───────────────────────────────
DEFAULT_EG_A = 2.30  # CsPbBr3 literature eV
DEFAULT_EG_B = 1.73  # CsPbI3 literature eV

Eg_A = df.loc[df["x"] < 0.05, "Eg_exp"].mean()
Eg_B = df.loc[df["x"] > 0.95, "Eg_exp"].mean()
Eg_A = float(Eg_A) if not np.isnan(Eg_A) else DEFAULT_EG_A
Eg_B = float(Eg_B) if not np.isnan(Eg_B) else DEFAULT_EG_B

# ── 3  Bowing slider + auto‑fit ────────────────────────────────
col_sl, col_bt = st.columns([3,1])
bow = col_sl.slider("Bowing parameter", 0.00, 1.00, 0.30, 0.01)

if col_bt.button("🔧 Auto‑fit"):
    grid = np.linspace(0, 1, 101)
    mae_grid = [
        mean_absolute_error(
            df["Eg_exp"],
            Eg_A*(1-df["x"]) + Eg_B*df["x"] - b*df["x"]*(1-df["x"])
        ) for b in grid
    ]
    bow = float(grid[int(np.argmin(mae_grid))])
    st.success(f"Best bowing ≈ {bow:.2f}")

# ── 4  Predict & score ─────────────────────────────────────────
Eg_pred = Eg_A*(1-df["x"]) + Eg_B*df["x"] - bow*df["x"]*(1-df["x"])
df["Eg_pred"] = Eg_pred

mae = mean_absolute_error(df["Eg_exp"], df["Eg_pred"])
r2  = r2_score(df["Eg_exp"], df["Eg_pred"])

col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error (eV)", f"{mae:.3f}")
col2.metric("R²", f"{r2:.2f}")

# ── 5  Parity plot ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(df["Eg_exp"], df["Eg_pred"], alpha=0.6)
lims = [df[["Eg_exp", "Eg_pred"]].min().min(), df[["Eg_exp", "Eg_pred"]].max().max()]
ax.plot(lims, lims, "k--")
ax.set_xlabel("Experimental Eg (eV)")
ax.set_ylabel("Predicted Eg (eV)")
ax.set_aspect("equal", "box")
st.pyplot(fig)

# ── 6  Download results ───────────────────────────────────────
st.download_button("Download results (CSV)",
                   df.to_csv(index=False).encode(),
                   "validation_results.csv", "text/csv")
