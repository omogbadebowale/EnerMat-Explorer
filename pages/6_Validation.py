import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from backend.perovskite_utils import mix_abx3
from pathlib import Path

try:
    from mp_api.client.core import MPRestError  # type: ignore
except Exception:  # fallback if mp_api not installed locally
    class MPRestError(Exception):
        pass

"""
📊 **Model Validation (auto‑fits bowing and handles API errors)**
===============================================================
* Autoloads `data/benchmark_eg.csv` (six‑column format).
* Computes MAE, R² + parity plot.
* **Auto‑fit** button finds the best bowing for any dataset.
* Catches Materials Project API errors and tells the user how to fix them.

**Required columns** (order doesn’t matter)
```
formula_A, formula_B, x, rh, temp, Eg_exp
```
For CsPbBr/I binaries `formula_A = CsPbBr3`, `formula_B = CsPbI3`, and
`x` is the I‑fraction.
"""

st.set_page_config(page_title="Model Validation", page_icon="✅")

st.title("📊 Model Validation (auto‑fits bowing)")

# ────────────────────────────────────────────────────────────────
# 0.  Materials Project API key check (needed by mix_abx3)
# ────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("MP_API_KEY")
if not API_KEY:
    st.warning("`MP_API_KEY` not found in env — mix_abx3 may fail.\n\nSet it in *Manage app → Secrets* (Streamlit Cloud) or locally via `export MP_API_KEY=...`.")

# ────────────────────────────────────────────────────────────────
# 1.  Load default CSV or user upload
# ────────────────────────────────────────────────────────────────
DEFAULT_PATH = Path("data/benchmark_eg.csv")

df_upload = st.file_uploader("⬆️ Upload experimental CSV (optional)", type=["csv"])
df_exp = pd.read_csv(df_upload) if df_upload else (pd.read_csv(DEFAULT_PATH) if DEFAULT_PATH.exists() else None)

if df_exp is None:
    st.info("Upload a CSV or place one at `data/benchmark_eg.csv`. Required columns: formula_A,B,x,rh,temp,Eg_exp")
    st.stop()

# column sanity
required = {"formula_A", "formula_B", "x", "rh", "temp", "Eg_exp"}
missing = required - set(df_exp.columns)
if missing:
    st.error(f"CSV missing column(s): {', '.join(missing)}")
    st.stop()

# numeric coercion
for c in ["x", "rh", "temp", "Eg_exp"]:
    df_exp[c] = pd.to_numeric(df_exp[c], errors="coerce")
df_exp = df_exp.dropna(subset=["x", "Eg_exp"]).copy()

# ────────────────────────────────────────────────────────────────
# 2.  Prediction helper (interpolated, with error handling)
# ────────────────────────────────────────────────────────────────

def predict_one(row, bow):
    try:
        tbl = mix_abx3(
            formula_A=row["formula_A"],
            formula_B=row["formula_B"],
            rh=row["rh"], temp=row["temp"],
            bowing=bow, bg_window=(0,4), dx=0.01,
        )
        return float(np.interp(row["x"], tbl["x"], tbl["Eg"]))
    except MPRestError as ex:
        # API call failed (key missing or network). Mark row as NaN.
        st.error("Materials Project API error → set MP_API_KEY then reload.\nThe app will skip rows until then.")
        return np.nan
    except Exception as ex:
        st.error(f"Prediction failed for row with x={row['x']}: {ex}")
        return np.nan

# ────────────────────────────────────────────────────────────────
# 3.  Bowing controls + auto‑fit
# ────────────────────────────────────────────────────────────────
col_sl, col_bt = st.columns([3,1])
bow = col_sl.slider("Bowing parameter", 0.00, 1.00, 0.30, 0.01)
if col_bt.button("🔧 Auto‑fit"):
    grid = np.linspace(0,1,41)
    maes = []
    for b in grid:
        preds = df_exp.apply(lambda r: predict_one(r,b), axis=1)
        maes.append(mean_absolute_error(df_exp["Eg_exp"], preds))
    bow = float(grid[int(np.argmin(maes))])
    st.success(f"Best bowing ≈ {bow:.2f}")

# ────────────────────────────────────────────────────────────────
# 4.  Run predictions
# ────────────────────────────────────────────────────────────────
df_exp["Eg_pred"] = df_exp.apply(lambda r: predict_one(r, bow), axis=1)
df_valid = df_exp.dropna(subset=["Eg_pred"])  # drop rows that failed due to API

if df_valid.empty:
    st.error("No predictions could be made — likely MP_API_KEY missing. Add the key and reload.")
    st.stop()

mae = mean_absolute_error(df_valid["Eg_exp"], df_valid["Eg_pred"])
r2  = r2_score        (df_valid["Eg_exp"], df_valid["Eg_pred"])

col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error (eV)", f"{mae:.3f}")
col2.metric("R²", f"{r2:.2f}")

# ────────────────────────────────────────────────────────────────
# 5.  Parity plot
# ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(df_valid["Eg_exp"], df_valid["Eg_pred"], alpha=0.6)
lims=[df_valid[["Eg_exp","Eg_pred"]].min().min(),
      df_valid[["Eg_exp","Eg_pred"]].max().max()]
ax.plot(lims, lims, "k--")
ax.set_xlabel("Experimental Eg (eV)")
ax.set_ylabel("Predicted Eg (eV)")
ax.set_aspect("equal","box")
st.pyplot(fig)

# ────────────────────────────────────────────────────────────────
# 6.  Download results
# ────────────────────────────────────────────────────────────────

st.download_button("Download results (CSV)",
                   df_valid.to_csv(index=False).encode(),
                   "validation_results.csv", "text/csv")
