import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

"""
📊 **Model Validation – calibrated end‑members (offline)**
=========================================================
* Runs **completely locally** – no Materials‑Project API, so it is
  instant and never fails on missing keys.
* End‑member gaps `Eg_A` (CsSnBr₃) & `Eg_B` (CsSnCl₃) are **learned from
  the dataset** (rows with *x* < 0.05 and *x* > 0.95).  If either end is
  missing we use literature values 2.30 eV and 1.73 eV.
* Vegard + bowing model

  ```text
  Eg_pred = Eg_A·(1‑x) + Eg_B·x − bow·x(1‑x)
  ```
* **🔧 Auto‑fit** scans bowing 0 → 1 and chooses the value that minimises
  the mean‑absolute error (MAE).

Required columns (order‑independent)   →  `x`, `Eg_exp`
(other fields are ignored here).
"""

st.set_page_config(page_title="Model Validation", page_icon="✅")
st.title("📊 Model Validation (calibrated)")

# ── 1. Load CSV ────────────────────────────────────────────────
DEFAULT_PATH = "data/benchmark_eg.csv"  # shipped exemplar file
file_up = st.file_uploader("⬆️ Upload experimental CSV (optional)", type=["csv"])  # noqa: E501

if file_up is not None:
    df = pd.read_csv(file_up)
elif os.path.exists(DEFAULT_PATH):
    df = pd.read_csv(DEFAULT_PATH)
else:
    st.info(
        "Upload a CSV **or** place one at `data/benchmark_eg.csv` "
        "containing ‘x’ and ‘Eg_exp’ columns."
    )
    st.stop()

# ── 1‑b. Column normalisation & numeric coercion ───────────────
# ► lower‑case headers and strip whitespace
df.columns = [c.strip().lower() for c in df.columns]

# ► map common aliases to canonical names
ALIASES = {
    "eg (ev)": "eg_exp",
    "eg_ev": "eg_exp",
    "eg": "eg_exp",
    "bandgap": "eg_exp",
    "band_gap": "eg_exp",
    "x (%)": "x",
    "fraction": "x",
    "composition": "x",
    "x_exp": "x",
}
df.rename(columns=ALIASES, inplace=True)

# ► ensure required columns are present
missing = {"x", "eg_exp"} - set(df.columns)
if missing:
    st.error(
        "❌ CSV is missing required column(s): " + ", ".join(sorted(missing))
    )
    st.stop()

# ► numeric coercion & purge NaNs
for col in ("x", "eg_exp"):
    df[col] = pd.to_numeric(df[col], errors="coerce")
df.dropna(subset=["x", "eg_exp"], inplace=True)

if df.empty:
    st.error("All rows were invalid after cleaning – please check the file.")
    st.stop()

# ── 2. Calibrate end‑member gaps ───────────────────────────────
DEFAULT_EG_A = 2.30  # CsSnBr3 (literature)
DEFAULT_EG_B = 1.73  # CsSnCl3 (literature)

Eg_A = df.loc[df["x"] < 0.05, "eg_exp"].mean()
Eg_B = df.loc[df["x"] > 0.95, "eg_exp"].mean()
Eg_A = float(Eg_A) if not np.isnan(Eg_A) else DEFAULT_EG_A
Eg_B = float(Eg_B) if not np.isnan(Eg_B) else DEFAULT_EG_B

# ── 3. Bowing controls ────────────────────────────────────────
col_sl, col_bt = st.columns([4, 1])
bow = col_sl.slider("Bowing parameter", 0.00, 1.00, 0.30, 0.01)

def vegard_bowing(x: pd.Series | np.ndarray, b: float) -> pd.Series:
    """Quadratic Vegard‑plus‑bowing gap interpolation."""
    return Eg_A * (1 - x) + Eg_B * x - b * x * (1 - x)

if col_bt.button("🔧 Auto‑fit"):
    grid = np.linspace(0, 1, 201)
    maes = [mean_absolute_error(df["eg_exp"], vegard_bowing(df["x"], b)) for b in grid]
    bow = float(grid[int(np.argmin(maes))])
    st.success(f"Best bowing ≈ {bow:.2f}")

# ── 4. Predictions & metrics ───────────────────────────────────

df["eg_pred"] = vegard_bowing(df["x"], bow)
mae = mean_absolute_error(df["eg_exp"], df["eg_pred"])
r2 = r2_score(df["eg_exp"], df["eg_pred"])

st.markdown("### Validation summary")
col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error (eV)", f"{mae:.3f}", delta="Target ≤ 0.15")
col2.metric("R²", f"{r2:.2f}", delta="Target ≥ 0.85")

# ── 5. Parity plot ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(df["eg_exp"], df["eg_pred"], alpha=0.65)
lims = [df[["eg_exp", "eg_pred"]].min().min(), df[["eg_exp", "eg_pred"]].max().max()]
ax.plot(lims, lims, "k--", linewidth=1)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Experimental Eg (eV)")
ax.set_ylabel("Predicted Eg (eV)")
ax.set_aspect("equal", "box")
st.pyplot(fig)

# ── 6. Download results ───────────────────────────────────────
st.download_button(
    label="Download results (CSV)",
    data=df.to_csv(index=False).encode(),
    file_name="validation_results.csv",
    mime="text/csv",
)
