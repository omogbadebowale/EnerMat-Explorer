import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from backend.perovskite_utils import mix_abx3
from pathlib import Path

st.set_page_config(page_title="Model Validation", page_icon="✅")
st.title("📊 Model Validation (auto-fits bowing)")

# ── 0. Load default CSV ──────────────────────────────────────────
df_default = Path("data/benchmark_eg.csv")
if df_default.exists():
    df_default = pd.read_csv(df_default)
else:
    df_default = None

# ── 1. Upload box ────────────────────────────────────────────────
df_upload = st.file_uploader("⬆️ Upload experimental CSV (optional)", type=["csv"])
df_exp = pd.read_csv(df_upload) if df_upload else df_default
if df_exp is None:
    st.info("Upload a CSV or add data/benchmark_eg.csv")
    st.stop()

req = {"formula_A","formula_B","x","rh","temp","Eg_exp"}
if not req.issubset(df_exp.columns):
    st.error(f"CSV is missing column(s): {', '.join(req - set(df_exp.columns))}")
    st.stop()

# ensure numeric
for c in ["x","rh","temp","Eg_exp"]:
    df_exp[c] = pd.to_numeric(df_exp[c], errors="coerce")
df_exp = df_exp.dropna(subset=["x","Eg_exp"])

# ── 2. Prediction helper (interpolated) ──────────────────────────
def predict_one(row, bow):
    tbl = mix_abx3(
        formula_A=row["formula_A"], formula_B=row["formula_B"],
        rh=row["rh"], temp=row["temp"],
        bowing=bow, bg_window=(0,4), dx=0.01)
    return float(np.interp(row["x"], tbl["x"], tbl["Eg"]))

# ── 3. Bowing slider + auto-fit button ──────────────────────────
col_sl, col_bt = st.columns([3,1])
bow_user = col_sl.slider("Bowing parameter", 0.00, 1.00, 0.30, 0.01,
                         help="0.30 is literature for CsPb(Br,I)₃")
auto = col_bt.button("🔧 Auto-fit")

if auto:
    # grid-search 0.05…1.00 to minimise MAE
    grid = np.linspace(0,1,41)
    maes = []
    for b in grid:
        preds = df_exp.apply(lambda r: predict_one(r,b), axis=1)
        maes.append(mean_absolute_error(df_exp["Eg_exp"], preds))
    bow_user = float(grid[int(np.argmin(maes))])
    st.success(f"Best bowing ≈ {bow_user:.2f}")

# ── 4. Run predictions & metrics ─────────────────────────────────
df_exp["Eg_pred"] = df_exp.apply(lambda r: predict_one(r, bow_user), axis=1)
mae = mean_absolute_error(df_exp["Eg_exp"], df_exp["Eg_pred"])
r2  = r2_score        (df_exp["Eg_exp"], df_exp["Eg_pred"])

col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error (eV)", f"{mae:.3f}")
col2.metric("R²", f"{r2:.2f}")

# ── 5. Parity plot ───────────────────────────────────────────────
st.markdown("### Parity plot")
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(df_exp["Eg_exp"], df_exp["Eg_pred"], alpha=0.6)
lims=[df_exp[["Eg_exp","Eg_pred"]].min().min(),
      df_exp[["Eg_exp","Eg_pred"]].max().max()]
ax.plot(lims, lims, "k--")
ax.set_xlabel("Experimental Eg (eV)")
ax.set_ylabel("Predicted Eg (eV)")
ax.set_aspect("equal","box")
st.pyplot(fig)

# ── 6. Download button ──────────────────────────────────────────
st.download_button("Download results (CSV)",
                   df_exp.to_csv(index=False).encode(),
                   "validation_results.csv",
                   "text/csv")
