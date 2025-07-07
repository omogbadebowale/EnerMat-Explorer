import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from backend.perovskite_utils import mix_abx3
from pathlib import Path

# ────────────────────────────────────────────────────────────────
# 📊  Model Validation  (always visible to visitors)
# ────────────────────────────────────────────────────────────────
"""
This page autoloads a bundled benchmark CSV (`data/benchmark_eg.csv`)
and shows MAE, R², and a parity plot.  Visitors can also upload their
own CSV to re-run the test live.
"""

st.set_page_config(page_title="Model Validation", page_icon="✅")
st.title("📊 Model Validation")

# ── 0.  Load default benchmark file ─────────────────────────────
DEFAULT_PATH = Path("data/benchmark_eg.csv")

def load_default_csv():
    return pd.read_csv(DEFAULT_PATH) if DEFAULT_PATH.exists() else None

# ── 1.  Helper: predict Eg for one row ──────────────────────────
def predict_band_gap(row, bowing):
    tbl = mix_abx3(
        formula_A=row["formula_A"],
        formula_B=row["formula_B"],
        rh=row["rh"],
        temp=row["temp"],
        bowing=bowing,
        bg_window=(0, 4),
        dx=0.0005,
    )
    hit = tbl.loc[(tbl["x"] - row["x"]).abs() < 1e-4]
    if not hit.empty:
        return hit.iloc[0]["Eg"]
    return tbl.iloc[(tbl["x"] - row["x"]).abs().argmin()]["Eg"]

# ── 2.  User controls ───────────────────────────────────────────
df_upload = st.file_uploader("⬆️ Upload your own experimental CSV (optional)",
                             type=["csv"])
bow = st.slider("Bowing parameter", 0.00, 1.00, 0.30, 0.05,
                help="Adjusts band-gap curvature between end-members.")

# ── 3.  Choose dataset (upload overrides default) ───────────────
df_exp = pd.read_csv(df_upload) if df_upload else load_default_csv()

if df_exp is None:
    st.info("No benchmark CSV found. Upload a file to see validation metrics.")
    st.stop()

required = {"formula_A", "formula_B", "x", "rh", "temp", "Eg_exp"}
if not required.issubset(df_exp.columns):
    missing = ", ".join(required - set(df_exp.columns))
    st.error(f"CSV missing column(s): {missing}")
    st.stop()

# ── 4.  Compute predictions + metrics ───────────────────────────
df_exp = df_exp.copy()
df_exp["Eg_pred"] = df_exp.apply(lambda r: predict_band_gap(r, bow), axis=1)

mae = mean_absolute_error(df_exp["Eg_exp"], df_exp["Eg_pred"])
r2  = r2_score        (df_exp["Eg_exp"], df_exp["Eg_pred"])

# ── 5.  Display results ─────────────────────────────────────────
col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error (eV)", f"{mae:.3f}")
col2.metric("R²", f"{r2:.2f}")

st.markdown("### Parity plot")
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(df_exp["Eg_exp"], df_exp["Eg_pred"], alpha=0.6)
lims = [df_exp[["Eg_exp", "Eg_pred"]].min().min(),
        df_exp[["Eg_exp", "Eg_pred"]].max().max()]
ax.plot(lims, lims, linestyle="--")
ax.set_xlabel("Experimental Eg (eV)")
ax.set_ylabel("Predicted Eg (eV)")
ax.set_aspect("equal", "box")
st.pyplot(fig)

# ── 6.  Download full table ─────────────────────────────────────
st.download_button(
    "Download results (CSV)",
    df_exp.to_csv(index=False).encode(),
    file_name="validation_results.csv",
    mime="text/csv",
)
