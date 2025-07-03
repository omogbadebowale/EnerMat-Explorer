# ───────────────────────────────────────────────────────────────────────────────
# Validation page – proves EnerMat’s predictions match trusted references
# Appears automatically under the sidebar because of Streamlit's multipage mode
# ───────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from matminer.datasets import load_dataset
from backend.screen import screen                 # ← import YOUR main scorer

st.title("📐 Model Validation")

# ---------- 1. Load reference datasets (cached) ------------------------------
@st.cache_data(show_spinner="📦 Downloading reference data…")
def load_refs():
    exp = pd.read_csv(
        "https://raw.githubusercontent.com/sg-bioenergy/perovskiteDB/main/perovskite_database.csv"
    )                                             # 7 k experimental band gaps
    dft = load_dataset("castelli_perovskites")    # 19 k DFT entries
    return exp, dft

exp_df, dft_df = load_refs()
st.success("Reference data loaded ✓")

# ---------- 2. Run predictions (cached) --------------------------------------
@st.cache_data(show_spinner="🔮 Predicting on full datasets…")
def run_benchmarks():
    def predict(row):
        out = screen(
            formula_A=row["A"], formula_B=row["B"],
            rh=50, temp=25, bg_window=(1.0, 1.4),
            bowing=0.30, dx=0.05
        )
        return out.iloc[0]["Eg"], out.iloc[0]["stability"]

    # --- Experimental gaps ----------------------------------------------------
    exp_df[["Eg_pred", "stab_pred"]] = exp_df.apply(
        predict, axis=1, result_type="expand"
    )
    mae  = mean_absolute_error(exp_df["Eg"], exp_df["Eg_pred"])
    r2   = r2_score(exp_df["Eg"], exp_df["Eg_pred"])

    # --- DFT stability classification ----------------------------------------
    dft_df["stab_pred"] = dft_df.apply(
        lambda r: predict(r)[1], axis=1
    )
    y_true  = dft_df["e_above_hull"] < 0.05          # stable if ≤50 meV
    auc     = roc_auc_score(y_true, dft_df["stab_pred"])

    return mae, r2, auc, exp_df

mae, r2, auc, exp_df = run_benchmarks()

# ---------- 3. Display headline numbers --------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("MAE (Eg)", f"{mae:.3f} eV")
col2.metric("R² (Eg)",  f"{r2:.2f}")
col3.metric("ROC-AUC (stability)", f"{auc:.2f}")

# ---------- 4. Parity plot ----------------------------------------------------
fig = px.scatter(
    exp_df, x="Eg", y="Eg_pred", opacity=0.6,
    labels={"Eg":"Experimental Eg (eV)", "Eg_pred":"Predicted Eg (eV)"},
    title=f"Predicted vs. Experimental Band Gap  (MAE {mae:.2f} eV)"
)
fig.add_shape(type="line", line=dict(dash="dash"), x0=0, y0=0,
              x1=4, y1=4)                         # 45° guide
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Reference data: Jacobsson *et al.* PerovskiteDB (exp) "
    "and Castelli *et al.* dataset in Matminer (DFT). "
    "Predictions generated on-the-fly with EnerMat Perovskite Explorer backend."
)
