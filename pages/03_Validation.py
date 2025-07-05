import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

from backend.perovskite_utils import featurize           # ← your featurization fn
from backend.validate import load_default_dataset        # ← your built-in 27-point loader

st.set_page_config(page_title="Validation", layout="wide")
st.markdown("## ✅ Validation – Experimental Band-Gap Benchmark")

# ─── Upload or fallback ──────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📥 Upload a CSV benchmark file", type=["csv"], accept_multiple_files=False
)
if uploaded:
    valid = pd.read_csv(uploaded)
else:
    valid = load_default_dataset()

# ─── Normalize column names ──────────────────────────────────────────────────
valid.columns = (
    valid.columns.str.strip()
                  .str.replace(" ", "_")
                  .str.replace(r"[^\w_]", "", regex=True)
)

# ─── Build feature matrix & target (with mask) ─────────────────────────────
X_full = pd.DataFrame([featurize(c) for c in valid["Composition"]])
y_full = valid["Eg_eV"].to_numpy()

mask = X_full.notnull().all(axis=1)
if not mask.all():
    dropped = (~mask).sum()
    st.warning(f"⚠️ Dropping {dropped} composition(s) with invalid features")

valid2 = valid.loc[mask].reset_index(drop=True)
X      = X_full.loc[mask].reset_index(drop=True)
y      = y_full[mask]

# ─── Train RidgeCV with 5-fold CV ────────────────────────────────────────────
alphas = np.logspace(-3, 2, 30)
model  = RidgeCV(alphas=alphas, cv=KFold(5, shuffle=True, random_state=0))
model.fit(X, y)

# ─── Display summary metrics ────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("N uploaded", valid.shape[0])
col2.metric("N validated", valid2.shape[0])

# compute cv errors only once
y_cv = cross_val_predict(model, X, y, cv=5)
errs = np.abs(y_cv - y)
col3.metric("5-Fold CV MAE", f"{errs.mean():.3f} ± {errs.std():.3f} eV")
col4.metric("Ridge α", f"{model.alpha_:.3f}")

# ─── Predict on valid2 & show error metrics ────────────────────────────────
valid2["Eg_pred"] = model.predict(X)
valid2["abs_err"] = (valid2["Eg_pred"] - valid2["Eg_eV"]).abs()

mae  = valid2["abs_err"].mean()
rmse = np.sqrt((valid2["abs_err"]**2).mean())
st.metric("MAE", f"{mae:.3f} eV")
st.metric("RMSE", f"{rmse:.3f} eV")
st.metric("R²", f"{model.score(X, y):.3f}")

# ─── Scatter + trendline ───────────────────────────────────────────────────
fig = px.scatter(
    valid2,
    x="Eg_eV", y="Eg_pred",
    trendline="ols",
    labels={"Eg_eV":"Experimental E₉ (eV)", "Eg_pred":"Predicted E₉ (eV)"},
)
st.plotly_chart(fig, use_container_width=True)

# ─── Residuals table & download ─────────────────────────────────────────────
res_outliers = valid2[valid2["abs_err"] > 0.15]
st.markdown("### Residuals (|error| > 0.15 eV flagged)")
st.dataframe(res_outliers, hide_index=True, height=300)

csv = valid2.to_csv(index=False).encode("utf-8")
st.download_button("💾 Download residuals CSV", data=csv, file_name="residuals.csv")
