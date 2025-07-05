# pages/03_Validation.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

from backend.validate import load_default_dataset
from backend.perovskite_utils import featurize  # <-- your featurizer

st.set_page_config(page_title="Validation – Experimental Band-Gap Benchmark")

st.markdown("# ✅ Validation – Experimental Band-Gap Benchmark")

# ─── 0) Let user upload or fall back to built-in ─────────────────────────
uploaded = st.file_uploader(
    "📥 Upload a CSV benchmark file (leave empty to use the built-in 27-point dataset)",
    type=["csv", "ods"],
)
if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"Loaded {len(df)} rows from your file")
else:
    df = load_default_dataset()
    st.info("Using the built-in 27-point dataset")

# ─── 1) Normalize column names ─────────────────────────────────────────
df = df.copy()
df.columns = (
    df.columns.str.strip()
              .str.replace(" ", "_")
              .str.replace(r"[^\w_]", "", regex=True)
)

# ─── 2) Force Eg_eV → float, drop bad targets ──────────────────────────
# coerce anything non-numeric into NaN
df["Eg_eV"] = pd.to_numeric(df.get("Eg_eV", pd.Series()), errors="coerce")
bad_y = df["Eg_eV"].isna().sum()
if bad_y:
    st.warning(f"⚠️ Dropping {bad_y} row(s) whose Eg_eV failed to parse")
    df = df[df["Eg_eV"].notna()].reset_index(drop=True)

# now bail if the two columns we absolutely need are missing
if not {"Composition", "Eg_eV"}.issubset(df.columns):
    st.error("❌ Your data must have columns ‘Composition’ and ‘Eg_eV’")
    st.stop()

# ─── 3) Build feature matrix + impute missing X ────────────────────────
# featurize each composition string
X_full = pd.DataFrame([featurize(c) for c in df["Composition"]])
y = df["Eg_eV"].to_numpy()

# drop any rows where your featurizer produced NaN features
mask = X_full.notna().all(axis=1)
if not mask.all():
    dropped = (~mask).sum()
    st.warning(f"⚠️ Dropping {dropped} composition(s) with invalid features")
X = X_full[mask].reset_index(drop=True)
y = y[mask]
valid = df.loc[mask].reset_index(drop=True)

# impute any remaining feature gaps (mean‐fill)
imp = SimpleImputer(strategy="mean")
X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

# ─── 4) Train RidgeCV via 5-fold CV ────────────────────────────────────
alphas = np.logspace(-3, 2, 30)
model = RidgeCV(alphas=alphas, cv=KFold(5, shuffle=True, random_state=0))
# cross_val_predict will internally re-fit 5 times and produce out-of-fold predictions
y_cv = cross_val_predict(model, X, y, cv=5)
cv_mae  = np.mean(np.abs(y_cv - y))
cv_std  = np.std (np.abs(y_cv - y))
model.fit(X, y)

# ─── 5) Display metrics & scatter ──────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("N points",       len(valid))
col2.metric("5-Fold CV MAE",  f"{cv_mae:.3f} ± {cv_std:.3f} eV")
col3.metric("Ridge α",        f"{model.alpha_:.3f}")
col4.metric("Dataset MAE",    f"{np.mean(np.abs(model.predict(X) - y)):.3f} eV")

fig = px.scatter(
    valid.assign(Eg_pred=model.predict(X)),
    x="Eg_eV", y="Eg_pred",
    trendline="ols",
    labels={"Eg_eV":"Experimental E₉ (eV)", "Eg_pred":"Predicted E₉ (eV)"}
)
st.plotly_chart(fig, use_container_width=True)

# ─── 6) Residuals table & download ──────────────────────────────────────
valid["Eg_pred"] = model.predict(X)
valid["abs_err"] = (valid["Eg_pred"] - valid["Eg_eV"]).abs()
res_out = valid[valid["abs_err"] > 0.15]
st.dataframe(res_out, hide_index=True, height=250)

csv = valid.to_csv(index=False)
st.download_button("📥 Download full residuals CSV", csv, "residuals.csv")
