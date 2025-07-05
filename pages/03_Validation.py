# pages/03_Validation.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

# your loader from backend.validate
from backend.validate import load_default_dataset  
# your featurizer from backend.perovskite_utils
from backend.perovskite_utils import featurize  

# ─── Page setup ────────────────────────────────────────────
st.set_page_config(page_title="Validation – Experimental Band-Gap Benchmark")
st.markdown("## ✅ Validation – Experimental Band-Gap Benchmark")

# ─── 1) Load experimental DataFrame ───────────────────────
uploaded = st.file_uploader("📥 Upload a CSV benchmark file", type="csv")
if uploaded:
    df_exp = pd.read_csv(uploaded)
    st.success(f"Loaded {df_exp.shape[0]} rows from your file")
else:
    df_exp = load_default_dataset()
    st.info(f"No upload detected – using built-in {len(df_exp)}-point dataset")

# ─── 2) Normalize column names ─────────────────────────────
df_exp = df_exp.copy()
df_exp.columns = (
    df_exp.columns.str.strip()
                     .str.replace(" ", "_")
                     .str.replace(r"[^\w_]", "", regex=True)
)

# require exactly these two columns
if not {"Composition","Eg_eV"}.issubset(df_exp.columns):
    st.error("❌ Your CSV must contain columns ‘Composition’ and ‘Eg_eV’")
    st.stop()

# ─── 3) Build feature matrix with featurize() ─────────────
# this will return NaN-filled dict whenever parsing fails
X_full = pd.DataFrame([featurize(c) for c in df_exp["Composition"]])

# warn if any compositions failed to featurize
n_bad = X_full.isna().any(axis=1).sum()
if n_bad:
    st.warning(f"⚠️  {n_bad} composition(s) failed to parse and will be mean-imputed")

# target vector
y = df_exp["Eg_eV"].to_numpy()

# ─── 4) Build and fit a Pipeline that imputes THEN RidgeCV ──
alphas = np.logspace(-3, 2, 30)
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("ridgecv", RidgeCV(alphas=alphas, cv=KFold(5, shuffle=True, random_state=0))),
])

# cross-val predictions (imputer+model always sees no NaNs)
y_cv = cross_val_predict(pipeline, X_full, y, cv=5)

# CV metrics
cv_mae = np.mean(np.abs(y_cv - y))
cv_std = np.std (np.abs(y_cv - y))

# now fit on the full dataset
pipeline.fit(X_full, y)
model = pipeline.named_steps["ridgecv"]

# ─── 5) Display summary metrics ────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("N points",     len(y))
col2.metric("5-Fold CV MAE", f"{cv_mae:.3f} ± {cv_std:.3f} eV")
col3.metric("RMSE (CV)",     f"{np.sqrt(np.mean((y_cv-y)**2)):.3f} eV")
col4.metric("R² (CV)",       f"{pipeline.score(X_full,y):.3f}")

# ─── 6) Predict on full set & plot ────────────────────────
df_exp["Eg_pred"] = model.predict(X_full)
df_exp["abs_err"] = (df_exp["Eg_pred"] - df_exp["Eg_eV"]).abs()

fig = px.scatter(df_exp, x="Eg_eV", y="Eg_pred", trendline="ols",
                 labels={"Eg_eV":"Experimental E₉","Eg_pred":"Predicted E₉"})
st.plotly_chart(fig, use_container_width=True)

# downstream residuals table
outliers = df_exp[df_exp["abs_err"] > 0.15]
st.markdown("### Residuals (|error| > 0.15 eV)")
st.dataframe(outliers[["Composition","Eg_eV","Eg_pred","abs_err"]], hide_index=True, height=300)

# Download CSV of the residuals
csv = outliers.to_csv(index=False)
st.download_button("Download residuals CSV", csv, "residuals.csv")
