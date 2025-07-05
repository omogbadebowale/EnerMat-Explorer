# pages/03_Validation.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.impute import SimpleImputer

# your featurizer; adjust path if needed
from backend.perovskite_utils import featurize  
from backend.validate import load_default_dataset 
from backend.perovskite_utils import featurize

st.set_page_config(page_title="Validation – Experimental Band-Gap Benchmark")
st.markdown("## ✅ Validation – Experimental Band-Gap Benchmark")

# 1. load uploaded or default
uploaded = st.file_uploader("📥 Upload a CSV benchmark file", type="csv")
if uploaded is None:
    st.warning("Please upload a CSV benchmark file. The built-in dataset is disabled.")
    st.stop()
df = pd.read_csv(uploaded)

# 2. normalize column names
df.columns = (
    df.columns
      .str.strip()
      .str.replace(" ", "_")
      .str.replace(r"[^\w_]", "", regex=True)
)

# 3. build features + target
df = df.reset_index(drop=True)
X_full = pd.DataFrame([featurize(comp) for comp in df["Composition"]])
y = df["Eg_eV"].to_numpy()

# 4. impute any NaNs in X
imp = SimpleImputer(strategy="mean")
X = imp.fit_transform(X_full)
n_imputed = np.isnan(X_full.values).sum()
if n_imputed > 0:
    st.warning(f"⚠️ Imputed {n_imputed} missing feature values (mean‐fill)")

# 5. train RidgeCV
alphas = np.logspace(-3, 2, 30)
cv = KFold(5, shuffle=True, random_state=0)
model = RidgeCV(alphas=alphas, cv=cv)
model.fit(X, y)

# 6. display upload + training metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("N uploaded", len(df))
col2.metric("N features", X.shape[1])
cv_preds = cross_val_predict(model, X, y, cv=cv)
cv_mae = np.mean(np.abs(cv_preds - y))
cv_std = np.std(np.abs(cv_preds - y))
col3.metric("5-Fold CV MAE", f"{cv_mae:.3f} ± {cv_std:.3f} eV")
col4.metric("Ridge α", f"{model.alpha_:.3f}")

# 7. predict & plot
df["Eg_pred"] = model.predict(X)
df["abs_err"] = np.abs(df["Eg_pred"] - df["Eg_eV"])
st.metric("MAE", f"{df['abs_err'].mean():.3f} eV")
st.metric("RMSE", f"{np.sqrt((df['abs_err']**2).mean()):.3f} eV")

fig = px.scatter(
    df, x="Eg_eV", y="Eg_pred", trendline="ols",
    labels={"Eg_eV": "Experimental E₉ (eV)", "Eg_pred":"Predicted E₉ (eV)"}
)
st.plotly_chart(fig, use_container_width=True)

# 8. show outliers & download
outliers = df[df["abs_err"] > 0.15]
st.markdown("### Residuals (|error| > 0.15 eV flagged)")
st.dataframe(outliers[["Composition","Eg_eV","Eg_pred","abs_err"]], hide_index=True, height=300)
csv = df.to_csv(index=False)
st.download_button("Download residuals CSV", csv, "residuals.csv")
