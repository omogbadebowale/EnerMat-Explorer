# pages/03_Validation.py

import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score

st.set_page_config(page_title="Validation", page_icon="✅", layout="wide")
st.markdown("## ✅ Validation – Experimental Band-Gap Benchmark")

# ─── Require user upload ───────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📥 Upload a CSV benchmark file",
    type=["csv"],
    key="only_csv"
)
if not uploaded:
    st.error("⚠️ Please upload a CSV benchmark file (no built-in data).")
    st.stop()

df_exp = pd.read_csv(uploaded)

# ─── Clean headers & types ─────────────────────────────────────────────────
df_exp.columns = (
    df_exp.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
)
df_exp["Eg_eV"] = pd.to_numeric(df_exp["Eg_eV"], errors="coerce")

# ─── Drop malformed rows and capture skipped ────────────────────────────────
valid = df_exp.dropna(subset=["Composition", "Eg_eV"]).reset_index(drop=True)
skipped = df_exp.loc[~df_exp.index.isin(valid.index)].copy()

# ─── Featurization utilities ───────────────────────────────────────────────
RADII = {"Cs":1.88,"MA":2.17,"FA":2.53,"Pb":1.19,"Sn":1.18,"I":2.20,"Br":1.96,"Cl":1.81}

def parse_comp(comp):
    tokens = re.findall(r"([A-Z][a-z]*)([0-9\.]*)", comp)
    cnt = {}
    for el, num in tokens:
        cnt[el] = float(num) if num else cnt.get(el,0)+1
    return cnt

def tolerance_factor(cnt):
    A = cnt.get("Cs",0)+cnt.get("MA",0)+cnt.get("FA",0)
    B = cnt.get("Pb",0)+cnt.get("Sn",0)
    X = cnt.get("I",0)+cnt.get("Br",0)+cnt.get("Cl",0)
    if not (A and B and X): return np.nan
    rA = sum(RADII[e]*cnt.get(e,0) for e in ["Cs","MA","FA"])/A
    rB = sum(RADII[e]*cnt.get(e,0) for e in ["Pb","Sn"])/B
    rX = sum(RADII[e]*cnt.get(e,0) for e in ["I","Br","Cl"])/X
    return (rA + rX)/np.sqrt(2*(rB + rX))

def featurize(comp):
    cnt = parse_comp(comp)
    totAB = cnt.get("Pb",0)+cnt.get("Sn",0)
    x_sn = cnt.get("Sn",0)/totAB if totAB>0 else 0
    return {
        "x_sn": x_sn,
        "bowing": x_sn*(1-x_sn),
        "tol": tolerance_factor(cnt),
        "A_MA": int(cnt.get("MA",0)>0),
        "A_FA": int(cnt.get("FA",0)>0),
        "X_Br": int(cnt.get("Br",0)>0),
        "X_Cl": int(cnt.get("Cl",0)>0),
    }

# ─── Build feature matrix & target ─────────────────────────────────────────
X = pd.DataFrame([featurize(c) for c in valid["Composition"]])
y = valid["Eg_eV"].to_numpy()

# ─── Train RidgeCV with 5-fold CV ──────────────────────────────────────────
alphas = np.logspace(-3,2,30)
model = RidgeCV(alphas=alphas, cv=KFold(5,shuffle=True,random_state=0))
model.fit(X, y)

cv_maes = -cross_val_score(
    model, X, y,
    cv=KFold(5,shuffle=True,random_state=0),
    scoring="neg_mean_absolute_error"
)
mean_cv, std_cv = cv_maes.mean(), cv_maes.std()

# ─── Display metrics ───────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("N uploaded", df_exp.shape[0])
col2.metric("N validated", valid.shape[0])
col3.metric("5-Fold CV MAE", f"{mean_cv:.3f} ± {std_cv:.3f} eV")
col4.metric("Ridge α", f"{model.alpha_: .3f}")

# ─── Predict, compute hold-out metrics ─────────────────────────────────────
valid["Eg_pred"] = model.predict(X)
valid["abs_err"] = (valid["Eg_pred"] - valid["Eg_eV"]).abs()

MAE = valid["abs_err"].mean()
RMSE = np.sqrt(((valid["Eg_pred"] - valid["Eg_eV"])**2).mean())
R2 = 1 - ((valid["Eg_pred"] - valid["Eg_eV"])**2).sum() / ((valid["Eg_eV"] - valid["Eg_eV"].mean())**2).sum()

st.write("")  # spacer
c1, c2, c3 = st.columns(3)
c1.metric("Hold-out MAE (eV)", f"{MAE:.3f}")
c2.metric("Hold-out RMSE (eV)", f"{RMSE:.3f}")
c3.metric("Hold-out R²", f"{R2:.3f}")

# ─── 95% CI on hold-out MAE via bootstrap ─────────────────────────────────
def bootstrap_ci(y_true, y_pred, n=1000):
    idx = np.random.randint(0, len(y_true), (n, len(y_true)))
    maes = np.abs(y_true[idx] - y_pred[idx]).mean(axis=1)
    return np.percentile(maes, [2.5,97.5])

lo, hi = bootstrap_ci(valid["Eg_eV"].to_numpy(), valid["Eg_pred"].to_numpy())
st.caption(f"95% CI on hold-out MAE: **{lo:.3f}–{hi:.3f} eV**")

# ─── Parity plot ───────────────────────────────────────────────────────────
fig = px.scatter(
    valid, x="Eg_eV", y="Eg_pred",
    hover_data=["Composition","abs_err"],
    labels={"Eg_eV":"Experimental E₉ (eV)", "Eg_pred":"Predicted E₉ (eV)"},
    height=500
)
x0, x1 = valid["Eg_eV"].min(), valid["Eg_eV"].max()
fig.add_shape(type="line", x0=x0, y0=x0, x1=x1, y1=x1,
              line=dict(dash="dash"))
st.plotly_chart(fig, use_container_width=True)

# ─── Residuals table ─────────────────────────────────────────────────────
valid["Outlier"] = valid["abs_err"] > 0.15
st.markdown("#### Residuals (|error| > 0.15 eV flagged)")
st.dataframe(valid[["Composition","Eg_eV","Eg_pred","abs_err","Outlier"]],
             hide_index=True, height=300)
st.download_button("💾 Download residuals CSV",
                   valid.to_csv(index=False).encode(),
                   "residuals.csv", "text/csv")

# ─── Skipped rows expander ─────────────────────────────────────────────────
if not skipped.empty:
    with st.expander(f"⚠️ {skipped.shape[0]} skipped row(s)"):
        st.dataframe(skipped, hide_index=True)
