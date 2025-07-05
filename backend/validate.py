# backend/model.py

import re
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score

# built-in fallback for pure endpoints
ENDMEM = {
    "CsSnI3": 1.30, "CsPbI3": 1.73,
    "MAPbI3": 1.55, "FAPbI3": 1.48,
    "CsSnBr3": 1.79, "CsSnCl3": 2.89,
}

# ionic radii (Shannon) Å
RADII = {
    "Cs": 1.88, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18,
    "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

def parse_comp(comp: str):
    """Very basic: extract counts of each element."""
    # assume formula like FA0.5Cs0.5Sn0.5Pb0.5I3 etc
    tokens = re.findall(r"([A-Z][a-z]*)([0-9\.]*)", comp)
    cnt = {}
    for el, num in tokens:
        cnt[el] = float(num) if num else cnt.get(el, 0) + 1
    return cnt

def tolerance_factor(cnt):
    """Goldschmidt tol. factor for ABX3."""
    A = cnt.get("Cs",0)+cnt.get("MA",0)+cnt.get("FA",0)
    B = cnt.get("Pb",0)+cnt.get("Sn",0)
    X = cnt.get("I",0)+cnt.get("Br",0)+cnt.get("Cl",0)
    if not (A and B and X): return np.nan
    rA = sum(RADII[e]*cnt[e] for e in ["Cs","MA","FA"] if e in cnt)/A
    rB = sum(RADII[e]*cnt[e] for e in ["Pb","Sn"]    if e in cnt)/B
    rX = sum(RADII[e]*cnt[e] for e in ["I","Br","Cl"] if e in cnt)/X
    return (rA + rX)/np.sqrt(2*(rB + rX))

def featurize(comp):
    cnt = parse_comp(comp)
    # fractions
    totAB = cnt.get("Pb",0) + cnt.get("Sn",0)
    x_sn = cnt.get("Sn",0)/totAB if totAB>0 else 0
    # one-hots
    A_cs = int(cnt.get("Cs",0)>0)
    A_ma = int(cnt.get("MA",0)>0)
    A_fa = int(cnt.get("FA",0)>0)
    X_i  = int(cnt.get("I",0)>0)
    X_br = int(cnt.get("Br",0)>0)
    X_cl = int(cnt.get("Cl",0)>0)
    # bowing term
    bow = x_sn*(1-x_sn)
    # tolerance
    tol = tolerance_factor(cnt)
    return {
        "x_sn": x_sn,
        "bowing": bow,
        "tol": tol,
        "A_MA": A_ma,
        "A_FA": A_fa,
        "X_Br": X_br,
        "X_Cl": X_cl,
    }

def load_default_dataset():
    # same as before
    from backend.validate import load_default_dataset as L
    return L()

def train_model(df: pd.DataFrame):
    # ensure y is numeric
    df = df.copy()
    df["Eg_eV"] = pd.to_numeric(df["Eg_eV"], errors="coerce")
    df = df.dropna(subset=["Composition","Eg_eV"])
    X = pd.DataFrame([featurize(c) for c in df["Composition"]])
    y = df["Eg_eV"].to_numpy()
    # Ridge with built-in alphas, 5-fold CV
    alphas = np.logspace(-3,2,30)
    model = RidgeCV(alphas=alphas, cv=KFold(5,shuffle=True,random_state=0))
    model.fit(X,y)
    # CV MAE
    cvmae = -cross_val_score(
        model, X,y, cv=KFold(5,shuffle=True,random_state=0),
        scoring="neg_mean_absolute_error"
    ).mean()
    return model, cvmae

# cache a single global model on the default dataset
_GLOBAL_MODEL = None
_GLOBAL_CVMAE = None

def get_default_model():
    global _GLOBAL_MODEL, _GLOBAL_CVMAE
    if _GLOBAL_MODEL is None:
        df = load_default_dataset()
        _GLOBAL_MODEL, _GLOBAL_CVMAE = train_model(df)
    return _GLOBAL_MODEL, _GLOBAL_CVMAE

def predict(comp: str, model=None):
    if model is None:
        model, _ = get_default_model()
    return model.predict(pd.DataFrame([featurize(comp)]))[0]
