"""
backend.validate  – benchmark 27 experimental band-gaps.
Run   : python -m backend.validate
Import: from backend.validate import validate
"""
from pathlib import Path
import re, json, numpy as np, pandas as pd
from .perovskite_utils import fetch_mp_data   # already exists in repo

CSV = Path(__file__).parent / "data" / "perovskite_bandgap_merged.csv"
EXP = pd.read_csv(CSV)

_cache: dict[str, float] = {}
def mp_gap(formula: str) -> float:
    if formula not in _cache:
        doc = fetch_mp_data(formula, ["band_gap"])
        _cache[formula] = doc["band_gap"] if doc else np.nan
    return _cache[formula]

def predict_gap(row: pd.Series, b: float = 0.30) -> float:
    m = re.match(r"CsSn(?P<x>[0-9.]+)Pb(?P<y>[0-9.]+)I3", row["Composition"])
    if not m:
        return np.nan
    x = float(m["x"])
    Eg_Sn, Eg_Pb = mp_gap("CsSnI3"), mp_gap("CsPbI3")
    return (1 - x) * Eg_Pb + x * Eg_Sn - b * x * (1 - x)

def validate(b: float = 0.30):
    df = EXP.copy()
    df["Eg_pred"] = df.apply(lambda r: predict_gap(r, b), axis=1)
    sub = df.dropna(subset=["Eg_pred"]).copy()
    sub["abs_err"] = (sub["Eg_pred"] - sub["Eg_eV"].astype(float)).abs()
    metrics = dict(
        N    = int(sub.shape[0]),
        MAE  = sub.abs_err.mean(),
        RMSE = np.sqrt((sub.abs_err**2).mean()),
        R2   = np.corrcoef(sub.Eg_pred, sub.Eg_eV.astype(float))[0, 1]**2
    )
    return metrics, sub[["Composition", "Eg_eV", "Eg_pred", "abs_err"]]

if __name__ == "__main__":
    m, res = validate()
    print(json.dumps(m, indent=2))
    res.to_csv("validation_residuals.csv", index=False)
    print("Residuals → validation_residuals.csv")
