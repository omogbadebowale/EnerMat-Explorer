"""
backend.validate  – benchmark 27 experimental band-gaps
Run   :  python -m backend.validate
Import:  from backend.validate import validate
"""
from __future__ import annotations
from pathlib import Path
import json, re, unicodedata
import numpy as np, pandas as pd

from .perovskite_utils import fetch_mp_data   # already exists in repo

# ────────────────────────────────────────────────────────────────
# Helper: strip brackets / spaces / greek γ so regex can match
# ────────────────────────────────────────────────────────────────
def clean_formula(raw: str) -> str:
    txt = unicodedata.normalize("NFKD", raw.strip()).replace("γ", "")
    txt = re.sub(r"\(.*?\)", "", txt)                 # remove (…) parts
    txt = txt.replace("–", "-").replace("—", "-").replace(" ", "")
    return txt
# ────────────────────────────────────────────────────────────────

CSV = Path(__file__).parent / "data" / "perovskite_bandgap_merged.csv"
EXP = pd.read_csv(CSV)

_cache: dict[str, float] = {}
def mp_gap(formula: str) -> float:
    """Cached Materials-Project band-gap lookup (SnI3 / PbI3 end-members)."""
    if formula not in _cache:
        doc = fetch_mp_data(formula, ["band_gap"])
        _cache[formula] = doc["band_gap"] if doc else np.nan
    return _cache[formula]

SN_GAP = mp_gap("CsSnI3")   # ≈1.30 eV
PB_GAP = mp_gap("CsPbI3")   # ≈1.73 eV  (example)

def predict_gap(row: pd.Series, b: float = 0.30) -> float:
    """
    Vegard + bowing model for CsSn₁₋ₓPbₓI₃ alloys.
    Any row that is NOT an iodide Sn/Pb alloy returns NaN → dropped later.
    """
    comp = clean_formula(row["Composition"])
    m = re.match(r"CsSn(?P<x>[0-9.]+)Pb(?P<y>[0-9.]+)I3$", comp)
    if not m:
        return np.nan
    x = float(m["y"])                    # Pb fraction (Sn = 1-x)
    return (1 - x) * SN_GAP + x * PB_GAP - b * x * (1 - x)

def validate(b: float = 0.30):
    df = EXP.copy()
    df["Eg_pred"] = df.apply(lambda r: predict_gap(r, b), axis=1)
    sub = df.dropna(subset=["Eg_pred"]).copy()

    sub["abs_err"] = (sub.Eg_pred - sub.Eg_eV.astype(float)).abs()
    metrics = dict(
        N    = int(sub.shape[0]),
        MAE  = sub.abs_err.mean(),
        RMSE = np.sqrt((sub.abs_err ** 2).mean()),
        R2   = np.corrcoef(sub.Eg_pred, sub.Eg_eV.astype(float))[0, 1] ** 2
    )
    return metrics, sub[["Composition", "Eg_eV", "Eg_pred", "abs_err"]]

# CLI helper ------------------------------------------------------
if __name__ == "__main__":
    m, res = validate()
    print(json.dumps(m, indent=2))
    res.to_csv("validation_residuals.csv", index=False)
    print("Residuals → validation_residuals.csv")
