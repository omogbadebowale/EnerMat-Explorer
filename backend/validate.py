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

# ------------------------------------------------------------------
_cache: dict[str, float] = {}
def mp_gap(formula: str) -> float:
    """
    Cached Materials-Project band-gap look-up.
    If the API call fails (rate-limit, network, etc.) fall back to
    NaN so the row is kept but skipped in the error metrics.
    """
    if formula in _cache:
        return _cache[formula]

    try:
        doc = fetch_mp_data(formula, ["band_gap"])
        val = doc["band_gap"] if doc else np.nan
    except Exception:                      # includes MPRestError
        val = np.nan

    _cache[formula] = val
    return val
# ------------------------------------------------------------------
SN_GAP = mp_gap("CsSnI3")   # ≈1.30 eV
PB_GAP = mp_gap("CsPbI3")   # ≈1.73 eV  (example)

# ------------------------------------------------------------------
def predict_gap(row: pd.Series, b: float = 0.30) -> float:
    """
    1. If the composition is a CsSn1-xPbxI3 alloy → Vegard + bowing
       (regex now tolerates trailing text like “… QDs (~4 nm)”).
    2. Otherwise fallback to Materials-Project band-gap for the
       cleaned formula (may be NaN if MP has no record).
    """
    comp = clean_formula(row["Composition"])

    # 1️⃣  Vegard + bowing for iodide Sn/Pb alloys
    m = re.match(r"CsSn(?P<x>[0-9.]+)Pb(?P<y>[0-9.]+)I3", comp)
    if m:
        x_pb = float(m["y"])                 # Pb fraction (Sn = 1-x)
        return (1 - x_pb) * SN_GAP + x_pb * PB_GAP - b * x_pb * (1 - x_pb)

    # 2️⃣  All other compositions – use Materials-Project gap
    return mp_gap(comp)                      # may be NaN → row dropped
# ------------------------------------------------------------------
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
