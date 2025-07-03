"""
backend.validate  – benchmark EXPERIMENTAL band-gaps

▪ Run   :  python -m backend.validate
▪ Import:  from backend.validate import validate
"""

from __future__ import annotations

# ── stdlib ───────────────────────────────────────────────────────────────
from pathlib import Path
import json, re, unicodedata
from functools import lru_cache

# ── third-party ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── project ───────────────────────────────────────────────────────────────
from .perovskite_utils import fetch_mp_data  # already in your repo

# ══════════════════════════════════════════════════════════════════════════
# 1.  Helpers
# ══════════════════════════════════════════════════════════════════════════
def clean_formula(raw: str) -> str:
    """
    Normalise unicode, drop text in brackets, kill spaces/dashes so that
        "CsSnI3 (γ-phase)"  →  "CsSnI3"
        "(FAPbI3)0.7(CsSnI3)0.3" → "FAPbI30.7CsSnI30.3"
    """
    txt = unicodedata.normalize("NFKD", raw).replace("γ", "")
    txt = re.sub(r"\(.*?\)", "", txt)          # delete (…) parts
    txt = txt.replace("–", "-").replace("—", "-").replace(" ", "")
    return txt.strip()


# keep only compositions the simple model understands
PATS = [
    re.compile(r"^CsSnI3$"),
    re.compile(r"^CsSnBr3$"),
    re.compile(r"^CsSnCl3$"),
    re.compile(r"^CsSn(?P<x>[0-9.]+)Pb(?P<y>[0-9.]+)I3$"),  # Sn/Pb iodide alloys
]

CSV = Path(__file__).parent / "data" / "perovskite_bandgap_merged.csv"
RAW = pd.read_csv(CSV)

def _match_any(comp: str) -> bool:
    comp = clean_formula(comp)
    return any(pat.search(comp) for pat in PATS)

EXP = RAW[RAW["Composition"].apply(_match_any)].copy()
print(f"[validate] keeping {EXP.shape[0]}/{RAW.shape[0]} rows that match model")

# ══════════════════════════════════════════════════════════════════════════
# 2.  Materials-Project band-gap cache
# ══════════════════════════════════════════════════════════════════════════
@lru_cache(maxsize=None)
def mp_gap(formula: str) -> float:
    """
    One-line MP look-up with silent failure → NaN.
    """
    try:
        doc = fetch_mp_data(formula, ["band_gap"])
        return doc["band_gap"] if doc else np.nan
    except Exception:                  # network, quota, 404 …
        return np.nan


SN_GAP = mp_gap("CsSnI3") or 1.30     # fall-back default
PB_GAP = mp_gap("CsPbI3") or 1.73

# ══════════════════════════════════════════════════════════════════════════
# 3.  Simple predictor
# ══════════════════════════════════════════════════════════════════════════
def predict_gap(row: pd.Series, b: float = 0.30) -> float:
    """
    ①  Vegard+bowing for CsSn₁₋ₓPbₓI₃ alloys (x = Pb fraction)
    ②  Else: single-compound gap from MP
    """
    comp = clean_formula(row["Composition"])

    m = re.match(r"CsSn(?P<x>[0-9.]+)Pb(?P<y>[0-9.]+)I3$", comp)
    if m:
        x_pb = float(m["y"])
        return (1 - x_pb) * SN_GAP + x_pb * PB_GAP - b * x_pb * (1 - x_pb)

    return mp_gap(comp)   # may be NaN → row dropped


# ══════════════════════════════════════════════════════════════════════════
# 4.  Public API
# ══════════════════════════════════════════════════════════════════════════
def validate(b: float = 0.30):
    """
    Returns
        metrics : dict   {N, MAE, RMSE, R2}
        resid   : DataFrame  (Composition, Eg_eV, Eg_pred, abs_err)
    """
    df = EXP.copy()
    df["Eg_pred"] = df.apply(lambda r: predict_gap(r, b), axis=1)

    sub = df.dropna(subset=["Eg_pred"]).copy()
    sub["abs_err"] = (sub.Eg_pred - sub.Eg_eV.astype(float)).abs()

    metrics = dict(
        N    = int(sub.shape[0]),
        MAE  = sub.abs_err.mean(),
        RMSE = np.sqrt((sub.abs_err**2).mean()),
        R2   = np.corrcoef(sub.Eg_pred, sub.Eg_eV.astype(float))[0, 1] ** 2
    )
    return metrics, sub[["Composition", "Eg_eV", "Eg_pred", "abs_err"]]


# ══════════════════════════════════════════════════════════════════════════
# 5.  CLI helper
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":            #  python -m backend.validate
    m, res = validate()
    print(json.dumps(m, indent=2))
    res.to_csv("validation_residuals.csv", index=False)
    print("Residuals → validation_residuals.csv")
