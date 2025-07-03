"""
backend.validate  – benchmark 27 experimental band-gaps
Run   :  python -m backend.validate
Import:  from backend.validate import validate
"""

from __future__ import annotations

# ── stdlib ─────────────────────────────────────────────────────────
from pathlib import Path
import json
import re
import unicodedata

# ── third-party ────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── internal ──────────────────────────────────────────────────────
from .perovskite_utils import fetch_mp_data   # already exists in repo


# ╭────────────────────────────────────────────────────────────────╮
# │  1. helpers                                                   │
# ╰────────────────────────────────────────────────────────────────╯
def clean_formula(raw: str) -> str:
    """
    Strip brackets, spaces, greek γ, etc. so strings such as
        (FAPbI3)0.7(CsSnI3)0.3   or   CsSnI3 (γ-phase)
    become parse-friendly:
        FAPbI30.7CsSnI30.3       or   CsSnI3
    """
    txt = unicodedata.normalize("NFKD", raw.strip()).replace("γ", "")
    txt = re.sub(r"\(.*?\)", "", txt)          # remove (…) parts
    txt = txt.replace("–", "-").replace("—", "-").replace(" ", "")
    return txt


# ╭────────────────────────────────────────────────────────────────╮
# │  2. MP band-gap cache                                         │
# ╰────────────────────────────────────────────────────────────────╯
_cache: dict[str, float] = {}


def mp_gap(formula: str) -> float:
    """
    Cached Materials-Project band-gap look-up.
    If MP raises MPRestError or returns no entry, fall back to NaN.
    """
    if formula in _cache:
        return _cache[formula]

    try:
        doc = fetch_mp_data(formula, ["band_gap"])
        val = doc["band_gap"] if doc else np.nan
    except Exception:
        val = np.nan

    _cache[formula] = val
    return val


# ╭────────────────────────────────────────────────────────────────╮
# │  3. load experimental CSV and append two anchor rows          │
# ╰────────────────────────────────────────────────────────────────╯
CSV = Path(__file__).parent / "data" / "perovskite_bandgap_merged.csv"
RAW = pd.read_csv(CSV)

ANCHORS = pd.DataFrame(
    {
        "Composition": ["CsSnI3", "CsPbI3"],
        "Eg_eV": [mp_gap("CsSnI3"), mp_gap("CsPbI3")],
        "Measurement_Comment": ["anchor", "anchor"],
        "Reference_URL": ["MP", "MP"],
    }
)

EXP = pd.concat([RAW, ANCHORS], ignore_index=True)


# ╭────────────────────────────────────────────────────────────────╮
# │  4. two end-member constants for Vegard model                 │
# ╰────────────────────────────────────────────────────────────────╯
SN_GAP = mp_gap("CsSnI3")  # ≈ 1.30 eV
PB_GAP = mp_gap("CsPbI3")  # ≈ 1.70 eV (example value)


def predict_gap(row: pd.Series, b: float = 0.30) -> float:
    """
    (1)  If the composition is a CsSn1-xPbxI3 alloy → Vegard + bowing.
         Regex allows trailing notes like “… QDs (~4 nm)”.
    (2)  Otherwise return mp_gap(clean_formula(comp)).
         NaN rows are kept but ignored in stats.
    """
    comp = clean_formula(row["Composition"])

    m = re.match(r"CsSn(?P<x>[0-9.]+)Pb(?P<y>[0-9.]+)I3", comp)
    if m:
        x_pb = float(m["y"])  # Pb fraction
        return (1 - x_pb) * SN_GAP + x_pb * PB_GAP - b * x_pb * (1 - x_pb)

    return mp_gap(comp)  # may be NaN (row dropped later)


# ╭────────────────────────────────────────────────────────────────╮
# │  5. public validate() interface                               │
# ╰────────────────────────────────────────────────────────────────╯
def validate(b: float = 0.30):
    """
    Returns (metrics_dict, residual_dataframe)
    metrics: N, MAE, RMSE, R²
    dataframe: Composition, Eg_eV, Eg_pred, abs_err
    """
    df = EXP.copy()
    df["Eg_pred"] = df.apply(lambda r: predict_gap(r, b), axis=1)

    sub = df.dropna(subset=["Eg_pred"]).copy()
    sub["abs_err"] = (sub.Eg_pred - sub.Eg_eV.astype(float)).abs()

    metrics = dict(
        N=int(sub.shape[0]),
        MAE=sub.abs_err.mean(),
        RMSE=np.sqrt((sub.abs_err**2).mean()),
        R2=np.corrcoef(sub.Eg_pred, sub.Eg_eV.astype(float))[0, 1] ** 2,
    )
    return metrics, sub[["Composition", "Eg_eV", "Eg_pred", "abs_err"]]


# ╭────────────────────────────────────────────────────────────────╮
# │  6. CLI helper                                                │
# ╰────────────────────────────────────────────────────────────────╯
if __name__ == "__main__":
    m, res = validate()
    print(json.dumps(m, indent=2))
    res.to_csv("validation_residuals.csv", index=False)
    print("Residuals → validation_residuals.csv")
