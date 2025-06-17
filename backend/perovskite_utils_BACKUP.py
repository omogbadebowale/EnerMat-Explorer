"""
Perovskite-utils — bug-fixed for mp-api ≥ 0.41
Only 120 lines, zero dependencies beyond numpy/pandas/pymatgen/mp-api.
"""

from __future__ import annotations
import os, functools, math
import numpy as np, pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition

# ── load Materials-Project API key ────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY")
if API_KEY is None or len(API_KEY) < 22:
    raise RuntimeError("🛑  MP_API_KEY missing or malformed; edit .env.")

mpr = MPRester(API_KEY)

# ── constants ─────────────────────────────────────────────────────────────────
IONIC_RADII = {"Cs":1.88,"Pb":1.19,"Sn":1.18,"I":2.20,"Br":1.96,"Cl":1.81}
END_MEMBERS = ["CsPbBr3","CsSnBr3","CsSnCl3","CsPbI3"]

# ── helper: fetch first stable SummaryDoc (FIXED) ─────────────────────────────
@functools.lru_cache(maxsize=64)
def _summary(formula: str):
    """
    Materials-Project mp-api now returns a LIST, not an iterator.
    We return the first stable entry, or None if nothing found.
    """
    docs = mpr.summary.search(
        formula=formula,
        fields=["band_gap", "energy_above_hull", "is_stable"]
    )
    if not docs:                 # empty list
        return None
    # pick first stable
    for doc in docs:
        if getattr(doc, "is_stable", True):
            return doc
    return docs[0]               # else give the first anyway

# ── helper: simple band-gap proximity score (0-1) ─────────────────────────────
def _gap_score(gap: float, lo: float, hi: float) -> float:
    if lo <= gap <= hi:
        return 1.0
    d = min(abs(gap - lo), abs(gap - hi))
    return max(0.0, 1 - d / (hi - lo))

# ── main screening routine (unchanged logic) ─────────────────────────────────
def screen(A: str, B: str, rh: float, temp: float,
           bg: tuple[float, float],
           bow: float = 0.3, dx: float = 0.05) -> pd.DataFrame:
    lo, hi = bg
    dA, dB = _summary(A), _summary(B)
    if not dA or not dB:
        return pd.DataFrame()

    rows = []
    for x in np.around(np.arange(0, 1 + 1e-6, dx), 3):
        Eg   = (1 - x) * dA.band_gap + x * dB.band_gap - bow * x * (1 - x)
        stab = max(0.0, 1 - dA.energy_above_hull)
        score = stab * _gap_score(Eg, lo, hi)

        rows.append({
            "x": x,
            "band_gap": round(Eg, 3),
            "stability": round(stab, 3),
            "score": round(score, 3),
            "formula": f"{A}-{B}  x={x:.2f}"
        })

    return (pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True))
