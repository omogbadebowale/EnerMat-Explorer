"""
backend/perovskite_utils.py
──────────────────────────
Light-weight helpers for EnerMat Streamlit front-end.

Changes vs. v9.6
────────────────
✓ fetch_mp_data now picks the **lowest-Ehull** polymorph
✓ stability weight uses an exponential form  exp(–Ehull / kT*)  with kT*=60 meV
✓ both binary and ternary rows include  "gap_score"  and "stability"
✓ screen_ternary gains alpha / beta moisture–temperature penalty
"""

from __future__ import annotations
import os, math
from typing import List, Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition

# ── environment ──────────────────────────────────────────────────────────────
load_dotenv()
MP_KEY = os.getenv("MP_API_KEY")
if not (MP_KEY and len(MP_KEY) == 32):
    raise RuntimeError("🛑  Set a valid 32-character MP_API_KEY")

mpr = MPRester(MP_KEY)

# ── convenience lists ────────────────────────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

IONIC_RADII = {  # Å
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18,
    "I": 2.20,  "Br": 1.96, "Cl": 1.81,
}

# ── helper ───────────────────────────────────────────────────────────────────
def fetch_mp_data(formula: str, fields: List[str]) -> Dict | None:
    """Return lowest-energy entry’s requested fields (or None)."""
    docs = mpr.summary.search(formula=formula, fields=fields + ["energy_above_hull"])
    if not docs:
        return None
    entry = min(docs, key=lambda d: d.energy_above_hull)
    return {f: getattr(entry, f) for f in fields if hasattr(entry, f)}


def score_band_gap(bg: float, lo: float, hi: float) -> float:
    """Triangular window centred on [lo, hi]."""
    if bg < lo:
        return max(0.0, 1 - (lo - bg) / (hi - lo))
    if bg > hi:
        return max(0.0, 1 - (bg - hi) / (hi - lo))
    return 1.0


# ── binary screen ────────────────────────────────────────────────────────────
def mix_abx3(
    formula_A: str,
    formula_B: str,
    rh: float,
    temp: float,
    bg_window: tuple[float, float],
    bowing: float = 0.00,
    dx: float = 0.05,
    alpha: float = 1.0,
    beta:  float = 1.0,
) -> pd.DataFrame:
    """Scan binary A-B line with step dx."""
    lo, hi = bg_window
    dA = fetch_mp_data(formula_A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(formula_B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()      # MP query failed → empty result

    # tolerance-factor geometry
    comp = Composition(formula_A)
    A_site = next(e.symbol for e in comp.elements if e.symbol in IONIC_RADII)
    B_site = next(e.symbol for e in comp.elements if e.symbol in {"Pb", "Sn"})
    X_site = next(e.symbol for e in comp.elements if e.symbol in {"I", "Br", "Cl"})
    rA, rB, rX = IONIC_RADII[A_site], IONIC_RADII[B_site], IONIC_RADII[X_site]

    rows: List[Dict] = []
    for x in np.arange(0, 1 + 1e-6, dx):
        Eg   = (1 - x) * dA["band_gap"]         + x * dB["band_gap"]         - bowing * x * (1 - x)
        Eh   = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        stability  = math.exp(-max(Eh, 0) / 0.06)      # kT* ≈ 60 meV
        gap_score  = score_band_gap(Eg, lo, hi)

        # formability (t, μ) probability
        t  = (rA + rX) / (math.sqrt(2) * (rB + rX))
        mu = rB / rX
        form_score = math.exp(-0.5 * ((t  - 0.90) / 0.07) ** 2) * \
                     math.exp(-0.5 * ((mu - 0.50) / 0.07) ** 2)

        env_pen = 1 + alpha * (rh / 100) + beta * (temp / 100)
        score   = form_score * stability * gap_score / env_pen

        rows.append(
            dict(
                x=round(x, 3),
                Eg=round(Eg, 3),
                stability=round(stability, 3),
                gap_score=round(gap_score, 3),
                score=round(score, 3),
                formula=f"{formula_A}-{formula_B} x={x:.2f}",
            )
        )

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


# ── ternary screen ───────────────────────────────────────────────────────────
def screen_ternary(
    A: str,
    B: str,
    C: str,
    rh: float,
    temp: float,
    bg: tuple[float, float],
    bows: dict[str, float],
    dx: float = 0.10,
    dy: float = 0.10,
    alpha: float = 1.0,
    beta:  float = 1.0,
) -> pd.DataFrame:
    """Regular-grid ternary scan (z = 1-x-y)."""
    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    lo, hi = bg
    rows: List[Dict] = []
    for x in np.arange(0, 1 + 1e-6, dx):
        for y in np.arange(0, 1 - x + 1e-6, dy):
            z = 1 - x - y
            Eg = (
                z * dA["band_gap"] + x * dB["band_gap"] + y * dC["band_gap"]
                - bows["AB"] * x * z
                - bows["AC"] * y * z
                - bows["BC"] * x * y
            )
            Eh = (
                z * dA["energy_above_hull"] + x * dB["energy_above_hull"] + y * dC["energy_above_hull"]
                + bows["AB"] * x * z
                + bows["AC"] * y * z
                + bows["BC"] * x * y
            )

            stability = math.exp(-max(Eh, 0) / 0.06)
            gap_score = score_band_gap(Eg, lo, hi)
            env_pen   = 1 + alpha * (rh / 100) + beta * (temp / 100)
            score     = stability * gap_score / env_pen

            rows.append(
                dict(
                    x=round(x, 3),
                    y=round(y, 3),
                    Eg=round(Eg, 3),
                    stability=round(stability, 3),
                    gap_score=round(gap_score, 3),
                    score=round(score, 3),
                )
            )

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

# shortcut used by app.py for single-entry summaries
_summary = fetch_mp_data
