# ─────────────────────────────────────────────────────────────────────────────
#  EnerMat backend  ·  Pb-free perovskite screening helpers
#  Fully self-contained; requires mp_api, pymatgen, pandas, numpy
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import os, math
import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Composition

# ── 1.  Materials-Project connection ────────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("Please export a valid 32-character MP_API_KEY")
mpr = MPRester(API_KEY)

# ── 2.  Constants & reference data ──────────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]  # dropdown list

IONIC_RADII = {            # Å   (Shannon / Pyykkö where required)
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

kT_EXP = 0.06             # eV – empirical “metastability temperature”
# ── 3.  Helpers ─────────────────────────────────────────────────────────────
def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    """Lowest-energy MP entry → {field: value} or None on failure."""
    docs = mpr.summary.search(formula=formula)
    if not docs:
        return None
    best = min(docs, key=lambda d: d.energy_above_hull or 1e6)
    return {f: getattr(best, f) for f in fields if hasattr(best, f)}


def score_band_gap(Eg: float, lo: float, hi: float) -> float:
    """Return 1.0 inside the [lo,hi] window and linearly taper outside."""
    if Eg < lo:
        return max(0.0, 1 - (lo - Eg) / (hi - lo))
    if Eg > hi:
        return max(0.0, 1 - (Eg - hi) / (hi - lo))
    return 1.0


def _exponential_stability(hull_eV: float) -> float:
    """exp(−E_hull / kT_exp) in the 0–1 range."""
    return math.exp(-max(hull_eV, 0) / kT_EXP)

# ── 4.  Binary grid scan ────────────────────────────────────────────────────
def mix_abx3(
    formula_A: str,
    formula_B: str,
    *,
    rh: float,
    temp: float,
    bg_window: tuple[float, float],
    bowing: float = 0.0,
    dx: float = 0.05,
    alpha: float = 1.0,
    beta: float  = 1.0,
) -> pd.DataFrame:
    """Return a DataFrame with columns x, Eg, stability, gap_score, score, …"""
    lo, hi = bg_window
    dA = fetch_mp_data(formula_A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(formula_B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    # Goldschmidt formability scores (needs A/B/X radii only once)
    comp  = Composition(formula_A)
    A_site = next(e.symbol for e in comp.elements if e.symbol in IONIC_RADII)
    B_site = next(e.symbol for e in comp.elements if e.symbol in {"Pb", "Sn"})
    X_site = next(e.symbol for e in comp.elements if e.symbol in {"I", "Br", "Cl"})
    rA, rB, rX = IONIC_RADII[A_site], IONIC_RADII[B_site], IONIC_RADII[X_site]

    rows: list[dict] = []
    for x in np.arange(0, 1 + 1e-6, dx):
        Eg    = (1 - x)*dA["band_gap"]         + x*dB["band_gap"]         - bowing*x*(1-x)
        Ehull = (1 - x)*dA["energy_above_hull"]+ x*dB["energy_above_hull"]
        stability = _exponential_stability(Ehull)
        gap_score = score_band_gap(Eg, lo, hi)

        # geometric form factor (unchanged from original paper)
        t  = (rA + rX) / (math.sqrt(2)*(rB + rX))
        mu = rB / rX
        form_score = math.exp(-0.5*((t-0.90)/0.07)**2) * math.exp(-0.5*((mu-0.50)/0.07)**2)

        env_pen = 1 + alpha*(rh/100) + beta*(temp/100)
        score   = form_score * stability * gap_score / env_pen

        rows.append(
            dict(
                x=round(x,3),
                Eg=round(Eg,3),
                stability=round(stability,3),
                gap_score=round(gap_score,3),
                score=round(score,3),
                formula=f"{formula_A}-{formula_B} x={x:.2f}",
            )
        )

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

# ── 5.  Ternary scan (pair-wise bowing) ─────────────────────────────────────
def screen_ternary(
    A: str,
    B: str,
    C: str,
    *,
    rh: float,
    temp: float,
    bg: tuple[float, float],
    bows: dict[str, float] | None = None,
    dx: float = 0.05,
    dy: float = 0.05,
    alpha: float = 1.0,
    beta: float  = 1.0,
) -> pd.DataFrame:
    """Grid scan of x, y (z = 1-x-y).  Adds stability & gap_score columns."""
    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    lo, hi = bg
    if bows is None:
        bows = {"AB": 0.0, "AC": 0.0, "BC": 0.0}

    rows: list[dict] = []
    for x in np.arange(0, 1 + 1e-6, dx):
        for y in np.arange(0, 1 - x + 1e-6, dy):
            z = 1 - x - y
            Eg = (
                z*dA["band_gap"] + x*dB["band_gap"] + y*dC["band_gap"]
                - bows["AB"]*x*z - bows["AC"]*y*z - bows["BC"]*x*y
            )
            Eh = (
                z*dA["energy_above_hull"] + x*dB["energy_above_hull"] + y*dC["energy_above_hull"]
                + bows["AB"]*x*z + bows["AC"]*y*z + bows["BC"]*x*y
            )
            stability = _exponential_stability(Eh)
            gap_score = score_band_gap(Eg, lo, hi)
            env_pen   = 1 + alpha*(rh/100) + beta*(temp/100)
            score     = stability * gap_score / env_pen

            rows.append(
                dict(
                    x=round(x,3),
                    y=round(y,3),
                    Eg=round(Eg,3),
                    stability=round(stability,3),
                    gap_score=round(gap_score,3),
                    score=round(score,3),
                )
            )

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

# convenience alias used by app.py
_summary = fetch_mp_data
# ─────────────────────────────────────────────────────────────────────────────
