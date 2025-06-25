"""
perovskite_utils.py · EnerMat 2025-06-25
---------------------------------------
✓ picks the *lowest-enthalpy* Materials-Project polymorph
✓ exponential stability weight  exp(-Ehull / 0.06 eV)
✓ logarithmic humidity penalty  1 + α ln(1+RH) + β T/100
✓ stability & gap recorded in every DataFrame (no more “N/A”)
"""

from __future__ import annotations
import os, numpy as np, pandas as pd
from dotenv import load_dotenv; load_dotenv()
from mp_api.client import MPRester
from pymatgen.core import Composition
import streamlit as st    # harmless import when used outside Streamlit

# ── API key ──────────────────────────────────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 Set a valid 32-character $MP_API_KEY")

mpr = MPRester(API_KEY)

# ── Constants ────────────────────────────────────────────────────────────────
IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81,
}
STAB_KT = 0.06          # eV atom⁻¹ → ≈700 K “experimental kT”

# ── Helper: query lowest-energy polymorph ────────────────────────────────────
def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    docs = mpr.summary.search(formula=formula)
    if not docs:
        return None
    entry = min(docs, key=lambda d: d.energy_above_hull or 999)
    return {f: getattr(entry, f) for f in fields if hasattr(entry, f)}

# ── Band-gap fitness (unity in the target window) ────────────────────────────
def score_band_gap(Eg: float, lo: float, hi: float) -> float:
    if Eg < lo:
        return max(0.0, 1 - (lo - Eg) / (hi - lo))
    if Eg > hi:
        return max(0.0, 1 - (Eg - hi) / (hi - lo))
    return 1.0

# ── Binary screen A–B ────────────────────────────────────────────────────────
def mix_abx3(
    formula_A: str,
    formula_B: str,
    rh: float,
    temp: float,
    bg_window: tuple[float, float],
    bowing: float = 0.0,
    dx: float = 0.05,
    alpha: float = 1.0,
    beta:  float = 1.0,
) -> pd.DataFrame:
    lo, hi = bg_window
    dA = fetch_mp_data(formula_A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(formula_B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    comp = Composition(formula_A)
    A_site = next(e.symbol for e in comp.elements if e.symbol in IONIC_RADII)
    B_site = next(e.symbol for e in comp.elements if e.symbol in {"Pb", "Sn"})
    X_site = next(e.symbol for e in comp.elements if e.symbol in {"I", "Br", "Cl"})
    rA, rB, rX = IONIC_RADII[A_site], IONIC_RADII[B_site], IONIC_RADII[X_site]

    rows = []
    for x in np.arange(0, 1 + 1e-8, dx):
        Eg  = (1-x)*dA["band_gap"] + x*dB["band_gap"] - bowing*x*(1-x)
        Eh  = (1-x)*dA["energy_above_hull"] + x*dB["energy_above_hull"]
        stability  = np.exp(-max(Eh, 0)/STAB_KT)        # exp weight
        gap_score  = score_band_gap(Eg, lo, hi)
        # geometric tolerance factor penalty
        t  = (rA+rX)/(np.sqrt(2)*(rB+rX))
        mu =  rB/rX
        form_score = np.exp(-0.5*((t-0.90)/0.07)**2) * np.exp(-0.5*((mu-0.50)/0.07)**2)
        # environment penalty (logarithmic in RH)
        env_pen = 1 + alpha*np.log1p(rh) + beta*(temp/100)
        S = form_score*stability*gap_score / env_pen
        rows.append(
            dict(
                x=round(x,3), Eg=round(Eg,3),
                stability=round(stability,3),
                gap_score=round(gap_score,3),
                score=round(S,3),
                formula=f"{formula_A}-{formula_B} x={x:.2f}",
            )
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# ── Ternary screen A–B–C ─────────────────────────────────────────────────────
def screen_ternary(
    A: str, B: str, C: str,
    rh: float, temp: float,
    bg: tuple[float, float],
    bows: dict[str,float],
    dx: float = 0.05, dy: float = 0.05,
) -> pd.DataFrame:
    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    lo, hi = bg
    rows = []
    for x in np.arange(0, 1+1e-8, dx):
        for y in np.arange(0, 1 - x + 1e-8, dy):
            z = 1 - x - y
            Eg = (
                z*dA["band_gap"] + x*dB["band_gap"] + y*dC["band_gap"]
                - bows["AB"]*x*z - bows["AC"]*y*z - bows["BC"]*x*y
            )
            Eh = (
                z*dA["energy_above_hull"] + x*dB["energy_above_hull"] + y*dC["energy_above_hull"]
                + bows["AB"]*x*z + bows["AC"]*y*z + bows["BC"]*x*y
            )
            stability  = np.exp(-max(Eh,0)/STAB_KT)
            gap_score  = score_band_gap(Eg, lo, hi)
            env_pen    = 1 + alpha*np.log1p(rh) + beta*(temp/100)   # same form
            S          = stability*gap_score / env_pen
            rows.append(dict(
                x=round(x,3), y=round(y,3), Eg=round(Eg,3),
                stability=round(stability,3),
                score=round(S,3)
            ))
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# convenient alias for Streamlit
_summary = fetch_mp_data
END_MEMBERS = ["CsPbBr3","CsSnBr3","CsSnCl3","CsPbI3"]
