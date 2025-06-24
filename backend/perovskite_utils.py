"""
EnerMat backend utilities – binary & ternary ABX₃ screening
Version : v9.6.1   (2025-06-24)
Changes : • stability column restored in ternary
          • bow-sign fix in Ehull
          • humidity/temperature penalty added to ternary
          • zero-width gap-window guard
          • MP fetch lru-cached + simple back-off
"""

from __future__ import annotations
import os, time, functools
from dotenv import load_dotenv; load_dotenv()

import numpy as np
import pandas as pd
import streamlit as st
from mp_api.client import MPRester
from pymatgen.core import Composition   # retained for future use

# ── Materials-Project key ───────────────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 MP_API_KEY missing or malformed.")
mpr = MPRester(API_KEY)

# ── End-member presets & ionic radii ────────────────────────────────────
END_MEMBERS = ["CsSnBr3", "CsSnCl3", "CsSnI3", "CsPbBr3"]

IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72,
    "Pb": 1.19, "Sn": 1.18,
    "I" : 2.20, "Br": 1.96, "Cl": 1.81,
}

# ── Helpers ─────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=256)
def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    """Slim dict of requested fields from the first MP entry."""
    try:
        docs = mpr.summary.search(formula=formula)
    except Exception:
        time.sleep(1)                # basic back-off
        docs = mpr.summary.search(formula=formula)
    if not docs:
        return None
    return {f: getattr(docs[0], f, None) for f in fields}

def score_band_gap(bg: float, lo: float, hi: float) -> float:
    """0–1 score for how well `bg` sits in [lo, hi]."""
    if hi == lo:                       # guard zero-width window
        return 1.0 if abs(bg-lo) < 1e-3 else 0.0
    if bg < lo:
        return max(0.0, 1 - (lo - bg) / (hi - lo))
    if bg > hi:
        return max(0.0, 1 - (bg - hi) / (hi - lo))
    return 1.0

def env_penalty(rh: float, temp: float,
                alpha: float=1.0, beta: float=1.0) -> float:
    """Humidity/temperature penalty Γ(RH,T)."""
    return 1 + alpha * (rh/100) + beta * (temp/100)

# ── Binary screening ────────────────────────────────────────────────────
def mix_abx3(
    formula_A: str,
    formula_B: str,
    rh: float,
    temp: float,
    bg_window: tuple[float, float],
    bowing: float = 0.0,
    dx: float = 0.05,
) -> pd.DataFrame:
    lo, hi = bg_window
    dA = fetch_mp_data(formula_A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(formula_B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    rows = []
    for x in np.linspace(0, 1, int(1/dx)+1):
        Eg = (1-x)*dA["band_gap"] + x*dB["band_gap"] - bowing*x*(1-x)
        Eh = (1-x)*dA["energy_above_hull"] + x*dB["energy_above_hull"]
        stability = max(0.0, 1 - Eh)           # 1 = on hull, 0 = ≥1 eV/atom
        gap_score = score_band_gap(Eg, lo, hi)
        score = stability * gap_score / env_penalty(rh, temp)
        rows.append(dict(
            x=round(x,3), Eg=round(Eg,3),
            stability=round(stability,3),
            score=round(score,3),
            formula=f"{formula_A}-{formula_B} x={x:.2f}",
        ))
    return (pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True))

# ── Ternary screening ──────────────────────────────────────────────────
def screen_ternary(
    A: str, B: str, C: str,
    rh: float, temp: float,
    bg: tuple[float,float],
    bows: dict[str,float],
    dx: float=0.05, dy: float=0.05,
) -> pd.DataFrame:

    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    lo, hi = bg
    rows = []
    for x in np.linspace(0, 1, int(1/dx)+1):
        for y in np.linspace(0, 1-x, int((1-x)/dy)+1):
            z = 1 - x - y
            Eg = (
                z*dA["band_gap"] + x*dB["band_gap"] + y*dC["band_gap"]
                - bows["AB"]*x*z - bows["AC"]*y*z - bows["BC"]*x*y
            )
            Eh = (
                z*dA["energy_above_hull"] + x*dB["energy_above_hull"] + y*dC["energy_above_hull"]
                - bows["AB"]*x*z - bows["AC"]*y*z - bows["BC"]*x*y    # sign fixed
            )
            stability = np.exp(-max(Eh,0)/0.1)        # 0-1 scale
            score = stability * score_band_gap(Eg, lo, hi) / env_penalty(rh, temp)
            rows.append(dict(
                x=round(x,3), y=round(y,3),
                Eg=round(Eg,3),
                stability=round(stability,3),
                score=round(score,3),
            ))
    return (pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True))

# alias for older imports
_summary = fetch_mp_data
