# backend/perovskite_utils.py
# ---------------------------------------------------------------------
# Author: EnerMat Explorer
# 2025-06-25  – unified binary/ternary scoring & exponential hull weight
# ---------------------------------------------------------------------
from __future__ import annotations
import os, math
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition
import streamlit as st

# ── 1.  Materials-Project key ─────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 Set a valid 32-character MP_API_KEY")
mpr = MPRester(API_KEY, monty_decode=False)

# ── 2.  Reference data ────────────────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]
IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

# ── 3.  Utilities ─────────────────────────────────────────────────────
def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    """Ground-state entry for `formula` with the requested fields."""
    docs = mpr.summary.search(formula=formula)
    if not docs:
        return None
    entry = min(docs, key=lambda d: d.energy_above_hull or 0)
    return {f: getattr(entry, f) for f in fields}

def score_band_gap(eg: float, lo: float, hi: float) -> float:
    if eg < lo:
        return max(0.0, 1 - (lo - eg) / (hi - lo))
    if eg > hi:
        return max(0.0, 1 - (eg - hi) / (hi - lo))
    return 1.0

def _exp_stability(ehull: float, k_eff: float = 0.06) -> float:
    """e^(–Ehull / k_eff)  with  k_eff≈0.06 eV mimicking metastability window."""
    return math.exp(-max(ehull, 0.0) / k_eff)

# ── 4.  Binary scan ───────────────────────────────────────────────────
def mix_abx3(
    formula_A: str,
    formula_B: str,
    rh: float,
    temp: float,
    bg_window: tuple[float, float],
    bowing: float = 0.0,
    dx: float = 0.05,
    alpha: float = 1.0,
    beta: float = 1.0,
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

    rows: list[dict] = []
    for x in np.arange(0, 1 + 1e-6, dx):
        Eg   = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bowing * x * (1 - x)
        Eh   = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        stab = _exp_stability(Eh)

        gap_score = score_band_gap(Eg, lo, hi)

        t  = (rA + rX) / (np.sqrt(2) * (rB + rX))
        mu = rB / rX
        form_score = math.exp(-0.5 * ((t  - 0.90)/0.07)**2) * \
                     math.exp(-0.5 * ((mu - 0.50)/0.07)**2)

        env_pen = 1 + alpha * (rh/100) + beta * (temp/100)
        S = form_score * stab * gap_score / env_pen

        rows.append({
            "x": round(x,3),
            "Eg": round(Eg,3),
            "stability": round(stab,3),
            "gap_score": round(gap_score,3),
            "score": round(S,3),
            "formula": f"{formula_A}-{formula_B} x={x:.2f}",
        })

    return (pd.DataFrame(rows)
              .sort_values("score", ascending=False)
              .reset_index(drop=True))

# ── 5.  Ternary scan ──────────────────────────────────────────────────
def screen_ternary(
    A: str, B: str, C: str,
    rh: float, temp: float,
    bg: tuple[float,float],
    bows: dict[str,float],
    dx: float = 0.1, dy: float = 0.1,
    alpha: float = 1.0, beta: float = 1.0,
) -> pd.DataFrame:
    dA = fetch_mp_data(A, ["band_gap","energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap","energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap","energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    lo, hi = bg
    rows: list[dict] = []

    for x in np.arange(0, 1+1e-6, dx):
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
            stab      = _exp_stability(Eh)
            gap_score = score_band_gap(Eg, lo, hi)
            env_pen   = 1 + alpha*(rh/100) + beta*(temp/100)
            S         = stab * gap_score / env_pen

            rows.append({
                "x": round(x,3), "y": round(y,3),
                "Eg": round(Eg,3),
                "stability": round(stab,3),
                "gap_score": round(gap_score,3),
                "score": round(S,3),
            })

    return (pd.DataFrame(rows)
              .sort_values("score", ascending=False)
              .reset_index(drop=True))

# back-compat alias
_summary = fetch_mp_data
