"""
backend/perovskite_utils.py
EnerMat Perovskite Explorer – CLEAN 2025-07-13

4-space indents • calibrated band gaps • strict optical filter
Binary + ternary screening with oxidation-energy term
"""

from __future__ import annotations

import functools
import os
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition
import streamlit as st

# ───────────────────────────────────────────────────────────── API KEY
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32-character MP_API_KEY missing in env or secrets")
mpr = MPRester(API_KEY)

# ─────────────────────────────────────────────── presets & corrections
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

CALIBRATED_GAPS = {
    "CsSnBr3": 1.79,
    "CsSnCl3": 2.83,
    "CsSnI3": 1.30,
    "CsPbBr3": 2.30,
    "CsPbI3": 1.73,
}
GAP_OFFSET = {"I": 0.90, "Br": 0.70, "Cl": 0.80}

IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81,
}
# ── mini-helper: return first halogen symbol contained in the formula ──────────
def _find_halide(formula: str) -> str:
    """
    Pick out the halide (I / Br / Cl) that appears in the chemical formula.

    >>> _find_halide("CsSnBr3")   -> "Br"
    >>> _find_halide("CsPbI3")    -> "I"
    """
    return next(h for h in ("I", "Br", "Cl") if h in formula)

# ──────────────────────────────────────────────────────────── helpers
def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    """Return MP summary dict; apply calibrated / offset gap **only if requested**."""
    doc = mpr.summary.search(formula=formula, fields=tuple(fields))
    if not doc:
        return None
    d: dict = {f: getattr(doc[0], f, None) for f in fields}

    # --- gap correction is needed only when the caller asked for "band_gap" ---
    if "band_gap" in fields:
        if formula in CALIBRATED_GAPS:
            d["band_gap"] = CALIBRATED_GAPS[formula]
        else:
            hal = _find_halide(formula)
            d["band_gap"] = (d["band_gap"] or 0) + GAP_OFFSET[hal]

    return d


# optical test – strict 0/1
def score_band_gap(Eg: float, lo: float, hi: float) -> float:
    return 1.0 if lo <= Eg <= hi else 0.0

# ───────── Oxidation driving force  CsSnX3 + ½O2 → ½Cs2SnX6 + ½SnO2
@functools.lru_cache(maxsize=None)
def oxidation_energy(formula: str) -> float:
    hal = _find_halide(formula)
    reac   = fetch_mp_data(f"CsSn{hal}3",  ["energy_per_atom"])["energy_per_atom"]
    prod1  = fetch_mp_data(f"Cs2Sn{hal}6", ["energy_per_atom"])["energy_per_atom"]
    prod2  = fetch_mp_data("SnO2",         ["energy_per_atom"])["energy_per_atom"]
    e_o2   = fetch_mp_data("O2",           ["energy_per_atom"])["energy_per_atom"]*2
    # ½ Cs2SnX6 + ½ SnO2  –  CsSnX3  –  ½ O2
    return 0.5*prod1 + 0.5*prod2 - reac - 0.5*e_o2   # eV per Sn

# ───────────────────────────────────────────────────── Binary screen
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

    comp = Composition(formula_A)  # use A only to get radii
    A_site = next(e.symbol for e in comp.elements if e.symbol in IONIC_RADII)
    B_site = next(e.symbol for e in comp.elements if e.symbol in {"Pb", "Sn"})
    X_site = _find_halide(formula_A)
    rA, rB, rX = IONIC_RADII[A_site], IONIC_RADII[B_site], IONIC_RADII[X_site]

    e_ox = oxidation_energy(formula_A)  # proxy for the entire series

    rows = []
    for x in np.arange(0.0, 1.0 + 1e-6, dx):
        Eg    = (1-x)*dA["band_gap"] + x*dB["band_gap"] - bowing*x*(1-x)
        Ehull = (1-x)*dA["energy_above_hull"] + x*dB["energy_above_hull"]
        stab  = max(0.0, 1 - Ehull / 0.025)          # 0 → 25 meV/atom window
        gap   = score_band_gap(Eg, lo, hi)
        t     = (rA + rX) / (math.sqrt(2)*(rB + rX))
        mu    = rB / rX
        form  = math.exp(-0.5*((t-0.90)/0.07)**2)*math.exp(-0.5*((mu-0.50)/0.07)**2)
        env   = 1 + alpha*rh/100 + beta*temp/100
        ox_pen= math.exp(e_ox/0.2)                   # ΔEox ≤0 → penalty ≤1
        score = form*stab*gap*ox_pen/env

        rows.append({
            "x": round(x,3),
            "Eg": round(Eg,3),
            "Ehull": round(Ehull,4),
            "Eox": round(e_ox,3),
            "score": round(score,3),
            "formula": f"{formula_A}-{formula_B} x={x:.2f}",
        })

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# ───────────────────────────────────────────────────── Ternary screen
def screen_ternary(
    A: str, B: str, C: str,
    rh: float, temp: float,
    bg: tuple[float, float],
    bows: dict[str, float] | None = None,
    dx: float = 0.10, dy: float = 0.10,
) -> pd.DataFrame:

    bows = bows or {"AB": 0.0, "AC": 0.0, "BC": 0.0}
    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    e_ox = oxidation_energy(A)  # simple proxy: use halide of A
    lo, hi = bg
    rows = []
    for x in np.arange(0.0, 1.0 + 1e-6, dx):
        for y in np.arange(0.0, 1.0 - x + 1e-6, dy):
            z = 1 - x - y
            Eg = (z*dA["band_gap"] + x*dB["band_gap"] + y*dC["band_gap"]
                 - bows["AB"]*x*z - bows["AC"]*y*z - bows["BC"]*x*y)
            Eh = (z*dA["energy_above_hull"] + x*dB["energy_above_hull"] + y*dC["energy_above_hull"]
                 + bows["AB"]*x*z + bows["AC"]*y*z + bows["BC"]*x*y)
            stab = math.exp(-max(Eh,0)/0.025)
            gap  = score_band_gap(Eg, lo, hi)
            score = stab*gap*math.exp(e_ox/0.2)

            rows.append({
                "x": round(x,3),
                "y": round(y,3),
                "Eg": round(Eg,3),
                "Ehull": round(Eh,4),
                "Eox": round(e_ox,3),
                "score": round(score,3),
                "formula": f"{A}-{B}-{C} x={x:.2f} y={y:.2f}",
            })

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# ──────────────────────────────────────────────────────── legacy name
_summary = fetch_mp_data
