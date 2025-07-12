"""
EnerMat Perovskite Explorer – backend/perovskite_utils.py
FULL BACKEND (2025‑07‑12) 
• calibrated 0‑K hull stability
• strict optical filter
• Sn²⁺ → Sn⁴⁺ oxidation penalty (ΔEₒₓ)
• formula column for binary + ternary
• four‑space indents only
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition

load_dotenv()

# ────────────────────────────────────────────────────
#  API KEY
# ────────────────────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32‑character MP_API_KEY missing")

mpr = MPRester(API_KEY)

# ────────────────────────────────────────────────────
#  STATIC DATA
# ────────────────────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

# Fully scissored experimental gaps for quick calibration
CALIBRATED_GAPS = {
    "CsSnBr3": 1.79,
    "CsSnCl3": 2.83,
    "CsSnI3" : 1.30,
    "CsPbBr3": 2.30,
    "CsPbI3" : 1.73,
}

# If a composition is missing from the dict above we apply an element‑wise offset
GAP_OFFSET = {"I": 0.90, "Br": 0.70, "Cl": 0.80}

IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

# Materials Project formation energies for oxidation products are reused repeatedly → small cache
_CACHE: dict[str, float] = {}

# ────────────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────────────

def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    """Return a dict with calibrated band‑gap; safe against MP hash issues."""
    fields_t = tuple(fields)
    docs = mpr.summary.search(formula=formula, fields=fields_t)
    if not docs:
        return None
    entry = docs[0]
    data = {f: getattr(entry, f, None) for f in fields}

    # band‑gap correction
    if formula in CALIBRATED_GAPS:
        data["band_gap"] = CALIBRATED_GAPS[formula]
    else:
        hal = next(h for h in ("I", "Br", "Cl") if h in formula)
        data["band_gap"] = (data["band_gap"] or 0.0) + GAP_OFFSET[hal]
    return data


def oxidation_energy(formula_sn2: str, hal: str) -> float:
    """ΔE per Sn for:  CsSnX₃ + ½ O₂ → ½ Cs₂SnX₆ + ½ SnO₂ (exergonic ⇒ < 0)."""
    key = f"{formula_sn2}|{hal}"
    if key in _CACHE:
        return _CACHE[key]

    # reactant (per atom energy)
    e_reac = fetch_mp_data(formula_sn2, ["energy_per_atom"])["energy_per_atom"]

    # products averaged per Sn
    e_cs2snx6 = fetch_mp_data(f"Cs2Sn{hal}6", ["energy_per_atom"])["energy_per_atom"]
    e_sno2    = fetch_mp_data("SnO2", ["energy_per_atom"])["energy_per_atom"]
    e_prod    = (e_cs2snx6 + e_sno2) / 2.0

    # Materials Project O₂ energy reference (incl. 1.36 eV correction)
    e_o2 = -9.86   # eV per O₂ molecule

    dE = (e_prod + 0.5 * e_o2) - e_reac
    _CACHE[key] = dE
    return dE


# strict optical gate
score_band_gap = lambda Eg, lo, hi: 1.0 if lo <= Eg <= hi else 0.0

K_T_EFF = 0.20   # eV – softness for oxidation penalty  (≈ 8 k_BT at 300 K)

# ────────────────────────────────────────────────────
#  BINARY  A–B  SCREEN
# ────────────────────────────────────────────────────

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

    # oxidation energy for the end‑member used across the loop
    dE_ox_A = oxidation_energy(formula_A, X_site)
    dE_ox_B = oxidation_energy(formula_B, X_site)

    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-6, dx):
        # band‑gap with bowing
        Eg = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bowing * x * (1 - x)

        # convex‑hull stability
        Ehull = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        stab  = max(0.0, 1 - Ehull)

        # oxidation penalty (linear interpolation)
        dEox   = (1 - x) * dE_ox_A + x * dE_ox_B
        ox_pen = np.exp(-max(dEox, 0.0) / K_T_EFF)

        # optical merit
        gap = score_band_gap(Eg, lo, hi)

        # geometric formability
        t  = (rA + rX) / (np.sqrt(2) * (rB + rX))
        mu = rB / rX
        form = np.exp(-0.5 * ((t - 0.90) / 0.07) ** 2) * np.exp(-0.5 * ((mu - 0.50) / 0.07) ** 2)

        # environment (humidity / temp)
        env = 1 + alpha * rh / 100 + beta * temp / 100

        score = form * stab * gap * ox_pen / env

        rows.append({
            "x":      round(x, 3),
            "Eg":     round(Eg, 3),
            "Ehull":  round(Ehull, 4),
            "Eox":    round(dEox, 3),
            "score":  round(score, 3),
            "formula": f"{formula_A}-{formula_B} x={x:.2f}",
        })

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

# ────────────────────────────────────────────────────
#  TERNARY  A–B–C  SCREEN
# ────────────────────────────────────────────────────

def screen_ternary(
    A: str, B: str, C: str,
    rh: float, temp: float,
    bg: tuple[float, float],
    bows: dict[str, float],
    dx: float = 0.1, dy: float = 0.1,
    n_mc: int = 200,
) -> pd.DataFrame:

    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (
