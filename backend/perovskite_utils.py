"""
EnerMat Perovskite Explorer – backend/perovskite_utils.py
Clean build • 2025‑07‑12 🟢
• calibrated experimental gaps
• strict optical window (step 0/1)
• convex‑hull lattice stability (Ehull)
• Sn²⁺→Sn⁴⁺ oxidation penalty ΔEox
• tidy binary + ternary screens, VALID Python
"""

from __future__ import annotations
import math, os
from functools import lru_cache

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from mp_api.client import MPRester
from pymatgen.core import Composition

# ─────────────────────────  API key  ──────────────────────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32‑character MP_API_KEY missing in env or secrets")

mpr = MPRester(API_KEY)

# ────────────────────── reference tables ─────────────────────
END_MEMBERS = [
    "CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3",
]

# experimentally calibrated gaps (eV)
CALIBRATED_GAPS = {
    "CsSnBr3": 1.79,
    "CsSnCl3": 2.83,
    "CsSnI3":  1.30,
    "CsPbBr3": 2.30,
    "CsPbI3":  1.73,
}

# generic offsets when no explicit calibration
GAP_OFFSET = {"I": 0.90, "Br": 0.70, "Cl": 0.80}

IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

# Materials‑Project O₂ energy (with FERE correction) per molecule
E_O2 = -9.86  # eV / O2
# effective temperature used in oxidation penalty
K_T_EFF = 0.20  # eV

# ───────────────────────── helpers ───────────────────────────

def fetch_mp_data(formula: str, fields: list[str]):
    """Return a dict of requested fields, with calibrated band‑gap."""
    docs = mpr.summary.search(formula=formula, fields=tuple(fields))
    if not docs:
        return None
    entry = docs[0]
    out = {f: getattr(entry, f, None) for f in fields}
    # apply gap calibration / offset
    if "band_gap" in fields:
        if formula in CALIBRATED_GAPS:
            out["band_gap"] = CALIBRATED_GAPS[formula]
        else:
            hal = next(h for h in ("I", "Br", "Cl") if h in formula)
            out["band_gap"] = (out["band_gap"] or 0.0) + GAP_OFFSET[hal]
    return out

@lru_cache(maxsize=64)
def oxidation_energy(formula_sn2: str, hal: str) -> float:
    """ΔE per Sn for   CsSnX3 + ½ O2 → ½ Cs2SnX6 + ½ SnO2  (negative → easy)."""
    e_reac = fetch_mp_data(formula_sn2, ["energy_per_atom"])["energy_per_atom"]
    e_prod1 = fetch_mp_data(f"Cs2Sn{hal}6", ["energy_per_atom"])["energy_per_atom"]
    e_prod2 = fetch_mp_data("SnO2", ["energy_per_atom"])["energy_per_atom"]
    e_products = (e_prod1 + e_prod2) / 2.0
    return (e_products + 0.5 * E_O2) - e_reac

score_band_gap = lambda Eg, lo, hi: 1.0 if lo <= Eg <= hi else 0.0

# ───────────────────────── Binary screen ─────────────────────

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

    hal = next(h for h in ("I", "Br", "Cl") if h in formula_A)
    rA, rB, rX = (IONIC_RADII[s] for s in ("Cs", "Sn", hal))
    dEox_A = oxidation_energy(formula_A, hal)
    dEox_B = oxidation_energy(formula_B, hal)

    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        Eg = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bowing * x * (1 - x)
        Eh = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        stab = math.exp(-max(Eh, 0) / 0.10)   # Boltzmann weight, Δ=0.1 eV
        dEox = (1 - x) * dEox_A + x * dEox_B
        ox_pen = math.exp(-max(dEox, 0) / K_T_EFF)
        gap = score_band_gap(Eg, lo, hi)
        t  = (rA + rX) / (math.sqrt(2) * (rB + rX))
        mu = rB / rX
        form = math.exp(-0.5*((t-0.90)/0.07)**2) * math.exp(-0.5*((mu-0.50)/0.07)**2)
        env = 1 + alpha*rh/100 + beta*temp/100
        score = form * stab * gap * ox_pen / env

        rows.append({
            "x": round(x, 3),
            "Eg": round(Eg, 3),
            "Ehull": round(Eh, 4),
            "Eox": round(dEox, 3),
            "score": round(score, 3),
            "formula": f"{formula_A}-{formula_B} x={x:.2f}",
        })

    return (pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True))

# ──────────────────────── Ternary screen ─────────────────────

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
    n_mc: int = 200,
) -> pd.DataFrame:

    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    # oxidation energies for each vertex (infer halide)
    halA = next(h for h in ("I", "Br", "Cl") if h in A)
    halB = next(h for h in ("I", "Br", "Cl") if h in B)
    halC = next(h for h in ("I", "Br", "Cl") if h in C)
    oxA, oxB, oxC = (oxidation_energy(f, h) for f, h in ((A, halA), (B, halB), (C, halC)))

    lo, hi = bg
    rows: list[dict] = []
    for x in np.arange(0, 1 + 1e-9, dx):
        for y in np.arange(0, 1 - x + 1e-9, dy):
            z = 1 - x - y
            Eg = (
                z * dA["band_gap"] + x * dB["band_gap"] + y * dC["band_gap"]
                - bows["AB"] * x * z - bows["AC"] * y * z - bows["BC"] * x * y
            )
            Eh = (
                z * dA["energy_above_hull"] + x * dB["energy_above_hull"] + y * dC["energy_above_hull"]
            )
            stab = math.exp(-max(Eh, 0) / 0.10)
            dEox = z*oxA + x*oxB + y*oxC
            ox_pen = math.exp(-max(dEox, 0) / K_T_EFF)
            score = stab * score_band_gap(Eg, lo, hi) * ox_pen

            rows.append({
                "x": round(x, 3), "y": round(y, 3),
                "Eg": round(Eg, 3), "Ehull": round(Eh, 4), "Eox": round(dEox, 3),
                "score": round(score, 3),
                "formula": f"{A}-{B}-{C} x={x:.2f} y={y:.2f}",
            })

    return (pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True))

# alias for legacy callers
_summary = fetch_mp_data
