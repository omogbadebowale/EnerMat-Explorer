"""
EnerMat Perovskite Explorer – backend/perovskite_utils.py
Version 2025‑07‑12‑Ox  (fully syntactically valid, 4‑space indents)
• calibrated experimental band‑gaps
• strict optical window (step 0/1)
• convex‑hull lattice stability (Ehull)
• oxidation penalty ΔEₒₓ for Sn²⁺ → Sn⁴⁺ (air exposure)
• formula column for binary & ternary screens
"""

from __future__ import annotations
import os, math
from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition

# ────────────────────────────────────────────────────────
# Environment / API key
# ────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32‑character MP_API_KEY missing")

mpr = MPRester(API_KEY)

# ────────────────────────────────────────────────────────
# Constants & lookup tables
# ────────────────────────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

CALIBRATED_GAPS = {
    "CsSnBr3": 1.79,
    "CsSnCl3": 2.83,
    "CsSnI3":  1.30,
    "CsPbBr3": 2.30,
    "CsPbI3":  1.73,
}

GAP_OFFSET = {"I": 0.90, "Br": 0.70, "Cl": 0.80}

IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

# Materials‑Project energies for oxidation products are cached in‑memory
# MP oxygen energy (with 1.36 eV correction) = −9.86 eV per O2 molecule
E_O2 = -9.86

# effective thermal energy used in exponential oxidation penalty
K_T_EFF = 0.20  # eV

# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────

def fetch_mp_data(formula: str, fields: list[str]):
    """Return dict of requested fields with calibrated band‑gap."""
    doc = mpr.summary.search(formula=formula, fields=tuple(fields))
    if not doc:
        return None
    entry = doc[0]
    out = {f: getattr(entry, f, None) for f in fields}
    # gap fix
    if formula in CALIBRATED_GAPS:
        out["band_gap"] = CALIBRATED_GAPS[formula]
    else:
        hal = next(h for h in ("I", "Br", "Cl") if h in formula)
        out["band_gap"] = (out["band_gap"] or 0.0) + GAP_OFFSET[hal]
    return out

@lru_cache(maxsize=64)
def oxidation_energy(formula_sn2: str, hal: str) -> float:
    """ΔE per Sn for  CsSnX3 + ½ O2 → ½ Cs2SnX6 + ½ SnO2 (negative → favorable)."""
    e_reac = fetch_mp_data(formula_sn2, ["energy_per_atom"])["energy_per_atom"]
    e_cs2snx6 = fetch_mp_data(f"Cs2Sn{hal}6", ["energy_per_atom"])["energy_per_atom"]
    e_sno2    = fetch_mp_data("SnO2", ["energy_per_atom"])["energy_per_atom"]
    e_prod = (e_cs2snx6 + e_sno2) / 2.0
    return (e_prod + 0.5 * E_O2) - e_reac

score_band_gap = lambda Eg, lo, hi: 1.0 if lo <= Eg <= hi else 0.0

# ────────────────────────────────────────────────────────
# Binary  A–B  screen
# ────────────────────────────────────────────────────────

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

    X_site = next(h for h in ("I", "Br", "Cl") if h in formula_A)
    rA, rB, rX = (IONIC_RADII[s] for s in ("Cs", "Sn", X_site))
    dEox_A = oxidation_energy(formula_A, X_site)
    dEox_B = oxidation_energy(formula_B, X_site)

    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        Eg = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bowing * x * (1 - x)
        Ehull = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        stab  = max(0.0, 1 - Ehull)
        dEox  = (1 - x) * dEox_A + x * dEox_B
        ox_pen = math.exp(-max(dEox, 0.0) / K_T_EFF)
        gap    = score_band_gap(Eg, lo, hi)
        t  = (rA + rX) / (math.sqrt(2) * (rB + rX))
        mu = rB / rX
        form = math.exp(-0.5 * ((t - 0.90) / 0.07) ** 2) * math.exp(-0.5 * ((mu - 0.50) / 0.07) ** 2)
        env = 1 + alpha * rh / 100 + beta * temp / 100
        score = form * stab * gap * ox_pen / env

        rows.append({
            "x": round(x, 3),
            "Eg": round(Eg, 3),
            "Ehull": round(Ehull, 4),
            "Eox": round(dEox, 3),
            "score": round(score, 3),
            "formula": f"{formula_A}-{formula_B} x={x:.2f}",
        })

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# ────────────────────────────────────────────────────────
# Ternary  A–B–C  screen
# ────────────────────────────────────────────────────────

def screen_ternary(
    A: str, B: str, C: str,
    rh: float, temp: float,
    bg: tuple[float, float],
    bows: dict[str, float],
    dx: float = 0.1, dy: float = 0.1,
    n_mc: int = 200,   # unused – kept for API compatibility
) -> pd.DataFrame:

    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    oxA = oxidation_energy(A, "Br")  # assume A=Bromide, B=Chloride, C=Iodide for demo
    oxB = oxidation_energy(B, "Cl")
    oxC = oxidation_energy(C, "I")

    lo, hi = bg
    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        for y in np.arange(0.0, 1.0 - x + 1e-9, dy):
            z = 1 - x - y
            Eg = (
                z * dA["band_gap"] + x * dB["band_gap"] + y * dC["band_gap"]
                - bows["AB"] * x * z - bows["AC"] * y * z - bows["BC"] * x * y
            )
            Eh = (
                z * dA["energy_above_hull"] + x * dB["energy_above_hull"] + y * dC["energy_above_hull"]
            )
            dEox = z * oxA + x * oxB + y * oxC
            score = math.exp(-max(Eh, 0) / 0.1) * score_band_gap(Eg, lo
