
"""
EnerMat Perovskite Explorer – backend/perovskite_utils.py
Clean build • 2025-07-13 🟢  (oxidation bug fully fixed)

Key features
• experimental band-gap calibration
• convex-hull lattice stability (Ehull)
• physically-correct Sn²⁺→Sn⁴⁺ oxidation penalty ΔEox
  – now computed from **total** (not per-atom) DFT energies
  – reaction:  CsSnX3 + ½ O₂  → ½ (Cs₂SnX₆ + SnO₂)
• binary and ternary alloy screen utilities
"""

from __future__ import annotations
import math, os, functools
from functools import lru_cache
from typing import Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from mp_api.client import MPRester
from pymatgen.core import Composition

# ─── Materials-Project connection ───────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32-character MP_API_KEY missing in env/secrets")

mpr = MPRester(API_KEY)

# ─── Reference data ─────────────────────────────────────────────────────────
END_MEMBERS = [
    "CsPbBr3", "CsSnBr3", "CsSnCl3", "CsSnI3", "CsPbI3"
]

CALIBRATED_GAPS: Dict[str, float] = {  # eV
    "CsSnBr3": 1.79,
    "CsSnCl3": 2.83,
    "CsSnI3":  1.30,
    "CsPbI3":  1.47,
    "CsPbBr3": 2.31,
}

GAP_OFFSET = {"I": +0.52, "Br": +0.88, "Cl": +1.10}  # eV

IONIC_RADII = {  # Å  (Shannon radii, 12-coord A-site / 6-coord B,X)
    "Cs": 1.88, "Rb": 1.72,
    "Sn": 1.18, "Pb": 1.19,
    "I":  2.20, "Br": 1.96, "Cl": 1.81,
}

K_T_EFF = 0.20         # eV – softness of exponential penalty terms
R_GAS   = 8.314462618  # J mol⁻¹ K⁻¹ (for future finite-T modelling)

# ─── Helper: fast MP summary fetch with gap correction ──────────────────────
def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    docs = mpr.summary.search(formula=formula, fields=tuple(fields))
    if not docs:
        return None
    doc  = docs[0]
    out  = {f: getattr(doc, f, None) for f in fields}
    if "band_gap" in fields:
        if formula in CALIBRATED_GAPS:
            out["band_gap"] = CALIBRATED_GAPS[formula]
        else:                                    # heuristic offset
            hal = next(h for h in ("I", "Br", "Cl") if h in formula)
            out["band_gap"] = (out["band_gap"] or 0.0) + GAP_OFFSET[hal]
    return out

# ─── NEW: physically-consistent oxidation enthalpy per Sn ───────────────────
@lru_cache(maxsize=128)
def oxidation_energy(formula_sn2: str) -> float:
    """
    Return ΔEox (eV per Sn) for
        CsSnX3  +  ½ O2   →   ½ (Cs2SnX6 + SnO2)

    • Positive  → oxidation **unfavourable** (good)
    • Negative  → oxidation thermodynamically driven (bad)

    For Pb-based or Sn-free formulas the result is 0.0 by definition.
    """
    if "Sn" not in formula_sn2:
        return 0.0

    try:
        hal = next(h for h in ("I", "Br", "Cl") if h in formula_sn2)
    except StopIteration:
        return 0.0

    # helper – total energy (eV)  NOT energy_per_atom
    def Etot(fml: str) -> float:
        d = fetch_mp_data(fml, ["energy_per_atom"])
        if not d or d["energy_per_atom"] is None:
            raise ValueError(f"No MP energy for {fml}")
        n = Composition(fml).num_atoms
        return d["energy_per_atom"] * n

    reac  = Etot(formula_sn2)                       # CsSnX3
    prod1 = 0.5 * Etot(f"Cs2Sn{hal}6")             # ½ Cs2SnX6
    prod2 = 0.5 * Etot("SnO2")                     # ½ SnO2

    # O2 total energy (PBE, spin-pol.), MP gives per-atom value
    e_o2_atom = fetch_mp_data("O2", ["energy_per_atom"])
    e_o2_atom = e_o2_atom["energy_per_atom"] if e_o2_atom else -4.93
    prod3 = 0.5 * (2.0 * e_o2_atom)               # ½ O2 → (½×2) atoms

    return (prod1 + prod2 + prod3 - reac)  # already per 1 Sn

score_band_gap = lambda Eg, lo, hi: 1.0 if lo <= Eg <= hi else 0.0

# ─── Binary ABX3 alloy screen ───────────────────────────────────────────────
def mix_abx3(
    A: str, B: str,
    rh: float, temp: float,
    bg_window: tuple[float, float],
    bowing: float = 0.0,
    dx: float = 0.05,
    alpha: float = 1.0,   # Ehull softness
    beta:  float = 1.0,   # tolerance-factor weight
) -> pd.DataFrame:

    lo, hi = bg_window
    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    oxA = oxidation_energy(A)
    oxB = oxidation_energy(B)
    hal = next(h for h in ("I", "Br", "Cl") if h in A)
    rA, rB_, rX = IONIC_RADII["Cs"], IONIC_RADII["Sn"], IONIC_RADII[hal]

    rows = []
    for x in np.arange(0, 1 + 1e-9, dx):
        Eg   = (1 - x)*dA["band_gap"] + x*dB["band_gap"] - bowing*x*(1 - x)
        Eh   = (1 - x)*dA["energy_above_hull"] + x*dB["energy_above_hull"]
        dEox = (1 - x)*oxA + x*oxB
        ox_pen = math.exp(dEox / K_T_EFF)          # ≤1 when ΔEox < 0
        stab   = math.exp(-Eh / (alpha*0.10))      # soft hull filter
        tfac   = (rA + rX) / (math.sqrt(2)*(rB_ + rX))
        geom   = math.exp(-beta * abs(tfac - 0.95))
        form   = score_band_gap(Eg, lo, hi)
        score  = form * stab * geom * ox_pen

        rows.append(dict(
            x=round(x,3), Eg=round(Eg,3), Ehull=round(Eh,4),
            Eox=round(dEox,3), score=round(score,3),
            formula=f"{A}-{B} x={x:.2f}"
        ))

    return (pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True))

# ─── Ternary screen (A-B-C) – unchanged logic except new ΔEox ───────────────
def screen_ternary(
    A: str, B: str, C: str,
    rh: float, temp: float,
    bg: tuple[float, float],
    bows: dict[str, float],
    dx: float = 0.10, dy: float = 0.10,
) -> pd.DataFrame:

    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    oxA, oxB, oxC = map(oxidation_energy, (A, B, C))
    lo, hi = bg
    rows   = []
    for x in np.arange(0, 1 + 1e-9, dx):
        for y in np.arange(0, 1 - x + 1e-9, dy):
            z     = 1 - x - y
            Eg    = (
                z*dA["band_gap"] + x*dB["band_gap"] + y*dC["band_gap"]
                - bows["AB"]*x*z - bows["AC"]*y*z - bows["BC"]*x*y
            )
            Eh    = (
                z*dA["energy_above_hull"] + x*dB["energy_above_hull"]
                + y*dC["energy_above_hull"]
            )
            dEox  = z*oxA + x*oxB + y*oxC
            ox_pen = math.exp(dEox / K_T_EFF)
            stab   = math.exp(-Eh / 0.10)
            form   = score_band_gap(Eg, lo, hi)
            score  = form * stab * ox_pen

            rows.append(dict(
                x=round(x,3), y=round(y,3),
                Eg=round(Eg,3), Ehull=round(Eh,4), Eox=round(dEox,3),
                score=round(score,3),
                formula=f"{A}-{B}-{C} x={x:.2f} y={y:.2f}"
            ))

    return (pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True))

# shorthand alias – Streamlit front-end expects it
_summary = fetch_mp_data
