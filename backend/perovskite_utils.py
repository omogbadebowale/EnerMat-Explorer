"""
EnerMat-Explorer – perovskite_utils.py
Clean 2025-07-13
• Calibrated band-gaps (+0.70/0.80/0.90 eV offsets)
• Strict optical filter (0/1)
• Oxidation penalty (ΔEox, eV Sn⁻¹) with correct O₂ reference
• Binary and ternary screening helpers
"""

from __future__ import annotations

import os, math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from mp_api.client import MPRester
from pymatgen.core import Composition

# ── Materials Project key ───────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32-character MP_API_KEY missing.")
mpr = MPRester(API_KEY)

# ── Presets & corrections ───────────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

CALIBRATED_GAPS = {
    "CsSnBr3": 1.79, "CsSnCl3": 2.83, "CsSnI3": 1.30,
    "CsPbBr3": 2.30, "CsPbI3": 1.73,
}
GAP_OFFSET = {"I": 0.90, "Br": 0.70, "Cl": 0.80}

IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

# ── MP helpers ──────────────────────────────────────────────────────
def _mp_summary(formula: str, fields: list[str]) -> dict | None:
    docs = mpr.summary.search(formula=formula, fields=tuple(fields))
    return docs[0].dict() if docs else None

# calibrated band-gap + hull
def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    data = _mp_summary(formula, fields)
    if not data:
        return None
    if formula in CALIBRATED_GAPS:
        data["band_gap"] = CALIBRATED_GAPS[formula]
    else:
        hal = next(h for h in "IBrCl" if h in formula)
        data["band_gap"] += GAP_OFFSET[hal]
    return {f: data.get(f, None) for f in fields}

# ── Energy of one reduced formula unit (eV) ─────────────────────────
def formula_energy(formula: str) -> float:
    """
    Total energy of one *reduced* formula unit (eV).
    energy_per_atom × (# atoms in reduced Composition)
    """
    doc = _mp_summary(formula, ["energy_per_atom"])
    if not doc:
        raise ValueError(f"MP entry for {formula} not found.")
    n_atoms = Composition(formula).num_atoms
    return doc["energy_per_atom"] * n_atoms

# ── Oxidation driving force CsSnX₃ + ½ O₂ → ½ Cs₂SnX₆ + ½ SnO₂ ──────
@st.cache_data(ttl=3600)          # caches each result for 1 h
def oxidation_energy(halide: str) -> float:
    """Return ΔEox (eV Sn⁻¹); negative = oxidation downhill."""
    react = formula_energy(f"CsSn{halide}3")

    prod1 = 0.5 * formula_energy(f"Cs2Sn{halide}6")
    prod2 = 0.5 * formula_energy("SnO2")

    # Materials Project O₂ energy (already corrected); 2 atoms / mol
    e_o2 = formula_energy("O2")
    prod_o2 = 0.5 * e_o2

    return (prod1 + prod2 + prod_o2) - react   # per Sn in reactant

# ── Misc helpers ────────────────────────────────────────────────────
score_band_gap = lambda Eg, lo, hi: 1.0 if lo <= Eg <= hi else 0.0

def _goldschmidt_t_mu(rA, rB, rX):
    t = (rA + rX) / (math.sqrt(2) * (rB + rX))
    mu = rB / rX
    return t, mu

# ── Binary screen ───────────────────────────────────────────────────
def mix_abx3(
    formula_A: str, formula_B: str,
    rh: float, temp: float,
    bg_window: tuple[float, float],
    bowing: float = 0.0, dx: float = 0.05,
    alpha: float = 1.0, beta: float = 1.0,
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

    ΔE_ox = oxidation_energy(X_site)          # constant for this binary

    rows = []
    for x in np.arange(0, 1 + 1e-6, dx):
        Eg     = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bowing * x * (1 - x)
        Eh     = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        gap    = score_band_gap(Eg, lo, hi)
        stab   = math.exp(-max(Eh, 0) / 0.1)

        t, mu  = _goldschmidt_t_mu(rA, rB, rX)
        form   = math.exp(-0.5 * ((t - 0.90) / 0.07) ** 2) * math.exp(-0.5 * ((mu - 0.50) / 0.07) ** 2)

        env    = 1 + alpha * rh / 100 + beta * temp / 100
        ox_pen = math.exp(-max(-ΔE_ox, 0) / 0.2)      # ΔEox < 0 penalises

        score  = form * stab * gap * ox_pen / env

        rows.append({
            "x": round(x, 3),
            "Eg": round(Eg, 3),
            "Ehull": round(Eh, 4),
            "Eox": round(ΔE_ox, 3),
            "score": round(score, 3),
            "formula": f"{formula_A}-{formula_B} x={x:.2f}",
        })

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# ── Ternary screen (A = Br, B = Cl, C = I example) ─────────────────
def screen_ternary(
    A: str, B: str, C: str,
    rh: float, temp: float,
    bg: tuple[float, float],
    bows: dict[str, float],
    dx: float = 0.1, dy: float = 0.1,
) -> pd.DataFrame:

    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    ΔEox_A = oxidation_energy("Br")
    ΔEox_B = oxidation_energy("Cl")
    ΔEox_C = oxidation_energy("I")

    lo, hi = bg
    rows = []
    for x in np.arange(0, 1 + 1e-6, dx):
        for y in np.arange(0, 1 - x + 1e-6, dy):
            z = 1 - x - y
            Eg = (
                z * dA["band_gap"] + x * dB["band_gap"] + y * dC["band_gap"]
                - bows["AB"] * x * z - bows["AC"] * y * z - bows["BC"] * x * y
            )
            Eh = (
                z * dA["energy_above_hull"] + x * dB["energy_above_hull"] + y * dC["energy_above_hull"]
                + bows["AB"] * x * z + bows["AC"] * y * z + bows["BC"] * x * y
            )
            Eox = z * ΔEox_A + x * ΔEox_B + y * ΔEox_C
            stab = math.exp(-max(Eh, 0) / 0.1)
            gap  = score_band_gap(Eg, lo, hi)
            oxp  = math.exp(-max(-Eox, 0) / 0.2)
            score = stab * gap * oxp

            rows.append({
                "x": round(x,3), "y": round(y,3),
                "Eg": round(Eg,3), "Ehull": round(Eh,4),
                "Eox": round(Eox,3), "score": round(score,3),
                "formula": f"CsSn(Br{z:.2f}Cl{x:.2f}I{y:.2f})₃",
            })

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# ── Convenience alias for legacy code ───────────────────────────────
_summary = fetch_mp_data
