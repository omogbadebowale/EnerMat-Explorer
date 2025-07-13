"""
EnerMat Perovskite Explorer – backend/perovskite_utils.py
Clean build • 2025‑07‑13 🟢   (patched with Sn‑oxidation fixes)
• calibrated experimental gaps
• strict optical window (step 0/1)
• convex‑hull lattice stability (Ehull)
• Sn²⁺→Sn⁴⁺ oxidation penalty ΔEox (fixed O₂ reference)
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
    "CsPbI3": 1.46,
    "CsPbBr3": 2.32,
}

# empirical gap offsets for raw DFT → experiment alignment (eV)
GAP_OFFSET = {"I": +0.52, "Br": +0.88, "Cl": +1.10}

# simplified ionic radii for Goldschmidt tolerance factor (Å)
IONIC_RADII = {
    "Cs": 1.88, "Sn": 1.18, "Pb": 1.19,
    "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

# effective ‘kT’ (eV) for soft penalty factors used in scoring
K_T_EFF = 0.20  # ≈8 kT at 300 K

# -----------------------------------------------------------------------------
# ↓↓↓  Low‑level helpers  ↓↓↓
# -----------------------------------------------------------------------------

def fetch_mp_data(formula: str, fields: list[str]):
    """Query MP‑API and return a dict of requested fields (may be None)."""
    docs = mpr.summary.search(formula=formula, fields=tuple(fields))
    if not docs:
        return None
    entry = docs[0]
    out = {f: getattr(entry, f, None) for f in fields}
    # apply experimental gap calibration
    if "band_gap" in fields:
        if formula in CALIBRATED_GAPS:
            out["band_gap"] = CALIBRATED_GAPS[formula]
        else:
            hal = next(h for h in ("I", "Br", "Cl") if h in formula)
            out["band_gap"] = (out["band_gap"] or 0.0) + GAP_OFFSET[hal]
    return out

@lru_cache(maxsize=64)
def oxidation_energy(formula_sn2: str) -> float:
    """
    ΔE_ox  (per Sn) for
        CsSnX3  + ½ O2  → ½ (Cs2SnX6 + SnO2)

    • Returns 0.0 when the formula contains no Sn.
    • Positive  value  ⇒ oxidation thermodynamically *unfavourable* (good).
      Negative value  ⇒ Sn2+ wants to oxidise (bad).

    IMPORTANT — normalisation:
      – All MP energies are **energy_per_atom** (eV/atom).  
      – We convert them to “per-formula” energies (eV/formula unit),
        carry out the reaction enthalpy, **then divide by the number of Sn atoms
        in the reactant formula** (usually 1) so the result is ≈ –0.1 … –0.3 eV,
        not –5 eV.
    """
    # ─── skip non-Sn formulas ───────────────────────────────────────────────
    if "Sn" not in formula_sn2:
        return 0.0

    # identify the halide (assumes only one of I / Br / Cl is present)
    try:
        hal = next(h for h in ("I", "Br", "Cl") if h in formula_sn2)
    except StopIteration:
        return 0.0  # exotic / non-halide – ignore

    # helper: energy per FORMULA (eV/formula)  ------------------------------
    def e_formula(formula: str) -> float:
        doc = fetch_mp_data(formula, ["energy_per_atom"])
        if not doc or doc["energy_per_atom"] is None:
            raise ValueError(f"Missing energy for {formula}")
        atoms = Composition(formula).num_atoms
        return doc["energy_per_atom"] * atoms  # convert to eV / formula unit

    # energies (eV / formula) ------------------------------------------------
    E_reac   = e_formula(formula_sn2)                 # CsSnX3
    E_prod1  = e_formula(f"Cs2Sn{hal}6")              # Cs2SnX6
    E_prod2  = e_formula("SnO2")                      # SnO2
    E_O2     = e_formula("O2")                        # O2 (2 atoms)

    # reaction enthalpy (eV / formula) --------------------------------------
    dE_total = 0.5 * (E_prod1 + E_prod2 + E_O2) - E_reac

    # normalise per Sn atom in the reactant formula -------------------------
    n_sn = Composition(formula_sn2).get_el_amt_dict().get("Sn", 1)
    return dE_total / n_sn
# -----------------------------------------------------------------------------
# ↓↓↓  Binary alloy screen  CsSnI₃ ↔ CsSnBr₃ etc.  ↓↓↓
# -----------------------------------------------------------------------------

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

    # choose halide from first end‑member for tolerance‑factor radius
    hal = next(h for h in ("I", "Br", "Cl") if h in formula_A)
    rA, rB, rX = (IONIC_RADII[s] for s in ("Cs", "Sn", hal))
    dEox_A = oxidation_energy(formula_A)
    dEox_B = oxidation_energy(formula_B)

    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        Eg = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bowing * x * (1 - x)
        Eh = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        dEox = (1 - x) * dEox_A + x * dEox_B
        ox_pen = math.exp(dEox / K_T_EFF)  # ≤1 when dEox < 0
        stab = math.exp(-Eh / (alpha * 0.0259))
        tfac = (rA + rX) / (math.sqrt(2) * (rB + rX))
        fit = math.exp(-beta * abs(tfac - 0.95))
        form = score_band_gap(Eg, lo, hi)
        score = form * stab * fit * ox_pen

        rows.append({
            "x": round(x, 3),
            "Eg": round(Eg, 3), "Ehull": round(Eh, 4), "Eox": round(dEox, 3),
            "score": round(score, 3),
            "formula": f"{formula_A}-{formula_B} x={x:.2f}",
        })

    return (pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True))

# -----------------------------------------------------------------------------
# ↓↓↓  Ternary compositional screen  ↓↓↓
# -----------------------------------------------------------------------------

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

    # oxidation energies for each vertex (infer halide internally)
    oxA = oxidation_energy(A)
    oxB = oxidation_energy(B)
    oxC = oxidation_energy(C)

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
            dEox = z*oxA + x*oxB + y*oxC
            ox_pen = math.exp(dEox / K_T_EFF)
            form = score_band_gap(Eg, lo, hi)
            stab = math.exp(-Eh / (0.0259 * 2.0))
            score = form * stab * ox_pen

            rows.append({
                "x": round(x, 3), "y": round(y, 3),
                "Eg": round(Eg, 3), "Ehull": round(Eh, 4), "Eox": round(dEox, 3),
                "score": round(score, 3),
                "formula": f"{A}-{B}-{C} x={x:.2f} y={y:.2f}",
            })

    return (pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True))

# legacy alias for backward compatibility
_summary = fetch_mp_data
