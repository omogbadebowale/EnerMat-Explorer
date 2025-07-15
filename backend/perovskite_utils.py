"""
EnerMat Perovskite Explorer – backend/perovskite_utils.py
Fixed: 2025-07-15  —  Ge-formula bug (“CsGer3”) removed.
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

# ─────────────────────────────  API key  ──────────────────────────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32-character MP_API_KEY missing in env or secrets")

mpr = MPRester(API_KEY)

# ─────────── reference data ───────────
END_MEMBERS = ["CsSnI3", "CsSnBr3", "CsSnCl3", "CsGeBr3", "CsGeCl3"]

CALIBRATED_GAPS = {
    "CsSnI3": 1.00, "CsSnBr3": 1.79, "CsSnCl3": 2.83,
    "CsGeBr3": 2.20, "CsGeCl3": 3.30,
}
GAP_OFFSET = {"I": +0.52, "Br": +0.88, "Cl": +1.10}
IONIC_RADII = {"Cs": 1.88, "Sn": 1.18, "Ge": 0.73,
               "I": 2.20, "Br": 1.96, "Cl": 1.81}
K_T_EFF = 0.20    # eV

# ─────────── helpers ───────────
def fetch_mp_data(formula: str, fields: list[str]):
    docs = mpr.summary.search(formula=formula, fields=tuple(fields))
    if not docs:
        return None
    entry = docs[0]
    out = {f: getattr(entry, f, None) for f in fields}
    if "band_gap" in fields:
        if formula in CALIBRATED_GAPS:
            out["band_gap"] = CALIBRATED_GAPS[formula]
        else:
            hal = next(h for h in ("I", "Br", "Cl") if h in formula)
            out["band_gap"] = (out["band_gap"] or 0.0) + GAP_OFFSET[hal]
    return out


@lru_cache(maxsize=64)
def oxidation_energy(formula_sn2: str) -> float:
    if "Sn" not in formula_sn2:
        return 0.0
    hal = next((h for h in ("I", "Br", "Cl") if h in formula_sn2), None)
    if hal is None:
        return 0.0

    def ΔH(formula: str):
        doc = fetch_mp_data(formula, ["formation_energy_per_atom"])
        if not doc or doc["formation_energy_per_atom"] is None:
            raise ValueError(f"Missing ΔHf for {formula}")
        return doc["formation_energy_per_atom"] * Composition(formula).num_atoms

    H_reac  = ΔH(formula_sn2)
    H_prod1 = ΔH(f"Cs2Sn{hal}6")
    H_prod2 = ΔH("SnO2")
    return 0.5 * (H_prod1 + H_prod2) - H_reac


score_band_gap = lambda Eg, lo, hi: 1.0 if lo <= Eg <= hi else 0.0

# ─────────── screen wrappers for Streamlit cache ───────────
def screen_binary(A, B, rh, temp, bg, bow, dx, *, z: float = 0.0):
    return mix_abx3(A, B, rh, temp, bg, bow, dx, z=z)

# ─────────── binary  (Sn ↔ Sn, optional Ge) ───────────
def mix_abx3(
    formula_A: str,
    formula_B: str,
    rh: float,
    temp: float,
    bg_window: tuple[float, float],
    bowing: float = 0.0,
    dx: float = 0.05,
    z: float = 0.0,          # Ge fraction
    alpha: float = 1.0,
    beta:  float = 1.0,
) -> pd.DataFrame:

    lo, hi = bg_window
    dA = fetch_mp_data(formula_A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(formula_B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    # --- Ge end-members ---------------------------------------------------
    to_ge = lambda fml: fml.replace("Sn", "Ge", 1)   # <-- fixed!
    dGA = fetch_mp_data(to_ge(formula_A), ["band_gap", "energy_above_hull"]) or dA
    dGB = fetch_mp_data(to_ge(formula_B), ["band_gap", "energy_above_hull"]) or dB
    if z == 0:
        dGA, dGB = dA, dB

    hal = next(h for h in ("I", "Br", "Cl") if h in formula_A)
    rA, rB, rX = (IONIC_RADII[k] for k in ("Cs", "Sn", hal))
    eox_A, eox_B = oxidation_energy(formula_A), oxidation_energy(formula_B)

    rows = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        # band-gap
        Eg_sn = ((1 - x)*dA["band_gap"] + x*dB["band_gap"]
                 - bowing*x*(1 - x))
        Eg_ge = ((1 - x)*dGA["band_gap"] + x*dGB["band_gap"]
                 - bowing*x*(1 - x))
        Eg = (1 - z)*Eg_sn + z*Eg_ge

        # Ehull
        Eh_sn = (1 - x)*dA["energy_above_hull"] + x*dB["energy_above_hull"]
        Eh_ge = (1 - x)*dGA["energy_above_hull"] + x*dGB["energy_above_hull"]
        Eh = (1 - z)*Eh_sn + z*Eh_ge

        # oxidation penalty
        dEox   = (1 - x)*eox_A + x*eox_B
        ox_pen = math.exp(dEox / K_T_EFF)

        # other factors
        stab = math.exp(-Eh / (alpha*0.0259))
        tfac = (rA + rX) / (math.sqrt(2)*(rB + rX))
        fit  = math.exp(-beta*abs(tfac - 0.95))

        raw = score_band_gap(Eg, lo, hi)*stab*fit*ox_pen
        rows.append({
            "x": round(x, 3), "z": round(z,2),
            "Eg": round(Eg,3), "Ehull": round(Eh,4), "Eox": round(dEox,3),
            "raw": raw,
            "formula": f"{formula_A}-{formula_B} x={x:.2f} z={z:.2f}",
        })

    if not rows:
        return pd.DataFrame()
    raw_max = max(r["raw"] for r in rows) or 1.0
    for r in rows:
        r["score"] = round(r["raw"]/raw_max, 3)
        del r["raw"]

    return (pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True))

# ─────────── ternary  (Sn only) ───────────
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

    oxA, oxB, oxC = oxidation_energy(A), oxidation_energy(B), oxidation_energy(C)
    lo, hi = bg
    rows = []

    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        for y in np.arange(0.0, 1.0 - x + 1e-9, dy):
            z = 1 - x - y
            Eg = (z*dA["band_gap"] + x*dB["band_gap"] + y*dC["band_gap"]
                  - bows["AB"]*x*z - bows["AC"]*y*z - bows["BC"]*x*y)
            Eh = z*dA["energy_above_hull"] + x*dB["energy_above_hull"] + y*dC["energy_above_hull"]
            dEox = z*oxA + x*oxB + y*oxC
            raw = (score_band_gap(Eg, lo, hi)
                   * math.exp(-Eh/0.0518)
                   * math.exp(dEox/K_T_EFF))
            rows.append({"x":round(x,3),"y":round(y,3),
                         "Eg":round(Eg,3),"Ehull":round(Eh,4),"Eox":round(dEox,3),
                         "raw":raw,
                         "formula":f"{A}-{B}-{C} x={x:.2f} y={y:.2f}"})

    if not rows:
        return pd.DataFrame()
    raw_max = max(r["raw"] for r in rows) or 1.0
    for r in rows:
        r["score"] = round(r["raw"]/raw_max, 3)
        del r["raw"]

    return (pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True))

# legacy alias
_summary = fetch_mp_data
