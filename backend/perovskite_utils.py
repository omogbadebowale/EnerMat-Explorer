from __future__ import annotations 
import math
import os
from functools import lru_cache
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from mp_api.client import MPRester
from pymatgen.core import Composition

# ─────────── Shockley–Queisser helper ───────────
# Make sure you have backend/sq.py with `def sq_efficiency(Eg: float) -> float: ...`
from backend.sq import sq_efficiency

# ─────────── API key ───────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32-character MP_API_KEY missing")

mpr = MPRester(API_KEY)

# ─────────── application-based band-gap targets ───────────
APPLICATION_CONFIG = {
    "single": {"range": (1.10, 1.40), "center": 1.25, "sigma": 0.10},
    "tandem": {"range": (1.60, 1.90), "center": 1.75, "sigma": 0.10},
    "indoor": {"range": (1.70, 2.20), "center": 1.95, "sigma": 0.15},
    "detector": {"range": (0.80, 3.00), "center": None,  "sigma": None},
}

# ─────────── reference data ───────────
END_MEMBERS = ["CsSnI3", "CsSnBr3", "CsSnCl3", "CsGeBr3", "CsGeCl3",  "CsPbCl3", "CsPbBr3", "CsPbI3"]

CALIBRATED_GAPS = {
    "CsSnBr3": 1.30,
    "CsSnCl3": 2.40,
    "CsSnI3":  1.00,
    "CsGeBr3": 2.20,
    "CsGeCl3": 2.7,
    "CsPbI3": 1.73,
    "CsPbBr3": 2.30,
    "CsPbCl3": 2.32,
}

GAP_OFFSET = {"I": +0.52, "Br": +0.88, "Cl": +1.10, "Pb": 1.31, }
IONIC_RADII = {"Cs": 1.88, "Sn": 1.18, "Ge": 0.73, "I": 2.20, "Br": 1.96, "Cl": 1.81, "Pb": 1.31, }

K_T_EFF = 0.20  # soft-penalty “kT” (eV)

# ─────────── band-gap scoring ───────────
def _score_band_gap(
    Eg: float,
    lo: float, hi: float,
    center: float | None,
    sigma: float | None
) -> float:
    if Eg < lo or Eg > hi:
        return 0.0
    if center is None or sigma is None:
        return 1.0
    # Gaussian weighting
    return math.exp(-((Eg - center) ** 2) / (2 * sigma * sigma))

score_band_gap = _score_band_gap  # alias

# ─────────── helpers ───────────
def fetch_mp_data(formula: str, fields: list[str]):
    docs = mpr.summary.search(formula=formula, fields=tuple(fields))
    if not docs:
        return None
    ent = docs[0]
    out = {f: getattr(ent, f, None) for f in fields}

    if "band_gap" in fields:
        if formula in CALIBRATED_GAPS:
            out["band_gap"] = CALIBRATED_GAPS[formula]
        else:
            hal = next(h for h in ("I", "Br", "Cl") if h in formula)
            out["band_gap"] = (out.get("band_gap", 0.0) or 0.0) + GAP_OFFSET[hal]
    return out

@lru_cache(maxsize=64)
def oxidation_energy(formula_sn2: str) -> float:
    """ΔEₒₓ per Sn for CsSnX₃ + ½ O₂ → ½ (Cs₂SnX₆ + SnO₂)."""
    if "Sn" not in formula_sn2:
        return 0.0
    hal = next((h for h in ("I", "Br", "Cl") if h in formula_sn2), None)
    if hal is None:
        return 0.0

    def formation_energy_fu(formula: str) -> float:
        doc = fetch_mp_data(formula, ["formation_energy_per_atom"])
        if not doc or doc.get("formation_energy_per_atom") is None:
            raise ValueError(f"Missing formation-energy for {formula}")
        comp = Composition(formula)
        return doc["formation_energy_per_atom"] * comp.num_atoms

    H_reac  = formation_energy_fu(formula_sn2)
    H_prod1 = formation_energy_fu(f"Cs2Sn{hal}6")
    H_prod2 = formation_energy_fu("SnO2")
    return 0.5 * (H_prod1 + H_prod2) - H_reac

# ─────────── doping integration ───────────
def apply_doping(A: str, doping_element: str, z: float) -> str:
    if doping_element == "Ge":
        return A.replace("Sn", "Ge") if "Sn" in A else A
    elif doping_element == "Sb":
        return A.replace("Sn", "Sb") if "Sn" in A else A
    elif doping_element == "Cu":
        return A.replace("Sn", "Cu") if "Sn" in A else A
    elif doping_element == "Mg":
        return A.replace("Sn", "Mg") if "Sn" in A else A
    elif doping_element == "Ca":
        return A.replace("Sn", "Ca") if "Sn" in A else A
    elif doping_element == "Ba":
        return A.replace("Sn", "Ba") if "Sn" in A else A
    elif doping_element == "Ni":
        return A.replace("Sn", "Ni") if "Sn" in A else A
    elif doping_element == "Zn":
        return A.replace("Sn", "Zn") if "Sn" in A else A
    else:
        return A  # no doping applied

# ─────────── binary screen ───────────
def screen_binary(
    A: str,
    B: str,
    rh: float,
    temp: float,
    bg: tuple[float, float],
    bow: float,
    dx: float,
    *,
    z: float = 0.0,
    doping_element: str = "None",
    application: str | None = None,
) -> pd.DataFrame:
    lo, hi = bg
    center = sigma = None
    if application in APPLICATION_CONFIG:
        cfg = APPLICATION_CONFIG[application]
        lo, hi = cfg["range"]
        center, sigma = cfg["center"], cfg["sigma"]

    # Apply doping
    A_doped = apply_doping(A, doping_element, z)
    B_doped = apply_doping(B, doping_element, z)

    return mix_abx3(A_doped, B_doped, rh, temp, (lo, hi), bow, dx,
                    z=z, center=center, sigma=sigma)

def mix_abx3(
    A: str,
    B: str,
    rh: float,
    temp: float,
    bg: tuple[float, float],
    bow: float,
    dx: float,
    *,
    z: float = 0.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    doping_element: str = "None",
    center: float | None = None,
    sigma: float | None = None,
) -> pd.DataFrame:
    lo, hi = bg
    A_doped = apply_doping(A, doping_element, z)
    B_doped = apply_doping(B, doping_element, z)

    dA = fetch_mp_data(A_doped, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B_doped, ["band_gap", "energy_above_hull"])
    
    if not (dA and dB):
        return pd.DataFrame()

    # rest of the calculations...
