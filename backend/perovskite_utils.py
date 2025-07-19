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
IONIC_RADII = {"Cs": 1.88, "Sn": 1.18, "Ge": 0.73,
               "I": 2.20, "Br": 1.96, "Cl": 1.81, "Pb": 1.31, }

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
    doping_element: str = "Ge",  # Add doping element as argument
    application: str | None = None,
) -> pd.DataFrame:
    lo, hi = bg
    center = sigma = None
    if application in APPLICATION_CONFIG:
        cfg = APPLICATION_CONFIG[application]
        lo, hi = cfg["range"]
        center, sigma = cfg["center"], cfg["sigma"]

    return mix_abx3(A, B, rh, temp, (lo, hi), bow, dx,
                    z=z, doping_element=doping_element, center=center, sigma=sigma)

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
    doping_element: str = "Ge",  # Add doping element as argument
    alpha: float = 1.0,
    beta: float = 1.0,
    center: float | None = None,
    sigma: float | None = None,
) -> pd.DataFrame:
    lo, hi = bg
    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    # Handle doping element (adjust composition based on doping element)
    if doping_element == "Ge":
        A_Doping = A.replace("Sn", "Ge")
        B_Doping = B.replace("Sn", "Ge")
    else:
        A_Doping = A
        B_Doping = B

    dA_Doping = fetch_mp_data(A_Doping, ["band_gap", "energy_above_hull"]) or dA
    dB_Doping = fetch_mp_data(B_Doping, ["band_gap", "energy_above_hull"]) or dB

    oxA_Doping = oxidation_energy(A_Doping)
    oxB_Doping = oxidation_energy(B_Doping)

    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        # Sn branch
        Eg_Sn   = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bow * x * (1 - x)
        Eh_Sn   = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        dEox_Sn = (1 - x) * oxA + x * oxB

        # Doping element branch (like Ge, Sb, etc.)
        Eg_Doping = (1 - x) * dA_Doping["band_gap"] + x * dB_Doping["band_gap"] - bow * x * (1 - x)
        Eh_Doping = (1 - x) * dA_Doping["energy_above_hull"] + x * dB_Doping["energy_above_hull"]
        dEox_Doping = (1 - x) * oxA_Doping + x * oxB_Doping

        # Interpolate based on doping element
        Eg = (1.0 - z) * Eg_Sn + z * Eg_Doping
        Eh = (1.0 - z) * Eh_Sn + z * Eh_Doping
        dEox = (1.0 - z) * dEox_Sn + z * dEox_Doping

        sbg = _score_band_gap(Eg, lo, hi, center, sigma)
        raw = (
            sbg
            * math.exp(-Eh / (alpha * 0.0259))
            * math.exp(dEox / K_T_EFF)
            * math.exp(-beta * abs((rA + rX) / (math.sqrt(2) * (rB + rX)) - 0.95))
        )

        # Shockley–Queisser PCE limit
        pce = sq_efficiency(Eg)

        rows.append({
            "x":           round(x, 3),
            "z":           round(z, 2),
            "Eg":          round(Eg, 3),
            "Ehull":       round(Eh, 4),
            "Eox":         round(dEox, 3),
            "raw":         raw,
            "formula":     f"{A}-{B} x={x:.2f} z={z:.2f}",
            "PCE_max (%)": round(pce * 100, 1),
        })

    if not rows:
        return pd.DataFrame()
    m = max(r["raw"] for r in rows) or 1.0
    for r in rows:
        r["score"] = round(r.pop("raw") / m, 3)

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
