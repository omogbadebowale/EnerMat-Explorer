# backend/perovskite_utils.py
"""
EnerMat utilities  v9.6  (2025-07-17, Ge-ready + PCE + Passivation)
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

# Shockley–Queisser helper
from backend.sq import sq_efficiency

# API key
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32-character MP_API_KEY missing")
mpr = MPRester(API_KEY)

# Application-based band-gap targets
APPLICATION_CONFIG = {
    "single":   {"range": (1.10, 1.40), "center": 1.25, "sigma": 0.10},
    "tandem":   {"range": (1.60, 1.90), "center": 1.75, "sigma": 0.10},
    "indoor":   {"range": (1.70, 2.20), "center": 1.95, "sigma": 0.15},
    "detector": {"range": (0.80, 3.00), "center": None,  "sigma": None},
}

# Reference data
END_MEMBERS = ["CsSnI3", "CsSnBr3", "CsSnCl3", "CsGeBr3", "CsGeCl3"]
CALIBRATED_GAPS = {"CsSnBr3":1.79, "CsSnCl3":2.83, "CsSnI3":1.00, "CsGeBr3":2.20, "CsGeCl3":3.30}
GAP_OFFSET    = {"I":+0.52, "Br":+0.88, "Cl":+1.10}
IONIC_RADII   = {"Cs":1.88, "Sn":1.18, "Ge":0.73, "I":2.20, "Br":1.96, "Cl":1.81}
K_T_EFF       = 0.20   # soft‐penalty "kT" (eV)
K_T           = 0.0259 # thermal energy at 300K

# Additive passivation bonuses
ADDITIVE_PENALTIES = {
    "none":   0.0,
    "SnF2":   0.50,
    "NH4SCN": 0.35,
    "PEABr":  0.40,
}

# Band-gap scoring helper
def _score_band_gap(Eg: float, lo: float, hi: float, center: float|None, sigma: float|None) -> float:
    if Eg < lo or Eg > hi:
        return 0.0
    if center is None or sigma is None:
        return 1.0
    return math.exp(-((Eg - center)**2) / (2*sigma*sigma))
score_band_gap = _score_band_gap

# Fetch from Materials Project
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
            hal = next(h for h in ("I","Br","Cl") if h in formula)
            out["band_gap"] = (out.get("band_gap",0.0) or 0.0) + GAP_OFFSET[hal]
    return out

# Sn(II) oxidation penalty
def oxidation_energy(formula_sn2: str) -> float:
    if "Sn" not in formula_sn2:
        return 0.0
    hal = next((h for h in ("I","Br","Cl") if h in formula_sn2), None)
    if hal is None:
        return 0.0
    def formation_energy_fu(frm: str) -> float:
        doc = fetch_mp_data(frm, ["formation_energy_per_atom"])
        if not doc or doc.get("formation_energy_per_atom") is None:
            raise ValueError(f"Missing formation-energy for {frm}")
        comp = Composition(frm)
        return doc["formation_energy_per_atom"] * comp.num_atoms
    H_reac  = formation_energy_fu(formula_sn2)
    H_prod1 = formation_energy_fu(f"Cs2Sn{hal}6")
    H_prod2 = formation_energy_fu("SnO2")
    return 0.5*(H_prod1 + H_prod2) - H_reac

# Binary screening
def screen_binary(
    A: str, B: str, rh: float, temp: float,
    bg: tuple[float,float], bow: float, dx: float,
    *, z: float = 0.0, application: str|None = None, additive: str = "none"
) -> pd.DataFrame:
    lo, hi = bg; center = sigma = None
    if application in APPLICATION_CONFIG:
        cfg = APPLICATION_CONFIG[application]
        lo, hi = cfg["range"]; center, sigma = cfg["center"], cfg["sigma"]
    return mix_abx3(
        A, B, rh, temp, (lo,hi), bow, dx,
        z=z, center=center, sigma=sigma, additive=additive
    )

def mix_abx3(
    A: str, B: str, rh: float, temp: float,
    bg: tuple[float,float], bow: float, dx: float,
    *, z: float = 0.0, alpha: float = 1.0, beta: float = 1.0,
    center: float|None = None, sigma: float|None = None, additive: str = "none"
) -> pd.DataFrame:
    lo, hi = bg
    dA = fetch_mp_data(A, ["band_gap","energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap","energy_above_hull"])
    if not (dA and dB): return pd.DataFrame()
    # Ge branch
    if z > 0:
        A_Ge = A.replace("Sn","Ge"); B_Ge = B.replace("Sn","Ge")
        dA_Ge = fetch_mp_data(A_Ge, ["band_gap","energy_above_hull"]) or dA
        dB_Ge = fetch_mp_data(B_Ge, ["band_gap","energy_above_hull"]) or dB
        oxA_Ge = oxidation_energy(A_Ge); oxB_Ge = oxidation_energy(B_Ge)
    else:
        dA_Ge, dB_Ge = dA, dB
        oxA_Ge, oxB_Ge = oxidation_energy(A), oxidation_energy(B)
    hal = next(h for h in ("I","Br","Cl") if h in A)
    rA, rB, rX = (IONIC_RADII[k] for k in ("Cs","Sn",hal))
    oxA, oxB = oxidation_energy(A), oxidation_energy(B)
    bonus = ADDITIVE_PENALTIES.get(additive, 0.0)
    rows = []
    for x in np.arange(0.0, 1.0+1e-9, dx):
        Eg_Sn   = (1-x)*dA["band_gap"] + x*dB["band_gap"] - bow*x*(1-x)
        Eh_Sn   = (1-x)*dA["energy_above_hull"] + x*dB["energy_above_hull"]
        dEox_Sn = (1-x)*oxA + x*oxB
        Eg_Ge   = (1-x)*dA_Ge["band_gap"] + x*dB_Ge["band_gap"] - bow*x*(1-x)
        Eh_Ge   = (1-x)*dA_Ge["energy_above_hull"] + x*dB_Ge["energy_above_hull"]
        dEox_Ge = (1-x)*oxA_Ge + x*oxB_Ge
        Eg   = (1-z)*Eg_Sn + z*Eg_Ge
        Eh   = (1-z)*Eh_Sn + z*Eh_Ge
        dEox = (1-z)*dEox_Sn + z*dEox_Ge
        sbg  = score_band_gap(Eg, lo, hi, center, sigma)
        raw  = (sbg * math.exp(-Eh/(alpha*K_T)) * math.exp(dEox/K_T_EFF)
                * math.exp(-beta*abs((rA+rX)/(math.sqrt(2)*(rB+rX))-0.95))
                * math.exp(bonus/K_T))
        pce  = sq_efficiency(Eg)
        rows.append({
            "x": round(x,3), "z": round(z,2), "Eg": round(Eg,3),
            "Ehull": round(Eh,4),    "Eox": round(dEox,3),
            "formula": f"{A}-{B} x={x:.2f} z={z:.2f}",
            "PCE_max (%)": round(pce*100,1), "raw": raw
        })
    if not rows: return pd.DataFrame()
    m = max(r["raw"] for r in rows) or 1.0
    for r in rows: r["score"] = round(r.pop("raw")/m, 3)
    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)
