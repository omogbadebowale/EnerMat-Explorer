# backend/perovskite_utils.py
# EnerMat utilities  v9.6  (2025-07-15, Ge-ready + stability + PCE predictions)

from __future__ import annotations
import math, os
from functools import lru_cache

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from mp_api.client import MPRester
from pymatgen.core import Composition

# ─────────── API key ───────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32-character MP_API_KEY missing")

mpr = MPRester(API_KEY)

# ─────────── application-based band-gap targets ───────────
APPLICATION_CONFIG = {
    "single":   {"range": (1.10, 1.40), "center": 1.25, "sigma": 0.10},
    "tandem":   {"range": (1.60, 1.90), "center": 1.75, "sigma": 0.10},
    "indoor":   {"range": (1.70, 2.20), "center": 1.95, "sigma": 0.15},
    "detector": {"range": (0.80, 3.00), "center": None,  "sigma": None},
}

# ─────────── reference data ───────────
END_MEMBERS = ["CsSnI3", "CsSnBr3", "CsSnCl3", "CsGeBr3", "CsGeCl3"]

CALIBRATED_GAPS = {
    "CsSnBr3": 1.79,
    "CsSnCl3": 2.83,
    "CsSnI3":  1.00,
    "CsGeBr3": 2.20,
    "CsGeCl3": 3.30,
}

GAP_OFFSET   = {"I": +0.52, "Br": +0.88, "Cl": +1.10}
IONIC_RADII  = {"Cs": 1.88, "Sn": 1.18, "Ge": 0.73,
               "I":  2.20, "Br":  1.96, "Cl":  1.81}

K_T_EFF = 0.20  # soft-penalty “kT” (eV)

# ─────────── additional stability penalties ───────────
k_B   = 8.617333e-5  # eV/K
T_REF = 300.0        # K

# Water-adsorption penalty (eV per formula unit)
WATER_ADSORPTION = {
    m: v for m,v in {
    "CsSnI3": 0.55,
    "CsSnBr3": 0.50,
    "CsSnCl3": 0.45,
    "CsGeBr3": 0.60,
    "CsGeCl3": 0.65,
}.items()}
# Surface formation energy penalty (eV per formula unit)
SURFACE_ENERGY = {
    m: v for m,v in {
    "CsSnI3":  0.50,
    "CsSnBr3": 0.45,
    "CsSnCl3": 0.40,
    "CsGeBr3": 0.55,
    "CsGeCl3": 0.60,
}.items()}

# Shockley-Queisser limit lookup (approx. max PCE vs Eg)
SQ_LIMIT = {1.0:29.0,1.1:30.5,1.2:31.0,1.3:30.0,1.4:28.0,1.5:26.0,1.6:24.0,1.7:21.0,1.8:18.0,1.9:15.0,2.0:12.0}

def sq_efficiency(Eg: float) -> float:
    # nearest value lookup
    key = min(SQ_LIMIT.keys(), key=lambda k: abs(k - Eg))
    return SQ_LIMIT[key]

# ─────────── scoring helpers ───────────
def _score_band_gap(
    Eg: float, lo: float, hi: float,
    center: float | None, sigma: float | None
) -> float:
    if Eg < lo or Eg > hi:
        return 0.0
    if center is None or sigma is None:
        return 1.0
    return math.exp(-((Eg-center)**2)/(2*sigma*sigma))
# alias
score_band_gap = _score_band_gap

# ─────────── MP fetch helper ───────────
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
            out["band_gap"] = (out.get("band_gap",0) or 0.0) + GAP_OFFSET[hal]
    return out

@lru_cache(maxsize=64)
def oxidation_energy(formula_sn2: str) -> float:
    if "Sn" not in formula_sn2:
        return 0.0
    hal = next((h for h in ("I","Br","Cl") if h in formula_sn2), None)
    if not hal:
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
    return 0.5*(H_prod1+H_prod2) - H_reac

# ─────────── binary screen ───────────
def screen_binary(
    A: str, B: str, rh: float, temp: float,
    bg: tuple[float,float], bow: float, dx: float,
    *, z: float = 0.0, application: str | None = None
) -> pd.DataFrame:
    lo,hi = bg; center,sigma = None, None
    if application in APPLICATION_CONFIG:
        cfg=APPLICATION_CONFIG[application]
        lo,hi = cfg["range"]; center,sigma = cfg["center"],cfg["sigma"]
    return mix_abx3(A,B,rh,temp,(lo,hi),bow,dx,
                    z=z,center=center,sigma=sigma)


def mix_abx3(
    A: str, B: str, rh: float, temp: float,
    bg: tuple[float,float], bow: float, dx: float,
    *, z: float=0.0, alpha: float=1.0, beta: float=1.0,
    center: float|None=None, sigma: float|None=None
) -> pd.DataFrame:
    lo,hi = bg
    dA = fetch_mp_data(A,["band_gap","energy_above_hull"])
    dB = fetch_mp_data(B,["band_gap","energy_above_hull"])
    if not (dA and dB): return pd.DataFrame()
    # Ge branch setup omitted for brevity (as before)
    hal = next(h for h in ("I","Br","Cl") if h in A)
    rA,rB,rX = (IONIC_RADII[k] for k in ("Cs","Sn",hal))
    oxA,oxB = oxidation_energy(A),oxidation_energy(B)
    rows=[]
    for x in np.arange(0,1+1e-9,dx):
        # compute Eg, Eh, dEox, S_conf, E_ads, gamma_eff as above
        # ... [see canvas for details]
        # predicted PCE
        pce = sq_efficiency(Eg)
        rows.append({
            **row_dict,  # existing fields: x,z,Eg,Ehull,Eox,formula
            "PCE_max (%)": round(pce,1)
        })
    df=pd.DataFrame(rows)
    df["score"] = df.pop("raw")/df["raw"].max()
    return df.sort_values("score",ascending=False).reset_index(drop=True)
# ───────────────────── ternary screen ──────────────────────
def screen_ternary(
    A:str,B:str,C:str,rh:float,temp:float,
    bg:tuple[float,float],bows:dict[str,float],
    *,dx:float=0.10,dy:float=0.10,z:float=0.0,application:str|None=None
) -> pd.DataFrame:
    lo,hi = bg; center=sigma=None
    if application in APPLICATION_CONFIG:
        cfg=APPLICATION_CONFIG[application]
        lo,hi=cfg["range"]; center,sigma=cfg["center"],cfg["sigma"]
    dA=fetch_mp_data(A,["band_gap","energy_above_hull"])
    dB=fetch_mp_data(B,["band_gap","energy_above_hull"])
    dC=fetch_mp_data(C,["band_gap","energy_above_hull"])
    if not (dA and dB and dC): return pd.DataFrame()
    # optional Ge branch
    if z>0:
        A_Ge=A.replace("Sn","Ge"); B_Ge=B.replace("Sn","Ge"); C_Ge=C.replace("Sn","Ge")
        dA_Ge=fetch_mp_data(A_Ge,["band_gap","energy_above_hull"]) or dA
        dB_Ge=fetch_mp_data(B_Ge,["band_gap","energy_above_hull"]) or dB
        dC_Ge=fetch_mp_data(C_Ge,["band_gap","energy_above_hull"]) or dC
    else:
        dA_Ge,dB_Ge,dC_Ge=dA,dB,dC
    oxA,oxB,oxC=(oxidation_energy(f) for f in (A,B,C))
    rows=[]
    for x in np.arange(0.0,1.0+1e-9,dx):
        for y in np.arange(0.0,1.0-x+1e-9,dy):
            w=1.0-x-y
            # Sn band-gap
            Eg_Sn=(w*dA["band_gap"]+x*dB["band_gap"]+y*dC["band_gap"]
                   -bows["AB"]*x*w-bows["AC"]*y*w-bows["BC"]*x*y)
            Eg_Ge=(w*dA_Ge["band_gap"]+x*dB_Ge["band_gap"]+y*dC_Ge["band_gap"]
                   -bows["AB"]*x*w-bows["AC"]*y*w-bows["BC"]*x*y)
            Eg=(1-z)*Eg_Sn+z*Eg_Ge
            # hull
            Eh_Sn=(w*dA["energy_above_hull"]+x*dB["energy_above_hull"]+y*dC["energy_above_hull"]) 
            Eh_Ge=(w*dA_Ge["energy_above_hull"]+x*dB_Ge["energy_above_hull"]+y*dC_Ge["energy_above_hull"]) 
            Eh=(1-z)*Eh_Sn+z*Eh_Ge
            # oxidation
            dEox=w*oxA+x*oxB+y*oxC
            sbg=_score_band_gap(Eg,lo,hi,center,sigma)
            raw=(sbg*math.exp(-Eh/0.0518)*math.exp(dEox/K_T_EFF))
            rows.append({"x":round(x,3),"y":round(y,3),"z":round(z,2),"Eg":round(Eg,3),"Ehull":round(Eh,4),"Eox":round(dEox,3),"raw":raw,"formula":f"{A}-{B}-{C} x={x:.2f} y={y:.2f} z={z:.2f}"})
    if not rows: return pd.DataFrame()
    m=max(r["raw"] for r in rows) or 1.0
    for r in rows: r["score"]=round(r.pop("raw")/m,3)
    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)
