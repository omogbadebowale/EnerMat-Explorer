"""
EnerMat backend – v9.6 (2025-07-13)
• calibrated band-gaps
• convex-hull stability
• optical window (strict)
• Sn-oxidation term  (CsSnX3 + ½ O2 → ½ Cs2SnX6 + ½ SnO2)
"""

from __future__ import annotations
import os, math, numpy as np, pandas as pd, streamlit as st
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition

# ── API key ─────────────────────────────────────────────────────────
load_dotenv()
API = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY", "")
if len(API) != 32:
    raise RuntimeError("🛑 MP_API_KEY missing or wrong length (32 chars)")
mpr = MPRester(API)

# ── presets / constants ────────────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsSnI3"]

CAL_GAP   = {"CsSnBr3":1.79,"CsSnCl3":2.83,"CsSnI3":1.30,
             "CsPbBr3":2.30,"CsPbI3":1.73}
GAP_SHIFT = {"I":0.90,"Br":0.70,"Cl":0.80}
IONIC_R   = {"Cs":1.88,"Rb":1.72,"MA":2.17,"FA":2.53,
             "Pb":1.19,"Sn":1.18,"I":2.20,"Br":1.96,"Cl":1.81}

# ── helper functions ───────────────────────────────────────────────
def _find_halide(formula:str) -> str:
    return next(h for h in ("I","Br","Cl") if h in formula)

def fetch_mp_data(formula:str, fields:list[str]) -> dict|None:
    """Return dict(field→value) with calibrated gap inserted."""
    docs = mpr.summary.search(formula=formula, fields=tuple(fields))
    if not docs:
        return None
    ent = docs[0]
    d   = {f:getattr(ent,f,None) for f in fields}

    if "band_gap" in fields:
        if formula in CAL_GAP:
            d["band_gap"] = CAL_GAP[formula]
        else:
            hal = _find_halide(formula)
            d["band_gap"] = (d["band_gap"] or 0) + GAP_SHIFT[hal]
    return d

def _E_formula(formula:str) -> float:
    d = fetch_mp_data(formula,["energy_per_atom"])
    if not d: raise RuntimeError(f"no MP entry for {formula}")
    return d["energy_per_atom"] * Composition(formula).num_atoms

def oxidation_energy(formula:str) -> float:
    """
    ΔE per Sn  for  CsSnX3 + ½O2 → ½(Cs2SnX6 + SnO2)   (eV Sn⁻¹)
    """
    hal   = _find_halide(formula)
    E_re  = _E_formula(f"CsSn{hal}3")
    E_p1  = _E_formula(f"Cs2Sn{hal}6")
    E_p2  = _E_formula("SnO2")
    E_O2  = _E_formula("O2")
    return 0.5*E_p1 + 0.5*E_p2 - E_re + 0.5*E_O2

score_gap = lambda Eg,lo,hi: 1.0 if lo<=Eg<=hi else 0.0

# ── binary screen ───────────────────────────────────────────────────
def mix_abx3(formA:str,formB:str,
             rh:float,temp:float,
             bg:(float,float),
             bow:float=0.0,dx:float=0.05,
             alpha:float=1.0,beta:float=1.0)->pd.DataFrame:

    dA = fetch_mp_data(formA,["band_gap","energy_above_hull"])
    dB = fetch_mp_data(formB,["band_gap","energy_above_hull"])
    if not(dA and dB): return pd.DataFrame()

    comp = Composition(formA)
    A_site = next(e.symbol for e in comp.elements if e.symbol in IONIC_R)
    rA,rB,rX = IONIC_R[A_site],IONIC_R["Sn" if "Sn" in formA else "Pb"],IONIC_R[_find_halide(formA)]

    eA,eB   = oxidation_energy(formA),oxidation_energy(formB)
    lo,hi   = bg; rows=[]
    for x in np.arange(0,1+1e-6,dx):
        Eg   = (1-x)*dA["band_gap"]+x*dB["band_gap"]-bow*x*(1-x)
        Eh   = (1-x)*dA["energy_above_hull"]+x*dB["energy_above_hull"]
        Eox  = (1-x)*eA + x*eB
        t    = (rA+rX)/ (math.sqrt(2)*(rB+rX))
        mu   = rB/rX
        form = math.exp(-0.5*((t-0.90)/0.07)**2)*math.exp(-0.5*((mu-0.50)/0.07)**2)
        env  = 1+alpha*rh/100+beta*temp/100
        score= form*max(0,1-Eh)*score_gap(Eg,lo,hi)*math.exp(Eox/0.2)/env
        rows.append(dict(x=round(x,3),Eg=round(Eg,3),
                         Ehull=round(Eh,4),Eox=round(Eox,3),
                         score=round(score,3),
                         formula=f"{formA}-{formB} x={x:.2f}"))
    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)

# ── ternary screen ──────────────────────────────────────────────────
def screen_ternary(A:str,B:str,C:str,
                   rh:float,temp:float,
                   bg:(float,float),
                   bows:dict[str,float],
                   dx:float=0.2,dy:float=0.2)->pd.DataFrame:

    dA=fetch_mp_data(A,["band_gap","energy_above_hull"])
    dB=fetch_mp_data(B,["band_gap","energy_above_hull"])
    dC=fetch_mp_data(C,["band_gap","energy_above_hull"])
    if not(dA and dB and dC): return pd.DataFrame()

    eA,eB,eC = oxidation_energy(A),oxidation_energy(B),oxidation_energy(C)
    lo,hi    = bg; rows=[]
    for x in np.arange(0,1+1e-6,dx):
        for y in np.arange(0,1-x+1e-6,dy):
            z  = 1-x-y
            Eg = (z*dA["band_gap"]+x*dB["band_gap"]+y*dC["band_gap"]
                  - bows["AB"]*x*z - bows["AC"]*y*z - bows["BC"]*x*y)
            Eh = (z*dA["energy_above_hull"]+x*dB["energy_above_hull"]+y*dC["energy_above_hull"]
                  + bows["AB"]*x*z + bows["AC"]*y*z + bows["BC"]*x*y)
            Eox= z*eA + x*eB + y*eC
            score= math.exp(-max(Eh,0)/0.1)*score_gap(Eg,lo,hi)*math.exp(Eox/0.2)
            rows.append(dict(x=round(x,2),y=round(y,2),
                             Eg=round(Eg,3),Ehull=round(Eh,4),
                             Eox=round(Eox,3),score=round(score,3),
                             formula=f"{A}-{B}-{C} x={x:.2f} y={y:.2f}"))
    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)

# ── public symbols (import *) ───────────────────────────────────────
__all__ = ["mix_abx3","screen_ternary","END_MEMBERS","fetch_mp_data"]
