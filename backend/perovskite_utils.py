"""
EnerMat Perovskite Explorer – backend/perovskite_utils.py
CLEAN 2025-07-13
• calibrated band gaps
• strict optical filter
• robust oxidation energy (ΔEox)  ← NEW
"""

from __future__ import annotations
import os, math, numpy as np, pandas as pd, streamlit as st
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition

# ── API KEY ───────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32-character MP_API_KEY missing")
mpr = MPRester(API_KEY)

# ── PRESETS & CORRECTIONS ─────────────────────────────────────────
END_MEMBERS   = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]
CALIBRATED_GAPS = {"CsSnBr3": 1.79, "CsSnCl3": 2.83, "CsSnI3": 1.30,
                   "CsPbBr3": 2.30, "CsPbI3": 1.73}
GAP_OFFSET    = {"I": 0.90, "Br": 0.70, "Cl": 0.80}
IONIC_RADII   = {"Cs":1.88,"Rb":1.72,"MA":2.17,"FA":2.53,
                 "Pb":1.19,"Sn":1.18,"I":2.20,"Br":1.96,"Cl":1.81}

# ── LOW-LEVEL HELPERS ─────────────────────────────────────────────
def _mp_summary(formula: str, fields: list[str]) -> dict | None:
    docs = mpr.summary.search(formula=formula, fields=tuple(set(fields)|{"nsites"}))
    return docs[0].model_dump() if docs else None

def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    doc = _mp_summary(formula, fields + ["band_gap"])
    if not doc: return None
    if formula in CALIBRATED_GAPS:
        doc["band_gap"] = CALIBRATED_GAPS[formula]
    else:
        hal = next(h for h in "IBrCl" if h in formula)
        doc["band_gap"] = (doc["band_gap"] or 0) + GAP_OFFSET[hal]
    return {k: doc.get(k) for k in fields}

def formula_energy(formula:str) -> tuple[float,int]:
    """(E_formula_unit , n_sites)"""
    doc = _mp_summary(formula, ["energy_per_atom","nsites"])
    return doc["energy_per_atom"]*doc["nsites"], doc["nsites"]

def oxidation_energy(formula:str, hal:str) -> float:
    """
    ΔE (eV Sn⁻¹) for:  CsSnX3 + ½ O2  →  ½ Cs2SnX6 + ½ SnO2
    sign convention:  negative = oxidation downhill (bad)
    """
    Er, _ = formula_energy(f"CsSn{hal}3")
    Ep1 , _= formula_energy(f"Cs2Sn{hal}6")
    Ep2 , _= formula_energy("SnO2")
    EO2 ,_ = formula_energy("O2")          # per molecule
    dE = 0.5*Ep1 + 0.5*Ep2 + 0.5*EO2 - Er  # eV per formula-unit
    return dE                              # one Sn per formula-unit

score_band_gap = lambda Eg,lo,hi: 1.0 if lo <= Eg <= hi else 0.0

# ── BINARY AB SCREEN ──────────────────────────────────────────────
def mix_abx3(formula_A:str, formula_B:str,
             rh:float, temp:float,
             bg_window:tuple[float,float],
             bowing:float=0.0, dx:float=0.05,
             alpha:float=1.0, beta:float=1.0)->pd.DataFrame:

    lo,hi = bg_window
    dA = fetch_mp_data(formula_A,["band_gap","energy_above_hull"])
    dB = fetch_mp_data(formula_B,["band_gap","energy_above_hull"])
    if not(dA and dB): return pd.DataFrame()

    comp = Composition(formula_A)
    A_site = next(e.symbol for e in comp.elements if e.symbol in IONIC_RADII)
    B_site = next(e.symbol for e in comp.elements if e.symbol in {"Pb","Sn"})
    X_site = next(e.symbol for e in comp.elements if e.symbol in {"I","Br","Cl"})
    rA,rB,rX = IONIC_RADII[A_site],IONIC_RADII[B_site],IONIC_RADII[X_site]

    rows=[]
    for x in np.arange(0,1+1e-6,dx):
        Eg  = (1-x)*dA["band_gap"] + x*dB["band_gap"] - bowing*x*(1-x)
        Eh  = (1-x)*dA["energy_above_hull"] + x*dB["energy_above_hull"]
        gap = score_band_gap(Eg,lo,hi)
        t   = (rA+rX)/(math.sqrt(2)*(rB+rX))
        mu  = rB/rX
        form= math.exp(-0.5*((t-0.90)/0.07)**2)*math.exp(-0.5*((mu-0.50)/0.07)**2)
        env = 1 + alpha*rh/100 + beta*temp/100

        # --- oxidation penalty --------------------------------------
        Eox = oxidation_energy(formula_A if x<0.5 else formula_B, X_site)
        oxi_penalty = math.exp(+Eox/0.25)   # mild penalty when Eox << 0

        score = form * max(0.0,1-Eh) * gap * oxi_penalty / env

        rows.append({"x":round(x,3),"Eg":round(Eg,3),
                     "Ehull":round(Eh,4),"Eox":round(Eox,3),
                     "score":round(score,3),
                     "formula":f"{formula_A}-{formula_B} x={x:.2f}"})

    return (pd.DataFrame(rows)
            .sort_values("score",ascending=False)
            .reset_index(drop=True))

# ── TERNARY ABC SCREEN  (unchanged except for oxidation + column) ─
def screen_ternary(A:str,B:str,C:str,
                   rh:float,temp:float,
                   bg:tuple[float,float],
                   bows:dict[str,float],
                   dx:float=0.1,dy:float=0.1,
                   n_mc:int=200)->pd.DataFrame:

    dA = fetch_mp_data(A,["band_gap","energy_above_hull"])
    dB = fetch_mp_data(B,["band_gap","energy_above_hull"])
    dC = fetch_mp_data(C,["band_gap","energy_above_hull"])
    if not(dA and dB and dC): return pd.DataFrame()

    lo,hi = bg ; rows=[]
    for x in np.arange(0,1+1e-6,dx):
        for y in np.arange(0,1-x+1e-6,dy):
            z = 1-x-y
            Eg = (z*dA["band_gap"]+x*dB["band_gap"]+y*dC["band_gap"]
                  - bows["AB"]*x*z - bows["AC"]*y*z - bows["BC"]*x*y)
            Eh = (z*dA["energy_above_hull"]+x*dB["energy_above_hull"]+y*dC["energy_above_hull"]
                  + bows["AB"]*x*z + bows["AC"]*y*z + bows["BC"]*x*y)
            gap = score_band_gap(Eg,lo,hi)
            Eox = (x*oxidation_energy(B,"Br")+y*oxidation_energy(C,"Cl")+z*oxidation_energy(A,"Br"))
            score = math.exp(-max(Eh,0)/0.1)*gap*math.exp(+Eox/0.25)

            rows.append({"x":round(x,3),"y":round(y,3),
                         "Eg":round(Eg,3),"Ehull":round(Eh,4),
                         "Eox":round(Eox,3),"score":round(score,3),
                         "formula":f"{A}-{B}-{C} x={x:.2f} y={y:.2f}"})

    return (pd.DataFrame(rows)
            .sort_values("score",ascending=False)
            .reset_index(drop=True))

# ── Legacy alias for the frontend ──────────────────────────────────
_summary = fetch_mp_data
