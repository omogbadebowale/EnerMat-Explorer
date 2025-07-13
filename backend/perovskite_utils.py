"""
EnerMat Perovskite Explorer – physics backend
version 2025-07-13
✓ calibrated band-gaps
✓ strict optical gate
✓ oxidation-energy penalty
✓ halogen-safe scanning (no StopIteration)
"""

from __future__ import annotations

import os, math, numpy as np, pandas as pd, streamlit as st
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition

# ─── Materials Project key ──────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32-character MP_API_KEY missing")
mpr = MPRester(API_KEY)

# ─── Presets & reference data ───────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

# experimentally calibrated end-member gaps
CALIBRATED_GAPS = {
    "CsSnBr3": 1.79, "CsSnCl3": 2.83, "CsSnI3": 1.30,
    "CsPbBr3": 2.30, "CsPbI3":  1.73,
}
# PBE→experiment halo offsets (eV)
GAP_OFFSET = {"I": 0.90, "Br": 0.70, "Cl": 0.80}

IONIC_RADII = {"Cs":1.88,"Rb":1.72,"MA":2.17,"FA":2.53,
               "Pb":1.19,"Sn":1.18,"I":2.20,"Br":1.96,"Cl":1.81}

# ─── utility: never crash if formula lacks a halogen char ───────────
def _find_halide(formula: str) -> str:
    """Return 'I', 'Br' or 'Cl' if present, else default 'Br'."""
    for h in ("I", "Br", "Cl"):
        if h in formula:
            return h
    return "Br"

# ─── Materials-Project summary with calibrated band-gap  ────────────
def fetch_mp_data(formula: str, fields: list[str]) -> dict|None:
    docs = mpr.summary.search(formula=formula, fields=tuple(fields))
    if not docs:
        return None
    entry = docs[0]
    d = {f: getattr(entry, f, None) for f in fields}

    if formula in CALIBRATED_GAPS:
        d["band_gap"] = CALIBRATED_GAPS[formula]
    else:
        hal = _find_halide(formula)
        d["band_gap"] = (d["band_gap"] or 0) + GAP_OFFSET[hal]
    return d

score_band_gap = lambda Eg, lo, hi: 1.0 if lo <= Eg <= hi else 0.0

# ─── oxidation driving force  CsSnX3 + ½O2 → ½Cs2SnX6 + ½SnO2 ───────
def oxidation_energy(formula: str) -> float:
    """
    ΔE per Sn (eV).  Negative = oxidation downhill (unstable).
    Uses MP formation energies – fast, no DFT run.
    """
    hal = _find_halide(formula)
    react = fetch_mp_data(f"CsSn{hal}3", ["energy_per_atom"])["energy_per_atom"]
    prod_1 = fetch_mp_data(f"Cs2Sn{hal}6", ["energy_per_atom"])["energy_per_atom"]
    prod_2 = fetch_mp_data("SnO2", ["energy_per_atom"])["energy_per_atom"]
    e_o2   = fetch_mp_data("O2",   ["energy_per_atom"])["energy_per_atom"] * 2
    return 0.5 * prod_1 + 0.5 * prod_2 - react + 0.5 * e_o2

# ─── Binary screening A–B ───────────────────────────────────────────
def mix_abx3(formula_A:str, formula_B:str,
             rh:float, temp:float, bg_window:tuple[float,float],
             bowing:float=0.0, dx:float=0.05,
             alpha:float=1.0, beta:float=1.0) -> pd.DataFrame:

    lo, hi = bg_window
    dA = fetch_mp_data(formula_A, ["band_gap","energy_above_hull"])
    dB = fetch_mp_data(formula_B, ["band_gap","energy_above_hull"])
    if not (dA and dB): return pd.DataFrame()

    comp = Composition(formula_A)
    A_site = next(e.symbol for e in comp.elements if e.symbol in IONIC_RADII)
    B_site = next(e.symbol for e in comp.elements if e.symbol in {"Pb","Sn"})
    X_site = _find_halide(formula_A)
    rA,rB,rX = IONIC_RADII[A_site],IONIC_RADII[B_site],IONIC_RADII[X_site]

    # oxidation reference (take A’s halide for the series)
    e_ox = oxidation_energy(formula_A)

    rows=[]
    for x in np.arange(0,1+1e-6,dx):
        Eg = (1-x)*dA["band_gap"] + x*dB["band_gap"] - bowing*x*(1-x)
        Eh = (1-x)*dA["energy_above_hull"] + x*dB["energy_above_hull"]
        stab   = max(0.0, 1-Eh)
        gap    = score_band_gap(Eg, lo, hi)
        t  = (rA+rX)/(math.sqrt(2)*(rB+rX))
        mu = rB/rX
        form  = math.exp(-0.5*((t-0.90)/0.07)**2)*math.exp(-0.5*((mu-0.50)/0.07)**2)
        env   = 1 + alpha*rh/100 + beta*temp/100
        ox_pen= math.exp(-max(e_ox,0)/0.2)         # tunable
        score = form * stab * gap * ox_pen / env
        rows.append(dict(
            x=round(x,3), Eg=round(Eg,3),
            Ehull=round(Eh,4), Eox=round(e_ox,3),
            score=round(score,3),
            formula=f"{formula_A}-{formula_B} x={x:.2f}"
        ))
    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)

# ─── Ternary screening (unchanged except for _find_halide) ──────────
def screen_ternary(A:str,B:str,C:str, rh:float,temp:float,
                   bg:tuple[float,float], bows:dict[str,float],
                   dx:float=0.1, dy:float=0.1, n_mc:int=200) -> pd.DataFrame:
    dA = fetch_mp_data(A,["band_gap","energy_above_hull"])
    dB = fetch_mp_data(B,["band_gap","energy_above_hull"])
    dC = fetch_mp_data(C,["band_gap","energy_above_hull"])
    if not(dA and dB and dC): return pd.DataFrame()

    lo,hi = bg
    e_ox  = oxidation_energy(A)  # use A’s halide as proxy

    rows=[]
    for x in np.arange(0,1+1e-6,dx):
        for y in np.arange(0,1-x+1e-6,dy):
            z = 1-x-y
            Eg = ( z*dA["band_gap"] + x*dB["band_gap"] + y*dC["band_gap"]
                   - bows["AB"]*x*z - bows["AC"]*y*z - bows["BC"]*x*y )
            Eh = ( z*dA["energy_above_hull"] + x*dB["energy_above_hull"] + y*dC["energy_above_hull"]
                   + bows["AB"]*x*z + bows["AC"]*y*z + bows["BC"]*x*y )
            score = math.exp(-max(Eh,0)/0.1) * score_band_gap(Eg,lo,hi) \
                    * math.exp(-max(e_ox,0)/0.2)
            rows.append(dict(
                x=round(x,3), y=round(y,3),
                Eg=round(Eg,3), Ehull=round(Eh,4), Eox=round(e_ox,3),
                score=round(score,3),
                formula=f"CsSn(Br{z:.2f}Cl{y:.2f}I{x:.2f})₃"
            ))
    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)

# backward-compat alias for app.py
_summary = fetch_mp_data
