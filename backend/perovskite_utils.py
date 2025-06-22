import os
from dotenv import load_dotenv
load_dotenv()

# for secrets fallback on Streamlit Cloud
import streamlit as st

import re
import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Composition

# ── Load Materials Project API key ─────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError(
        "🛑 Please set MP_API_KEY to your 32-character Materials Project API key"
    )
mpr = MPRester(API_KEY)

# ── Supported end-members ──────────────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

# ── Ionic radii (Å) for Goldschmidt tolerance ──────────────────────────
IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

# ── Subscript and organic formula normalization ─────────────────────────
SUBSCRIPT_MAP = {
    "₀":"0","₁":"1","₂":"2","₃":"3","₄":"4",
    "₅":"5","₆":"6","₇":"7","₈":"8","₉":"9"
}
ORGANIC_MAP = {"MA": "CH3NH3", "FA": "CH(NH2)2"}

def normalize_formula(formula: str) -> str:
    """
    Convert Unicode subscripts to ASCII and expand MA/FA shorthand to full chemical.
    """
    f = formula.strip()
    for uni, digit in SUBSCRIPT_MAP.items():
        f = f.replace(uni, digit)
    for short, full in ORGANIC_MAP.items():
        if f.startswith(short):
            f = f.replace(short, full, 1)
    return f


def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    """Return a dict of the first matching entry's requested fields, or None."""
    norm = normalize_formula(formula)
    try:
        comp = Composition(norm)
    except Exception:
        raise ValueError(f"Invalid formula: {formula} (normalized to {norm})")
    search_formula = comp.reduced_formula
    docs = mpr.summary.search(formula=search_formula)
    if not docs:
        return None
    entry = docs[0]
    out: dict = {}
    for f in fields:
        if hasattr(entry, f):
            out[f] = getattr(entry, f)
    return out if out else None


def score_band_gap(bg: float, lo: float, hi: float) -> float:
    """How close bg is to the [lo, hi] window."""
    if bg < lo:
        return max(0.0, 1 - (lo - bg) / (hi - lo))
    if bg > hi:
        return max(0.0, 1 - (bg - hi) / (hi - lo))
    return 1.0


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
    """Binary screening A–B across x from 0→1."""
    formula_A = normalize_formula(formula_A)
    formula_B = normalize_formula(formula_B)
    dA = fetch_mp_data(formula_A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(formula_B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        raise ValueError(f"No MP data for '{formula_A}' or '{formula_B}'")
    lo, hi = bg_window
    comp = Composition(formula_A)
    A_site = next(e.symbol for e in comp.elements if e.symbol in IONIC_RADII)
    B_site = next(e.symbol for e in comp.elements if e.symbol in {"Pb","Sn"})
    X_site = next(e.symbol for e in comp.elements if e.symbol in {"I","Br","Cl"})
    rA, rB, rX = IONIC_RADII[A_site], IONIC_RADII[B_site], IONIC_RADII[X_site]
    rows = []
    for x in np.arange(0,1+1e-6,dx):
        Eg = (1-x)*dA["band_gap"] + x*dB["band_gap"] - bowing*x*(1-x)
        hull = (1-x)*dA["energy_above_hull"] + x*dB["energy_above_hull"]
        stability = max(0.0,1-hull)
        gap_score = score_band_gap(Eg, lo, hi)
        t=(rA+rX)/(np.sqrt(2)*(rB+rX)); mu=rB/rX
        form_score=np.exp(-0.5*((t-0.90)/0.07)**2)*np.exp(-0.5*((mu-0.50)/0.07)**2)
        env_pen=1+alpha*(rh/100)+beta*(temp/100)
        score=form_score*stability*gap_score/env_pen
        rows.append({"x":round(x,3),"Eg":round(Eg,3),"stability":round(stability,3),"score":round(score,3),"formula":f"{formula_A}-{formula_B} x={x:.2f}"})
    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)


def screen_ternary(
    A:str,B:str,C:str,rh:float,temp:float,bg:tuple[float,float],bows:dict[str,float],dx:float=0.1,dy:float=0.1,n_mc:int=200
)->pd.DataFrame:
    A=normalize_formula(A);B=normalize_formula(B);C=normalize_formula(C)
    dA=fetch_mp_data(A,["band_gap","energy_above_hull"])
    dB=fetch_mp_data(B,["band_gap","energy_above_hull"])
    dC=fetch_mp_data(C,["band_gap","energy_above_hull"])
    if not(dA and dB and dC): raise ValueError(f"Missing data for {A},{B},{C}")
    lo,hi=bg;rows=[]
    for x in np.arange(0,1+1e-6,dx):
        for y in np.arange(0,1-x+1e-6,dy): z=1-x-y;Eg=(z*dA["band_gap"]+x*dB["band_gap"]+y*dC["band_gap"]-bows["AB"]*x*z-bows["AC"]*y*z-bows["BC"]*x*y);Eh=(z*dA["energy_above_hull"]+x*dB["energy_above_hull"]+y*dC["energy_above_hull"]+bows["AB"]*x*z+bows["AC"]*y*z+bows["BC"]*x*y);stability=np.exp(-max(Eh,0)/0.1);rows.append({"x":round(x,3),"y":round(y,3),"Eg":round(Eg,3),"score":round(stability*score_band_gap(Eg,lo,hi),3)})
    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)

# Sanity check when run as script
if __name__=="__main__":
    print("Running sanity checks...")
    # Lead-based
    try:
        df_lead = mix_abx3("CsPbBr3","CsPbI3",50,25,(0.5,3.0),0.30,0.01)
        print("Lead-based:", df_lead[df_lead.x.isin([0.0,0.75,1.0])])
    except Exception as e:
        print("Lead-based failed:", e)
    # Lead-free
    try:
        df_sn = mix_abx3("CsSnBr3","CsSnCl3",50,25,(0.5,3.0),0.35,0.01)
        print("Lead-free:", df_sn[df_sn.x.isin([0.0,0.36,1.0])])
    except Exception as e:
        print("Lead-free failed:", e)

# app.py (unchanged) ...

