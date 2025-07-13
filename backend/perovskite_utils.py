# ============================================================================
#  EnerMat Perovskite Explorer  v9.6  —  COMPLETE BACK‑END + FRONT‑END STACK
#  Paste the two modules below into your repo exactly as‑is:
#     • backend/perovskite_utils.py
#     • app.py  (at project root)
#  They already include the oxidation‑energy fix, halide fallback and all
#  previous stability / band‑gap improvements.  Nothing else is required.
# ============================================================================

# ---------------------------------------------------------------------------
#  File: backend/perovskite_utils.py
# ---------------------------------------------------------------------------
"""Physics helpers for EnerMat Explorer – binary & ternary screens,   
    calibrated gaps, strict optical filter, ΔE_ox correction (2025‑07‑13).
"""
from __future__ import annotations

import os, math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition

__all__ = [
    "mix_abx3", "screen_ternary", "END_MEMBERS", "fetch_mp_data",
]

# ──  API  ──────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or os.getenv("MP_API_KEY","")
if len(API_KEY) != 32:
    raise RuntimeError("🛑  32‑character MP_API_KEY missing in Secrets")
mpr = MPRester(API_KEY)

# ──  Constants & empirical offsets  ───────────────────────────────────────
END_MEMBERS   = ["CsPbBr3","CsSnBr3","CsSnCl3","CsPbI3"]
CALIBRATED_GAPS = {
    "CsSnBr3": 1.79, "CsSnCl3": 2.83, "CsSnI3": 1.30,
    "CsPbBr3": 2.30, "CsPbI3": 1.73,
}
GAP_OFFSET   = {"I":0.90,"Br":0.70,"Cl":0.80}     # PBE→exp shift
IONIC_RADII  = {
    "Cs":1.88,"Rb":1.72,"MA":2.17,"FA":2.53,
    "Pb":1.19,"Sn":1.18,"I":2.20,"Br":1.96,"Cl":1.81,
}

# ──  Tiny helpers  ────────────────────────────────────────────────────────

def _find_halide(formula:str) -> str|None:
    """Return first halogen symbol (I/Br/Cl) in formula or *None*."""
    for h in ("I","Br","Cl"):
        if h in formula:
            return h
    return None


def fetch_mp_data(formula:str, fields:list[str]) -> dict|None:
    """Single‑doc query, adds calibrated / shifted gap where needed."""
    doc = mpr.summary.search(formula=formula, fields=tuple(fields))
    if not doc:
        return None
    d = {f:getattr(doc[0],f,None) for f in fields}

    if formula in CALIBRATED_GAPS:
        d["band_gap"] = CALIBRATED_GAPS[formula]
    else:
        hal = _find_halide(formula)
        if hal is not None:
            d["band_gap"] = (d["band_gap"] or 0) + GAP_OFFSET[hal]
    return d


def _mp_formula_energy(formula:str) -> float:
    """DFT total energy (eV) per *formula* (not per atom)."""
    e_pa = fetch_mp_data(formula,["energy_per_atom"])["energy_per_atom"]
    return e_pa * Composition(formula).num_atoms


def oxidation_energy(formula:str) -> float:
    """ΔE (eV Sn⁻¹)  for  CsSnX₃  + ½ O₂  →  ½ Cs₂SnX₆  + ½ SnO₂."""
    hal  = _find_halide(formula) or "Br"          # graceful fallback
    react  = _mp_formula_energy(f"CsSn{hal}3")
    prod1  = 0.5 * _mp_formula_energy(f"Cs2Sn{hal}6")
    prod2  = 0.5 * _mp_formula_energy("SnO2")
    e_o2   = 0.5 * _mp_formula_energy("O2")
    return prod1 + prod2 - react + e_o2           # +  = uphill (good)

score_band_gap = lambda Eg,lo,hi: 1.0 if lo<=Eg<=hi else 0.0

# ──  Binary screen  ───────────────────────────────────────────────────────

def mix_abx3(
    formula_A:str, formula_B:str,
    rh:float, temp:float,
    bg_window:tuple[float,float], bowing:float=0.0, dx:float=0.05,
    alpha:float=1.0, beta:float=1.0,
) -> pd.DataFrame:
    lo,hi = bg_window
    dA,dB = (fetch_mp_data(f,["band_gap","energy_above_hull"]) for f in (formula_A,formula_B))
    if not(dA and dB):
        return pd.DataFrame()

    comp  = Composition(formula_A)
    A_site = next(e.symbol for e in comp.elements if e.symbol in IONIC_RADII)
    B_site = next(e.symbol for e in comp.elements if e.symbol in {"Pb","Sn"})
    X_site = _find_halide(formula_A) or "Br"
    rA,rB,rX = IONIC_RADII[A_site],IONIC_RADII[B_site],IONIC_RADII[X_site]

    e_ox = oxidation_energy(formula_A)   # proxy for full series
    rows=[]
    for x in np.arange(0,1+1e-6,dx):
        Eg     = (1-x)*dA["band_gap"] + x*dB["band_gap"] - bowing*x*(1-x)
        Ehull  = (1-x)*dA["energy_above_hull"] + x*dB["energy_above_hull"]
        stab   = math.exp(-max(Ehull,0)/0.1)
        gap_ok = score_band_gap(Eg,lo,hi)
        t = (rA+rX)/(math.sqrt(2)*(rB+rX));  mu=rB/rX
        form   = math.exp(-0.5*((t-0.90)/0.07)**2) * math.exp(-0.5*((mu-0.50)/0.07)**2)
        env    = 1 + alpha*rh/100 + beta*temp/100
        score  = form*stab*gap_ok*math.exp(e_ox/0.2)/env   # ΔE_ox penalty
        rows.append({"x":round(x,3),"Eg":round(Eg,3),"Ehull":round(Ehull,4),
                      "Eox":round(e_ox,3),"score":round(score,3),
                      "formula":f"{formula_A}-{formula_B} x={x:.2f}"})
    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)

# ──  Ternary screen  ──────────────────────────────────────────────────────

def screen_ternary(A:str,B:str,C:str,rh:float,temp:float,
                   bg:tuple[float,float],bows:dict[str,float],
                   dx:float=0.1,dy:float=0.1,n_mc:int=200) -> pd.DataFrame:
    dA,dB,dC = (fetch_mp_data(f,["band_gap","energy_above_hull"]) for f in (A,B,C))
    if not(dA and dB and dC):
        return pd.DataFrame()
    lo,hi=bg
    rows=[]
    for x in np.arange(0,1+1e-6,dx):
        for y in np.arange(0,1-x+1e-6,dy):
            z=1-x-y
            Eg = (z*dA["band_gap"]+x*dB["band_gap"]+y*dC["band_gap"]
                  - bows["AB"]*x*z - bows["AC"]*y*z - bows["BC"]*x*y)
            Eh = (z*dA["energy_above_hull"]+x*dB["energy_above_hull"]+y*dC["energy_above_hull"])
            stab = math.exp(-max(Eh,0)/0.1)
            gap_ok=score_band_gap(Eg,lo,hi)
            # use A's halide as surrogate for ΔE_ox
            e_ox = oxidation_energy(A)
            score=stab*gap_ok*math.exp(e_ox/0.2)
            rows.append({"x":round(x,3),"y":round(y,3),"Eg":round(Eg,3),
                          "Ehull":round(Eh,4),"Eox":round(e_ox,3),
                          "score":round(score,3),
                          "formula":f"CsSn(Br{1-x-y:.2f}Cl{y:.2f}I{x:.2f})₃"})
    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)

# legacy alias -------------------------------------------------------------
_summary = fetch_mp_data
