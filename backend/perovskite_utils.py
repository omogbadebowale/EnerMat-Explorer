"""
EnerMat Perovskite Explorer – backend helpers
2025-07-13  • 4-space indents, calibrated gaps
"""

from __future__ import annotations
import os, math, numpy as np, pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition

# ── API KEY ─────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32-character MP_API_KEY missing")
mpr = MPRester(API_KEY)

# ── PRESETS & CORRECTIONS ───────────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

CALIBRATED_GAPS = {
    "CsSnBr3": 1.79, "CsSnCl3": 2.83, "CsSnI3": 1.30,
    "CsPbBr3": 2.30, "CsPbI3": 1.73,
}
GAP_OFFSET = {"I": 0.90, "Br": 0.70, "Cl": 0.80}

IONIC_RADII = {"Cs": 1.88, "Pb": 1.19, "Sn": 1.18,
               "I":  2.20, "Br": 1.96, "Cl": 1.81}

# ── INTERNAL HELPERS ────────────────────────────────────────────────
def _find_halide(formula: str) -> str:
    return next(h for h in ("I", "Br", "Cl") if h in formula)

def fetch_mp_data(formula: str, fields: list[str]) -> dict:
    doc = mpr.summary.search(formula=formula, fields=tuple(fields))[0]
    d = {f: getattr(doc, f, None) for f in fields}
    if formula in CALIBRATED_GAPS:
        d["band_gap"] = CALIBRATED_GAPS[formula]
    else:
        hal = _find_halide(formula)
        d["band_gap"] = (d["band_gap"] or 0) + GAP_OFFSET[hal]
    return d

def _mp_formula_energy(formula: str) -> float:
    """DFT total energy (eV) per *formula unit*"""
    e_pa = fetch_mp_data(formula, ["energy_per_atom"])["energy_per_atom"]
    return e_pa * Composition(formula).num_atoms

def oxidation_energy(formula: str) -> float:
    """
    ΔE (eV Sn⁻¹) for:  CsSnX3 + ½O2 → ½Cs2SnX6 + ½SnO2
    Positive → oxidation uphill (good)
    """
    hal   = _find_halide(formula)
    react = _mp_formula_energy(formula)
    prod1 = 0.5 * _mp_formula_energy(f"Cs2Sn{hal}6")
    prod2 = 0.5 * _mp_formula_energy("SnO2")
    e_o2  = 0.5 * _mp_formula_energy("O2")
    return prod1 + prod2 - react + e_o2     # already per Sn (one Sn each side)

score_gap = lambda Eg, lo, hi: 1.0 if lo <= Eg <= hi else 0.0

# ── BINARY SCREEN ───────────────────────────────────────────────────
def mix_abx3(
    A: str, B: str,
    rh: float, temp: float,
    bg_window: tuple[float, float],
    bowing: float = 0.0,
    dx: float = 0.05,
    alpha: float = 1.0, beta: float = 1.0,
) -> pd.DataFrame:

    dA, dB = fetch_mp_data(A, ["band_gap","energy_above_hull"]), \
             fetch_mp_data(B, ["band_gap","energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    rA,rB,rX = (IONIC_RADII[k] for k in ("Cs","Sn",_find_halide(A)))
    e_oxA = oxidation_energy(A)            # proxy for whole series

    lo, hi = bg_window
    rows=[]
    for x in np.arange(0,1+1e-6,dx):
        Eg = (1-x)*dA["band_gap"] + x*dB["band_gap"] - bowing*x*(1-x)
        Eh = (1-x)*dA["energy_above_hull"] + x*dB["energy_above_hull"]
        stab = math.exp(-max(Eh,0)/0.1)
        gap  = score_gap(Eg, lo, hi)
        t = (rA+rX)/(math.sqrt(2)*(rB+rX)); mu = rB/rX
        form = math.exp(-0.5*((t-0.9)/0.07)**2)*math.exp(-0.5*((mu-0.5)/0.07)**2)
        env  = 1 + alpha*rh/100 + beta*temp/100
        score = form*stab*gap*math.exp(e_oxA/0.2)/env

        rows.append(dict(x=round(x,3), Eg=round(Eg,3),
                         Ehull=round(Eh,4), Eox=round(e_oxA,3),
                         score=round(score,3),
                         formula=f"{A}-{B} x={x:.2f}"))
    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)

# ── TERNARY SCREEN (unchanged except Eox column added) ──────────────
def screen_ternary(
    A:str,B:str,C:str, rh:float,temp:float,
    bg:tuple[float,float], bows:dict[str,float],
    dx:float=0.1, dy:float=0.1,
) -> pd.DataFrame:

    dA,dB,dC = [fetch_mp_data(f,["band_gap","energy_above_hull"]) for f in (A,B,C)]
    if not (dA and dB and dC):
        return pd.DataFrame()

    lo,hi = bg
    e_oxA = oxidation_energy(A)
    rows=[]
    for x in np.arange(0,1+1e-6,dx):
      for y in np.arange(0,1-x+1e-6,dy):
        z=1-x-y
        Eg = z*dA["band_gap"]+x*dB["band_gap"]+y*dC["band_gap"] \
             - bows["AB"]*x*z - bows["AC"]*y*z - bows["BC"]*x*y
        Eh = z*dA["energy_above_hull"]+x*dB["energy_above_hull"]+y*dC["energy_above_hull"]
        score = math.exp(-max(Eh,0)/0.1)*score_gap(Eg,lo,hi)*math.exp(e_oxA/0.2)
        rows.append(dict(x=round(x,3),y=round(y,3),Eg=round(Eg,3),
                         Ehull=round(Eh,4),Eox=round(e_oxA,3),
                         score=round(score,3),
                         formula=f"CsSn(Br{z:.2f}Cl{y:.2f}I{x:.2f})3"))
    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)
