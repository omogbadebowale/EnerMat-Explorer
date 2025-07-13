"""
EnerMat Perovskite Explorer backend
✓ calibrated band-gaps
✓ strict optical filter
✓ Sn-oxidation penalty (ΔEox)
✓ legacy alias so old imports keep working
"""

from __future__ import annotations
import os, math, numpy as np, pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition

# ──────────────────────────────────────────────────────────────────────
# API key
# ──────────────────────────────────────────────────────────────────────
load_dotenv()
KEY = os.getenv("MP_API_KEY")
if not (KEY and len(KEY) == 32):
    raise RuntimeError("🛑 32-character MP_API_KEY missing")
mpr = MPRester(KEY)

# ──────────────────────────────────────────────────────────────────────
# presets / constants
# ──────────────────────────────────────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

GAP_OFFSET = {"I": 0.90, "Br": 0.70, "Cl": 0.80}   # PBE→exp shift (eV)

IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I" : 2.20, "Br": 1.96, "Cl": 1.81,
}

# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────
def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    """Return selected MP summary fields + calibrated band-gap."""
    doc = mpr.summary.search(formula=formula, fields=tuple(fields))
    if not doc:
        return None
    d = {f: getattr(doc[0], f) for f in fields}
    hal = next(h for h in ("I", "Br", "Cl") if h in formula)
    if "band_gap" in d and d["band_gap"] is not None:
        d["band_gap"] += GAP_OFFSET[hal]
    return d

def oxidation_energy(formula: str) -> float:
    """
    ΔE (eV per Sn) for:  CsSnX3 + ½ O2 → ½ Cs2SnX6 + ½ SnO2
    Negative = oxidation downhill (= bad for Sn2+ stability)
    """
    hal = next(h for h in ("I", "Br", "Cl") if h in formula)
    reac   = fetch_mp_data(f"CsSn{hal}3",   ["energy_per_atom"])["energy_per_atom"]
    prod_1 = fetch_mp_data(f"Cs2Sn{hal}6",  ["energy_per_atom"])["energy_per_atom"]
    prod_2 = fetch_mp_data("SnO2",          ["energy_per_atom"])["energy_per_atom"]
    e_o2   = fetch_mp_data("O2",            ["energy_per_atom"])["energy_per_atom"] * 2
    return 0.5 * prod_1 + 0.5 * prod_2 + 0.5 * e_o2 - reac   # per Sn

score_bg = lambda Eg, lo, hi: 1.0 if lo <= Eg <= hi else 0.0

# ──────────────────────────────────────────────────────────────────────
# binary screen
# ──────────────────────────────────────────────────────────────────────
def mix_abx3(
    A: str, B: str,
    rh: float, temp: float,
    bg_win: tuple[float,float],
    bow: float = 0.0, dx: float = 0.05,
) -> pd.DataFrame:

    lo, hi = bg_win
    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    comp = Composition(A)
    A_site = next(e.symbol for e in comp.elements if e.symbol in IONIC_RADII)
    B_site = next(e.symbol for e in comp.elements if e.symbol in {"Pb", "Sn"})
    X_site = next(e.symbol for e in comp.elements if e.symbol in {"I","Br","Cl"})
    rA,rB,rX = IONIC_RADII[A_site], IONIC_RADII[B_site], IONIC_RADII[X_site]

    rows=[]
    e_ox = oxidation_energy(A)  # use A’s halide as proxy for the series
    for x in np.arange(0,1+1e-6,dx):
        Eg    = (1-x)*dA["band_gap"] + x*dB["band_gap"] - bow*x*(1-x)
        Ehull = (1-x)*dA["energy_above_hull"] + x*dB["energy_above_hull"]
        t     = (rA+rX)/ (math.sqrt(2)*(rB+rX))
        mu    = rB/rX
        form  = math.exp(-0.5*((t-0.90)/0.07)**2) * math.exp(-0.5*((mu-0.50)/0.07)**2)
        stab  = math.exp(-max(Ehull,0)/0.1)
        oxi   = math.exp(-max(-e_ox,0)/0.2)         # mild penalty for |ΔEox|
        score = form * stab * score_bg(Eg,lo,hi) * oxi / (1+rh/100+temp/100)

        rows.append({
            "x": round(x,3),
            "Eg": round(Eg,3),
            "Ehull": round(Ehull,4),
            "Eox": round(e_ox,3),
            "score": round(score,3),
            "formula": f"{A}-{B} x={x:.2f}",
        })
    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────
# ternary screen (unchanged except for new column names)
# ──────────────────────────────────────────────────────────────────────
def screen_ternary(
    A:str,B:str,C:str, rh:float,temp:float,
    bg:tuple[float,float], bows:dict[str,float],
    dx:float=0.1, dy:float=0.1
)->pd.DataFrame:

    dA=fetch_mp_data(A,["band_gap","energy_above_hull"])
    dB=fetch_mp_data(B,["band_gap","energy_above_hull"])
    dC=fetch_mp_data(C,["band_gap","energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    lo,hi = bg
    rows=[]
    for x in np.arange(0,1+1e-6,dx):
        for y in np.arange(0,1-x+1e-6,dy):
            z = 1-x-y
            Eg = (z*dA["band_gap"] + x*dB["band_gap"] + y*dC["band_gap"]
                  - bows["AB"]*x*z - bows["AC"]*y*z - bows["BC"]*x*y)
            Eh = (z*dA["energy_above_hull"] + x*dB["energy_above_hull"] + y*dC["energy_above_hull"]
                  + bows["AB"]*x*z + bows["AC"]*y*z + bows["BC"]*x*y)
            score = math.exp(-max(Eh,0)/0.1) * score_bg(Eg,lo,hi)
            rows.append({
                "x":round(x,3),"y":round(y,3),
                "Eg":round(Eg,3),"Ehull":round(Eh,4),
                "score":round(score,3),
                "formula":f"{A}:{1-x-y:.2f} {B}:{x:.2f} {C}:{y:.2f}"
            })
    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────
# legacy alias so old imports (fetch_mp_doc) still work
# ──────────────────────────────────────────────────────────────────────
fetch_mp_doc = fetch_mp_data   # <-- keeps earlier app.py versions alive
