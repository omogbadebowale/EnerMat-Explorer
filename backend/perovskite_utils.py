"""
EnerMat Perovskite Explorer – physics backend
clean 2025-07-13
───────────────────────────────────────────────────────────────
* calibrated PBE gaps  (+0.70/0.80/0.90 eV offsets)
* strict 0/1 optical window
* formation-energy based Sn-oxidation penalty  (ΔEox)
* Ehull, Eg, Eox and formula surfaced for both binary & ternary grids
───────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import os, math, numpy as np, pandas as pd, streamlit as st
from dotenv import load_dotenv
from pymatgen.core import Composition
from mp_api.client import MPRester

# ─────────────────────────  API  ─────────────────────────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑  32-character MP_API_KEY missing")
mpr = MPRester(API_KEY)

# ──────────────── presets, offsets, constants ────────────────
END_MEMBERS   = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]
GAP_OFFSET    = {"I": 0.90, "Br": 0.70, "Cl": 0.80}          # eV
IONIC_RADII   = {"Cs":1.88,"Rb":1.72,"MA":2.17,"FA":2.53,
                 "Pb":1.19,"Sn":1.18,"I":2.20,"Br":1.96,"Cl":1.81}

# calibrated single-phase gaps (if available)
CALIBRATED_GAPS = {
    "CsSnBr3": 1.79, "CsSnCl3": 2.83, "CsSnI3": 1.30,
    "CsPbBr3": 2.30, "CsPbI3": 1.73,
}

# ─────────────────── helpers (cached) ─────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _mp_summary(formula: str, fields: list[str]) -> dict | None:
    docs = mpr.summary.search(formula=formula, fields=tuple(fields))
    return {k:getattr(docs[0],k) for k in fields} if docs else None


def fetch_gap(formula: str) -> float:
    """
    Calibrated band-gap for a single phase:
    • use hard-coded value if in CALIBRATED_GAPS
    • else apply halogen-specific offset to MP PBE gap
    """
    if formula in CALIBRATED_GAPS:
        return CALIBRATED_GAPS[formula]

    s = _mp_summary(formula, ["band_gap"])
    if not s: raise ValueError(f"gap: {formula} not in MP")
    hal = next(h for h in ("I","Br","Cl") if h in formula)
    return (s["band_gap"] or 0.0) + GAP_OFFSET[hal]


def formula_energy(formula: str) -> float:
    """
    Formation energy (eV) of one *reduced* formula unit.
    Positive = unstable wrt. elements   (MP sign convention)
    """
    s = _mp_summary(formula, ["formation_energy_per_atom"])
    if not s: raise ValueError(f"energy: {formula} not in MP")
    n = Composition(formula).num_atoms
    return s["formation_energy_per_atom"] * n


@st.cache_data(ttl=3600, show_spinner=False)      # cache per halide an hour
def oxidation_energy(hal: str) -> float:
    """
    ΔE (eV Sn⁻¹) for  2 CsSnX₃ + O₂ → Cs₂SnX₆ + SnO₂   (X = hal)
    Negative = oxidation exergonic (undesirable)
    """
    if hal not in "IBrCl": raise ValueError("hal must be I/Br/Cl")
    reac   = 2.0 * formula_energy(f"CsSn{hal}3")
    prod   =       formula_energy(f"Cs2Sn{hal}6") + formula_energy("SnO2")
    o2     =       formula_energy("O2")
    return (prod - (reac + o2))        # already “per Sn” (1 Sn in lhs)


def optical_weight(Eg: float, lo: float, hi: float) -> float:
    return 1.0 if lo <= Eg <= hi else 0.0


# ─────────────────────  binary scan  ──────────────────────────
def mix_abx3(
    A: str, B: str,
    rh: float, temp: float,
    bg_window: tuple[float,float],
    bow: float = 0.0, dx: float = 0.05,
    alpha: float = 1.0, beta: float = 1.0,
) -> pd.DataFrame:

    gA, gB = fetch_gap(A), fetch_gap(B)
    sA, sB = (_mp_summary(A,["energy_above_hull"])["energy_above_hull"],
              _mp_summary(B,["energy_above_hull"])["energy_above_hull"])
    if sA is None or sB is None: return pd.DataFrame()

    X_site = next(h for h in ("I","Br","Cl") if h in A)
    ΔEox   = oxidation_energy(X_site)            # constant along the tie-line

    comp   = Composition(A)
    A_site = next(e.symbol for e in comp.elements if e.symbol in IONIC_RADII)
    rA,rB,rX = IONIC_RADII[A_site],IONIC_RADII["Sn"],IONIC_RADII[X_site]

    lo,hi  = bg_window
    rows   = []
    for x in np.arange(0,1+1e-6,dx):
        Eg     = (1-x)*gA + x*gB - bow*x*(1-x)
        Eh     = (1-x)*sA + x*sB
        stab   = math.exp(-max(Eh,0)/0.1)            # 0.1 eV width
        gap    = optical_weight(Eg,lo,hi)
        t      = (rA+rX)/(math.sqrt(2)*(rB+rX))
        mu     = rB/rX
        form   = math.exp(-0.5*((t-0.90)/0.07)**2)*math.exp(-0.5*((mu-0.50)/0.07)**2)
        env    = 1 + alpha*rh/100 + beta*temp/100
        oxi    = math.exp(ΔEox/0.20)                 # penalty: more neg → smaller
        score  = form * stab * gap * oxi / env

        rows.append(dict(x=round(x,3), Eg=round(Eg,3),
                         Ehull=round(Eh,4), Eox=round(ΔEox,3),
                         score=round(score,3),
                         formula=f"{A}-{B} x={x:.2f}"))

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# ─────────────────────  ternary scan  ─────────────────────────
def screen_ternary(
    A:str, B:str, C:str, rh:float, temp:float,
    bg:tuple[float,float], bows:dict[str,float],
    dx:float=0.1, dy:float=0.1, n_mc:int=200,
)->pd.DataFrame:

    gA,gB,gC = fetch_gap(A),fetch_gap(B),fetch_gap(C)
    sA,sB,sC = (_mp_summary(A,["energy_above_hull"])["energy_above_hull"],
                _mp_summary(B,["energy_above_hull"])["energy_above_hull"],
                _mp_summary(C,["energy_above_hull"])["energy_above_hull"])
    if None in (sA,sB,sC): return pd.DataFrame()

    # oxidation term for each halogen
    ox = {h:oxidation_energy(h) for h in "IBrCl"}

    lo,hi = bg
    rows  = []
    for x in np.arange(0,1+1e-6,dx):
        for y in np.arange(0,1-x+1e-6,dy):
            z   = 1-x-y
            Eg  = (z*gA + x*gB + y*gC
                   - bows["AB"]*x*z - bows["AC"]*y*z - bows["BC"]*x*y)
            Eh  = (z*sA + x*sB + y*sC
                   + bows["AB"]*x*z + bows["AC"]*y*z + bows["BC"]*x*y)

            # simple halogen-fraction weighted ΔEox
            hal_frac = dict(I=0,Br=0,Cl=0)
            for frac,form in ((z,A),(x,B),(y,C)):
                h = next(h for h in "IBrCl" if h in form)
                hal_frac[h] += frac
            ΔEox = sum(hal_frac[h]*ox[h] for h in hal_frac)

            score = (math.exp(-max(Eh,0)/0.1) * optical_weight(Eg,lo,hi)
                     * math.exp(ΔEox/0.20))    # same 0.20 eV scale
            rows.append(dict(x=round(x,3),y=round(y,3),Eg=round(Eg,3),
                             Ehull=round(Eh,4),Eox=round(ΔEox,3),
                             score=round(score,3),
                             formula=f"{A}-{B}-{C}  x={x:.2f} y={y:.2f}"))

    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)

# legacy alias
_summary = _mp_summary
