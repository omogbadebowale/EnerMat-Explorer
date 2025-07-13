"""
EnerMat-Explorer backend — calibrated gaps, strict optical filter,
optional Sn-oxidation penalty.
CLEAN 2025-07-13  (safe fetch_mp_data)
"""

from __future__ import annotations
import os, math, numpy as np, pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition

# ── API key ─────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32-character MP_API_KEY missing")
mpr = MPRester(API_KEY)

# ── Presets & corrections ──────────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

CALIBRATED_GAPS = {          # experimental eV
    "CsSnBr3": 1.79,
    "CsSnCl3": 2.83,
    "CsSnI3":  1.30,
    "CsPbBr3": 2.30,
    "CsPbI3":  1.73,
}
GAP_OFFSET = {"I": 0.90, "Br": 0.70, "Cl": 0.80}

IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

# ── Helpers ────────────────────────────────────────────────────────
def _find_halide(formula: str) -> str:
    return next(h for h in ("I", "Br", "Cl") if h in formula)

def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    """
    Query Materials Project for *fields* and apply calibrated /
    offset band-gap if that field is requested.
    Safe even when 'band_gap' is NOT in *fields*.
    """
    docs = mpr.summary.search(formula=formula, fields=tuple(fields))
    if not docs:
        return None

    entry = docs[0]
    d = {f: getattr(entry, f, None) for f in fields}

    if "band_gap" in fields:
        if formula in CALIBRATED_GAPS:
            d["band_gap"] = CALIBRATED_GAPS[formula]
        else:
            hal = _find_halide(formula)
            raw_bg = d.get("band_gap", 0.0) or 0.0
            d["band_gap"] = raw_bg + GAP_OFFSET[hal]

    return d

score_band_gap = lambda Eg, lo, hi: 1.0 if lo <= Eg <= hi else 0.0

# ── Oxidation energy (Sn²⁺ → Sn⁴⁺) ────────────────────────────────
def oxidation_energy(formula: str) -> float:
    """
    ΔE per Sn for:
        CsSnX₃ + ½ O₂  →  ½ Cs₂SnX₆ + ½ SnO₂
    Negative ⇒ oxidation downhill.
    """
    hal  = _find_halide(formula)
    react = fetch_mp_data(f"CsSn{hal}3",  ["energy_per_atom"])["energy_per_atom"]
    prod1 = fetch_mp_data(f"Cs2Sn{hal}6", ["energy_per_atom"])["energy_per_atom"]
    prod2 = fetch_mp_data("SnO2",         ["energy_per_atom"])["energy_per_atom"]
    e_o2  = fetch_mp_data("O2",           ["energy_per_atom"])["energy_per_atom"] * 2
    return 0.5*prod1 + 0.5*prod2 - react + 0.5*e_o2   # eV per Sn

# ── Binary screen ─────────────────────────────────────────────────
def mix_abx3(
        formula_A: str,
        formula_B: str,
        rh: float, temp: float,
        bg_window: tuple[float, float],
        bowing: float = 0.0,
        dx: float = 0.05,
        alpha: float = 1.0, beta: float = 1.0,
) -> pd.DataFrame:

    lo, hi = bg_window
    dA = fetch_mp_data(formula_A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(formula_B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    comp = Composition(formula_A)              # A-site is shared
    A_site = next(e.symbol for e in comp.elements if e.symbol in IONIC_RADII)
    B_site = next(e.symbol for e in comp.elements if e.symbol in {"Pb", "Sn"})
    X_site = _find_halide(formula_A)
    rA, rB, rX = IONIC_RADII[A_site], IONIC_RADII[B_site], IONIC_RADII[X_site]

    e_ox = oxidation_energy(formula_A)         # use A’s halide as proxy
    rows = []
    for x in np.arange(0.0, 1.0 + 1e-6, dx):
        Eg    = (1-x)*dA["band_gap"] + x*dB["band_gap"] - bowing*x*(1-x)
        Ehull = (1-x)*dA["energy_above_hull"] + x*dB["energy_above_hull"]
        stab  = max(0.0, 1 - Ehull)
        gap   = score_band_gap(Eg, lo, hi)
        t     = (rA+rX) / (math.sqrt(2)*(rB+rX))
        mu    = rB / rX
        form  = math.exp(-0.5*((t-0.90)/0.07)**2) * math.exp(-0.5*((mu-0.50)/0.07)**2)
        env   = 1 + alpha*rh/100 + beta*temp/100
        ox_pen = math.exp(-max(e_ox,0)/0.2)

        score = form * stab * gap * ox_pen / env
        rows.append({
            "x":      round(x,3),
            "Eg":     round(Eg,3),
            "Ehull":  round(Ehull,4),
            "Eox":    round(e_ox,3),
            "score":  round(score,3),
            "formula": f"{formula_A}-{formula_B} x={x:.2f}",
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# ── Ternary screen (unchanged) ─────────────────────────────────────
def screen_ternary(
        A: str, B: str, C: str,
        rh: float, temp: float,
        bg: tuple[float,float],
        bows: dict[str,float],
        dx: float=0.1, dy: float=0.1,
) -> pd.DataFrame:

    dA = fetch_mp_data(A, ["band_gap","energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap","energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap","energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    lo, hi = bg
    rows=[]
    for x in np.arange(0,1+1e-6,dx):
        for y in np.arange(0,1-x+1e-6,dy):
            z = 1 - x - y
            Eg = (z*dA["band_gap"] + x*dB["band_gap"] + y*dC["band_gap"]
                  - bows["AB"]*x*z - bows["AC"]*y*z - bows["BC"]*x*y)
            Eh = (z*dA["energy_above_hull"] + x*dB["energy_above_hull"] + y*dC["energy_above_hull"]
                  + bows["AB"]*x*z + bows["AC"]*y*z + bows["BC"]*x*y)
            score = math.exp(-max(Eh,0)/0.1) * score_band_gap(Eg, lo, hi)
            rows.append({"x":round(x,3),"y":round(y,3),"Eg":round(Eg,3),
                         "Ehull":round(Eh,4),"score":round(score,3)})
    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)

# legacy alias
_summary = fetch_mp_data
