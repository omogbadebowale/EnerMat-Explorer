# backend/perovskite_utils.py   ⟨v0.9.7 – 25 Jun 2025⟩
import os, numpy as np, pandas as pd, streamlit as st
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition

load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("MP_API_KEY missing or wrong length (needs 32 chars)")

mpr = MPRester(API_KEY)

# —————————————————————————————————————————
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]
IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81,
}
kT_EXP = 0.06               # eV – controls exponential stability weight
# —————————————————————————————————————————


def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    """Return dict of lowest-energy MP entry’s requested fields or None."""
    docs = mpr.summary.search(formula=formula)
    if not docs:
        return None
    entry = min(docs, key=lambda d: d.energy_above_hull)   # <-- always ground-state
    return {f: getattr(entry, f) for f in fields if hasattr(entry, f)}


def score_band_gap(bg: float, lo: float, hi: float) -> float:
    if lo <= bg <= hi:
        return 1.0
    delta = min(abs(bg - lo), abs(bg - hi))
    return max(0.0, 1 - delta / (hi - lo))


def _form_score(rA, rB, rX) -> float:
    t  = (rA + rX) / (np.sqrt(2) * (rB + rX))
    mu =  rB / rX
    return np.exp(-0.5 * ((t  - 0.90) / 0.07)**2) * \
           np.exp(-0.5 * ((mu - 0.50) / 0.07)**2)


# ─────────────────────────────────────────────────────────────────────────────
def mix_abx3(
    formula_A: str,
    formula_B: str,
    rh: float,
    temp: float,
    bg_window: tuple[float, float],
    bowing : float = 0.30,
    dx     : float = 0.05,
    alpha  : float = 1.0,
    beta   : float = 1.0,
) -> pd.DataFrame:
    """Binary scan 0 ≤ x ≤ 1 (step = dx)."""
    lo, hi = bg_window
    dA = fetch_mp_data(formula_A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(formula_B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    comp  = Composition(formula_A)
    A_site = next(e.symbol for e in comp.elements if e.symbol in IONIC_RADII)
    B_site = next(e.symbol for e in comp.elements if e.symbol in {"Pb", "Sn"})
    X_site = next(e.symbol for e in comp.elements if e.symbol in {"I", "Br", "Cl"})
    rA, rB, rX = IONIC_RADII[A_site], IONIC_RADII[B_site], IONIC_RADII[X_site]
    form_factor = _form_score(rA, rB, rX)

    rows: list[dict] = []
    for x in np.arange(0, 1 + 1e-9, dx):
        Eg   = (1 - x)*dA["band_gap"] + x*dB["band_gap"] - bowing * x*(1 - x)
        Eh   = (1 - x)*dA["energy_above_hull"] + x*dB["energy_above_hull"]
        stab = np.exp(- max(Eh, 0) / kT_EXP)          # 0-1 exponential weight
        gap  = score_band_gap(Eg, lo, hi)
        env  = 1 + alpha*np.log1p(rh) + beta*(temp/100)
        S    = form_factor * stab * gap / env
        rows.append({
            "x": round(x,3), "Eg": round(Eg,3),
            "stability": round(stab,3),
            "gap_score": round(gap,3),
            "score": round(S,3),
            "formula": f"{formula_A}-{formula_B} x={x:.2f}",
        })

    return (pd.DataFrame(rows)
              .sort_values("score", ascending=False)
              .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
def screen_ternary(
    A       : str, B: str, C: str,
    rh      : float, temp: float,
    bg      : tuple[float, float],
    bows    : dict[str, float],
    dx      : float = 0.1, dy: float = 0.1,
) -> pd.DataFrame:
    """Full grid (x,y) where z = 1-x-y."""
    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    lo, hi = bg
    rows: list[dict] = []

    for x in np.arange(0, 1 + 1e-9, dx):
        for y in np.arange(0, 1 - x + 1e-9, dy):
            z  = 1 - x - y
            Eg = (z*dA["band_gap"] + x*dB["band_gap"] + y*dC["band_gap"]
                  - bows["AB"]*x*z - bows["AC"]*y*z - bows["BC"]*x*y)
            Eh = (z*dA["energy_above_hull"] + x*dB["energy_above_hull"] + y*dC["energy_above_hull"]
                  + bows["AB"]*x*z + bows["AC"]*y*z + bows["BC"]*x*y)
            stab = np.exp(- max(Eh, 0) / kT_EXP)
            gap  = score_band_gap(Eg, lo, hi)
            S    = stab * gap                    # env_pen omitted to match paper
            rows.append({
                "x": round(x,3), "y": round(y,3),
                "Eg": round(Eg,3),
                "stability": round(stab,3),
                "gap_score": round(gap,3),
                "score": round(S,3),
            })

    return (pd.DataFrame(rows)
              .sort_values("score", ascending=False)
              .reset_index(drop=True))


# for app-level import compatibility
_summary = fetch_mp_data
