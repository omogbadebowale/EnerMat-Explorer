"""
EnerMat Perovskite Explorer – backend/perovskite_utils.py
CLEAN VERSION (2025‑07‑12)
• hash‑safe MP queries
• calibrated gaps + scissor offsets
• strict optical filter
• formula column included
• syntax errors removed
"""

from __future__ import annotations
import os, numpy as np, pandas as pd, streamlit as st
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition

load_dotenv()

# ── API KEY ─────────────────────────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32‑character MP_API_KEY missing")

mpr = MPRester(API_KEY)

# ── PRESETS & DATA ─────────────────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

CALIBRATED_GAPS = {
    "CsSnBr3": 1.79,
    "CsSnCl3": 2.83,
    "CsSnI3" : 1.30,
    "CsPbBr3": 2.30,
    "CsPbI3" : 1.73,
}
GAP_OFFSET = {"I": 0.90, "Br": 0.70, "Cl": 0.80}

IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

# ── HELPER: SAFE MP FETCH ──────────────────────────────────────────

def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    fields_t = tuple(fields)  # hash‑safe
    try:
        docs = mpr.summary.search(formula=formula, fields=fields_t)
    except TypeError as err:
        raise RuntimeError(f"MP query failed for {formula}: {err}") from err
    if not docs:
        return None
    entry = docs[0]
    data = {f: getattr(entry, f, None) for f in fields}
    # gap correction
    if formula in CALIBRATED_GAPS:
        data["band_gap"] = CALIBRATED_GAPS[formula]
    else:
        hal = next(h for h in ("I", "Br", "Cl") if h in formula)
        data["band_gap"] = (data["band_gap"] or 0) + GAP_OFFSET[hal]
    return data

# ── OPTICAL MERIT ──────────────────────────────────────────────────
score_band_gap = lambda Eg, lo, hi: 1.0 if lo <= Eg <= hi else 0.0

# ── BINARY SCREEN ─────────────────────────────────────────────────-

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

    lo, hi = bg_window
    dA = fetch_mp_data(formula_A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(formula_B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    comp = Composition(formula_A)
    A_site = next(e.symbol for e in comp.elements if e.symbol in IONIC_RADII)
    B_site = next(e.symbol for e in comp.elements if e.symbol in {"Pb", "Sn"})
    X_site = next(e.symbol for e in comp.elements if e.symbol in {"I", "Br", "Cl"})
    rA, rB, rX = IONIC_RADII[A_site], IONIC_RADII[B_site], IONIC_RADII[X_site]

    rows: list[dict] = []
    for x in np.arange(0, 1 + 1e-6, dx):
        Eg = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bowing * x * (1 - x)
        Ehull = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        stab = max(0.0, 1 - Ehull)
        gap  = score_band_gap(Eg, lo, hi)
        t    = (rA + rX) / (np.sqrt(2) * (rB + rX))
        mu   = rB / rX
        form = np.exp(-0.5 * ((t - 0.90)/0.07)**2) * np.exp(-0.5 * ((mu - 0.50)/0.07)**2)
        env  = 1 + alpha * rh/100 + beta * temp/100
        score = form * stab * gap / env
        rows.append({
            "x": round(x, 3),
            "Eg": round(Eg, 3),
            "Ehull": round(Ehull, 4),
            "score": round(score, 3),
            "formula": f"{formula_A}-{formula_B} x={x:.2f}"
        })

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# ── TERNARY SCREEN ─────────────────────────────────────────────────

def screen_ternary(
    A: str, B: str, C: str,
    rh: float, temp: float,
    bg: tuple[float, float],
    bows: dict[str, float],
    dx: float = 0.1, dy: float = 0.1,
    n_mc: int = 200,
) -> pd.DataFrame:

    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    lo, hi = bg
    rows: list[dict] = []
    for x in np.arange(0, 1 + 1e-6, dx):
        for y in np.arange(0, 1 - x + 1e-6, dy):
            z = 1 - x - y
            Eg = (
                z*dA["band_gap"] + x*dB["band_gap"] + y*dC["band_gap"]
                - bows["AB"]*x*z - bows["AC"]*y*z - bows["BC"]*x*y
            )
            Eh = (
                z*dA["energy_above_hull"] + x*dB["energy_above_hull"] + y*dC["energy_above_hull"]
                + bows["AB"]*x*z + bows["AC"]*y*z + bows["BC"]*x*y
            )
            score = np.exp(-max(Eh,0)/0.1) * score_band_gap(Eg, lo, hi)
            rows.append({
                "x": round(x, 3), "y": round(y, 3),
                "Eg": round(Eg, 3), "Ehull": round(Eh, 4),
                "score": round(score, 3)
            })

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# ── Alias for legacy code ─────────────────────────────────────────
_summary = fetch_mp_data
