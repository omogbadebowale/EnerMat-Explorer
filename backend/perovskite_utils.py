"""
EnerMat Perovskite Explorer – backend/perovskite_utils.py
FULL FILE – hash‑safe MP query (2025‑07‑12)
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Composition

# ── Materials‑Project key ───────────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32‑char MP_API_KEY missing")

mpr = MPRester(API_KEY)

# ── Preset end‑members ─────────────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

# ── Experimental gaps & scissor offsets ────────────────────────────
CALIBRATED_GAPS = {
    "CsSnBr3": 1.79, "CsSnCl3": 2.83, "CsSnI3": 1.30,
    "CsPbBr3": 2.30, "CsPbI3": 1.73,
}
GAP_OFFSET = {"I": 0.90, "Br": 0.70, "Cl": 0.80}

# ── Ionic radii (Å) ────────────────────────────────────────────────
IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

# ── Safe MP fetch helper ───────────────────────────────────────────

def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    """Return dict of requested fields; robust to list‑hash issue."""
    f_tuple = tuple(fields)                     # hash‑safe for mp_api
    try:
        docs = mpr.summary.search(formula=formula, fields=f_tuple)
    except TypeError as err:
        raise RuntimeError(f"MP query failed for {formula}: {err}") from err

    if not docs:
        return None

    entry = docs[0]
    data = {f: getattr(entry, f, None) for f in fields}

    # apply experimental override or scissor
    if formula in CALIBRATED_GAPS:
        data["band_gap"] = CALIBRATED_GAPS[formula]
    else:
        hal = next(h for h in ("I", "Br", "Cl") if h in formula)
        data["band_gap"] = (data["band_gap"] or 0) + GAP_OFFSET[hal]
    return data

# ── Optical merit (strict) ─────────────────────────────────────────
score_band_gap = lambda Eg, lo, hi: 1.0 if lo <= Eg <= hi else 0.0

# ── Binary screen ─────────────────────────────────────────────────

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

    rows = []
    for x in np.arange(0, 1 + 1e-6, dx):
        Eg = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bowing * x * (1 - x)
        Ehull = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        stab = max(0.0, 1 - Ehull)
        gap = score_band_gap(Eg, lo, hi)
        t = (rA + rX) / (np.sqrt(2) * (rB + rX))
        mu = rB / rX
        form = np.exp(-0.5 * ((t - 0.90)/0.07)**2) * np.exp(-0.5 * ((mu - 0.50)/0.07)**2)
        env = 1 + alpha * rh/100 + beta * temp/100
        score = form * stab * gap / env
        rows.append({"x": round(x,3), "Eg": round(Eg,3), "Ehull": round(Ehull,4), "score": round(score,3)})

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# ── Ternary screen (unchanged logic but Ehull column kept) ─────────

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
    rows = []
    for x in np.arange(0, 1 + 1e-6, dx):
        for y in np.arange(0, 1 - x + 1e-6, dy):
            z = 1 - x - y
            Eg = (z*dA["band_gap"] + x*dB["band_gap"] + y*dC["band_gap"]
                 - bows["AB"]*x*z - bows["AC"]*y*z - bows["BC"]*x*y)
            Eh = (z*dA["energy_above_hull"] + x*dB["energy_above_hull"] + y*dC["energy_above_hull"]
                 + bows["AB"]*x*z + bows["AC"]*y*z + bows["BC"]*x*y)
            score = np.exp(-max(Eh,0)/0.1) * score_band_gap(Eg,lo,hi)
            rows.append({"x":round(x,3),"y":round(y,3),"Eg":round(Eg,3),"Ehull":round(Eh,4),"score":round(score,3)})
    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)

# legacy alias
_summary = fetch_mp_data
