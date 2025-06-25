# ─────────────────────────────────────────────────────────────────────────────
#  Perovskite-utils v1.2  (2025-06-25)       author: G. T. Yusuf et al.
#  • lowest-Ehull polymorph selection
#  • exponential stability weight  exp(-Ehull / kTexp)   (kTexp = 0.06 eV)
#  • ‘gap_score’ column exposed
#  • ternary env-penalty now uses the same α, β knobs as binary
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()          # .env for local dev; silently ignored on Streamlit Cloud

import streamlit as st
from mp_api.client import MPRester
from pymatgen.core import Composition

# ── Materials-Project key ────────────────────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32-character MP_API_KEY missing.")

mpr = MPRester(API_KEY)

# ── End-member presets & ionic radii ─────────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

# ── Helper: always pick the lowest-Ehull polymorph ───────────────────────────
def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    docs = mpr.summary.search(formula=formula)
    if not docs:
        return None
    entry = min(docs, key=lambda d: d.energy_above_hull or 1e9)
    return {f: getattr(entry, f) for f in fields if hasattr(entry, f)}

# ── Band-gap fitness inside a target window ──────────────────────────────────
def score_band_gap(Eg: float, lo: float, hi: float) -> float:
    if Eg < lo:
        return max(0.0, 1 - (lo - Eg) / (hi - lo))
    if Eg > hi:
        return max(0.0, 1 - (Eg - hi) / (hi - lo))
    return 1.0

# ── Binary screen ────────────────────────────────────────────────────────────
def mix_abx3(
    formula_A: str,
    formula_B: str,
    rh: float,
    temp: float,
    bg_window: tuple[float, float],
    bowing: float = 0.0,
    dx: float = 0.05,
    alpha: float = 1.0,
    beta:  float = 1.0,
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
    rA, rB, rX = IONIC_RADII[A_site], IONIC_RADII[B_site], IONIC_RADII[X_SITE]

    kT_exp   = 0.06  #   “effective synthesis temperature” (eV)

    rows = []
    grid = np.arange(0, 1 + 1e-6, dx)
    for x in grid:
        Eg   = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bowing * x * (1 - x)
        Eh   = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        stab = np.exp(-max(Eh, 0) / kT_exp)          # 0…1
        gfit = score_band_gap(Eg, lo, hi)

        # Goldschmidt formability term
        t  = (rA + rX) / (np.sqrt(2) * (rB + rX))
        mu = rB / rX
        form = np.exp(-0.5 * ((t  - 0.90) / 0.07) ** 2) * \
               np.exp(-0.5 * ((mu - 0.50) / 0.07) ** 2)

        env  = 1 + alpha * np.log1p(rh) + beta * (temp / 100)
        S    = form * stab * gfit / env

        rows.append(dict(
            x=round(x,3), Eg=round(Eg,3),
            stability=round(stab,3), gap_score=round(gfit,3),
            score=round(S,3),
            formula=f"{formula_A}-{formula_B} x={x:.2f}",
        ))

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# ── Ternary screen ───────────────────────────────────────────────────────────
def screen_ternary(
    A: str, B: str, C: str,
    rh: float, temp: float,
    bg: tuple[float,float],
    bows: dict[str,float],
    dx: float = 0.1, dy: float = 0.1,
    alpha: float = 1.0, beta: float = 1.0,
) -> pd.DataFrame:

    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    lo, hi  = bg
    kT_exp  = 0.06
    env_pen = 1 + alpha * np.log1p(rh) + beta * (temp / 100)

    rows = []
    for x in np.arange(0, 1 + 1e-6, dx):
        for y in np.arange(0, 1 - x + 1e-6, dy):
            z = 1 - x - y

            Eg = (
                z * dA["band_gap"] + x * dB["band_gap"] + y * dC["band_gap"]
                - bows["AB"] * x * z - bows["AC"] * y * z - bows["BC"] * x * y
            )
            Eh = (
                z * dA["energy_above_hull"] + x * dB["energy_above_hull"] + y * dC["energy_above_hull"]
                + bows["AB"] * x * z + bows["AC"] * y * z + bows["BC"] * x * y
            )

            stab = np.exp(-max(Eh, 0) / kT_exp)
            gfit = score_band_gap(Eg, lo, hi)
            S    = stab * gfit / env_pen

            rows.append(dict(
                x=round(x,3), y=round(y,3), Eg=round(Eg,3),
                stability=round(stab,3), gap_score=round(gfit,3),
                score=round(S,3)
            ))

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# legacy alias used by app.py v9.x
_summary = fetch_mp_data
