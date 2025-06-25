# backend/perovskite_utils.py
"""
Utility layer for EnerMat Explorer
––––––––––––––––––––––––––––––––––
•  Lowest-enthalpy polymorph is now selected explicitly.
•  Stability weight uses an exponential (exp(–Ehull/0.06 eV)).
•  gap_score is returned in every row (binary + ternary).
•  score_band_gap() tapers to zero 0.6 eV beyond the user-set window.
"""

from __future__ import annotations
import os, numpy as np, pandas as pd
from dotenv import load_dotenv
load_dotenv()

# optional fallback when running on Streamlit Cloud
import streamlit as st

from mp_api.client import MPRester
from pymatgen.core import Composition

API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑  Set a valid 32-character MP_API_KEY.")
mpr = MPRester(API_KEY)

# ─── Presets & constants ─────────────────────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18,
    "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

K_BOLTZ = 0.06  # eV – “effective temperature” for metastability weight


# ─── Helper wrappers ──────────────────────────────────────────────────────────
def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    """Return the requested fields for the *ground-state* polymorph."""
    docs = mpr.summary.search(formula=formula, fields=fields + ["energy_above_hull"])
    if not docs:
        return None
    entry = min(docs, key=lambda d: d.energy_above_hull)  # lowest-Ehull
    return {f: getattr(entry, f) for f in fields if hasattr(entry, f)}


def score_band_gap(Eg: float, lo: float, hi: float) -> float:
    """
    1   inside window  [lo, hi]
    ↓   linear taper to zero 0.6 eV beyond each edge
    0   for Eg < lo-0.6 or Eg > hi+0.6
    """
    span = hi - lo
    span_ext = 0.6  # width of taper (eV)
    if lo <= Eg <= hi:
        return 1.0
    if Eg < lo:
        return max(0.0, 1 - (lo - Eg) / span_ext)
    return max(0.0, 1 - (Eg - hi) / span_ext)


# ─── Binary screen ───────────────────────────────────────────────────────────
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
        hull = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        stability = np.exp(-max(hull, 0) / K_BOLTZ)

        gap_score = score_band_gap(Eg, lo, hi)

        t = (rA + rX) / (np.sqrt(2) * (rB + rX))
        mu = rB / rX
        form_score = np.exp(-0.5 * ((t - 0.90) / 0.07) ** 2) \
                   * np.exp(-0.5 * ((mu - 0.50) / 0.07) ** 2)

        env_pen = 1 + alpha * (rh / 100) + beta * (temp / 100)
        score = form_score * stability * gap_score / env_pen

        rows.append(
            dict(
                x=round(x, 3),
                Eg=round(Eg, 3),
                stability=round(stability, 3),
                gap_score=round(gap_score, 3),
                score=round(score, 3),
                formula=f"{formula_A}-{formula_B} x={x:.2f}",
            )
        )

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


# ─── Ternary screen ──────────────────────────────────────────────────────────
def screen_ternary(
    A: str,
    B: str,
    C: str,
    rh: float,
    temp: float,
    bg: tuple[float, float],
    bows: dict[str, float],
    dx: float = 0.1,
    dy: float = 0.1,
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
                z * dA["band_gap"] + x * dB["band_gap"] + y * dC["band_gap"]
                - bows["AB"] * x * z - bows["AC"] * y * z - bows["BC"] * x * y
            )

            Eh = (
                z * dA["energy_above_hull"] + x * dB["energy_above_hull"] + y * dC["energy_above_hull"]
                + bows["AB"] * x * z + bows["AC"] * y * z + bows["BC"] * x * y
            )

            stability = np.exp(-max(Eh, 0) / K_BOLTZ)
            gap_score = score_band_gap(Eg, lo, hi)
            score = stability * gap_score  # env_pen = 1 (for now)

            rows.append(
                dict(
                    x=round(x, 3),
                    y=round(y, 3),
                    Eg=round(Eg, 3),
                    stability=round(stability, 3),
                    gap_score=round(gap_score, 3),
                    score=round(score, 3),
                )
            )

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


# -- legacy alias ─────────────────────────────────────────────────────────────
_summary = fetch_mp_data
