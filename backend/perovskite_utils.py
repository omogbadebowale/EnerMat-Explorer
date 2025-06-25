# backend/perovskite_utils.py
# ─────────────────────────────────────────────────────────────────────────────
"""
Utility layer for EnerMat Perovskite Explorer -- v9.8
• lowest-energy Materials-Project entry is selected (not the first hit)
• stability weight exp(-Ehull / 0.06 eV)
• gap_score column exposed so the Streamlit app can list / download it
"""
from __future__ import annotations
import os, math
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition

# ── API key ──────────────────────────────────────────────────────────────────
load_dotenv()
import streamlit as st
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑  MP_API_KEY missing or invalid (32 chars expected).")
mpr = MPRester(API_KEY)

# ── Constants & lookup tables ────────────────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

IONIC_RADII = {           # Shannon / Pyykkö Å
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def fetch_mp_data(formula: str, fields: list[str]) -> dict[str, Any] | None:
    """Return lowest-Ehull entry’s selected fields, or None if absent."""
    docs = mpr.summary.search(formula=formula)
    if not docs:
        return None
    entry = min(docs, key=lambda d: d.energy_above_hull or 1e9)
    return {f: getattr(entry, f) for f in fields if hasattr(entry, f)}


def score_band_gap(bg: float, lo: float, hi: float) -> float:
    """Cosine taper outside the [lo, hi] window (1 inside, 0 well outside)."""
    if lo <= bg <= hi:
        return 1.0
    d = abs(bg - (hi if bg > hi else lo))
    w = hi - lo
    return max(0.0, 0.5 * (1 + math.cos(min(d, w) / w * math.pi)))


# ── Binary screen ────────────────────────────────────────────────────────────
def mix_abx3(
    formula_A: str,
    formula_B: str,
    rh: float,
    temp: float,
    bg_window: tuple[float, float],
    bowing: float = 0.30,
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
    rA, rB, rX = IONIC_RADII[A_site], IONIC_RADII[B_site], IONIC_RADII[X_site]

    rows: list[dict] = []
    for x in np.arange(0, 1 + 1e-6, dx):
        Eg   = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bowing * x * (1 - x)
        Eh   = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        stability   = math.exp(-max(Eh, 0) / 0.06)           # exp-weight
        gap_score   = score_band_gap(Eg, lo, hi)

        # geometric formability penalty (t, μ Gaussian)
        t  = (rA + rX) / (math.sqrt(2) * (rB + rX))
        μ  = rB / rX
        form_score = math.exp(-0.5 * ((t  - 0.90) / 0.07) ** 2) * \
                     math.exp(-0.5 * ((μ  - 0.50) / 0.07) ** 2)

        env_pen = 1 + alpha * (rh / 100) + beta * (temp / 100)
        score   = form_score * stability * gap_score / env_pen

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

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

# ── Ternary screen ───────────────────────────────────────────────────────────
def screen_ternary(
    A: str, B: str, C: str,
    rh: float, temp: float,
    bg: tuple[float, float],
    bows: dict[str, float],
    dx: float = 0.10, dy: float = 0.10,
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
            z  = 1 - x - y
            Eg = (
                z * dA["band_gap"] + x * dB["band_gap"] + y * dC["band_gap"]
                - bows["AB"] * x * z - bows["AC"] * y * z - bows["BC"] * x * y
            )
            Eh = (
                z * dA["energy_above_hull"] + x * dB["energy_above_hull"] + y * dC["energy_above_hull"]
                + bows["AB"] * x * z + bows["AC"] * y * z + bows["BC"] * x * y
            )
            stability = math.exp(-max(Eh, 0) / 0.06)
            gap_score = score_band_gap(Eg, lo, hi)
            score     = stability * gap_score

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

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

# backwards-compat (app originally imported _summary)
_summary = fetch_mp_data
