"""
EnerMat Perovskite Explorer – backend/perovskite_utils.py
Fully patched for demo‑proof reliability (2025‑07‑12)
* Single source‑of‑truth gap corrections (CALIBRATED_GAPS + GAP_OFFSET)
* Strict 0/1 optical merit
* Restores Ehull column for every candidate
* Public API unchanged:  mix_abx3(), screen_ternary(), fetch_mp_data()
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
load_dotenv()

# fallback when running on Streamlit Cloud
import streamlit as st
import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Composition

# ────────────────────────────────────────────────────────────────────
# Materials Project API key
# -------------------------------------------------------------------
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 Please set a valid 32‑character MP_API_KEY")

mpr = MPRester(API_KEY)

# ────────────────────────────────────────────────────────────────────
# Global constants
# -------------------------------------------------------------------
END_MEMBERS: list[str] = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

# Experimental end‑member gaps (eV) – authoritative where available
CALIBRATED_GAPS: dict[str, float] = {
    "CsSnBr3": 1.79,   # Weller et al., 2015
    "CsSnCl3": 2.83,   # Sun et al., 2021
    "CsSnI3":  1.30,   # Hao et al., 2014
    "CsPbBr3": 2.30,
    "CsPbI3":  1.73,
}

# Generic PBE → experiment scissor offsets by halide
GAP_OFFSET = {"I": 0.90, "Br": 0.70, "Cl": 0.80}

# Shannon ionic radii (Å) – used for Goldschmidt factors
IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18,
    "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

# ────────────────────────────────────────────────────────────────────
# Helpers
# -------------------------------------------------------------------
@lru_cache(maxsize=None)
def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    """Return dict of requested fields for the first MP entry (cached)."""
    docs = mpr.summary.search(formula=formula, fields=fields)
    if not docs:
        return None
    entry = docs[0]
    return {f: getattr(entry, f) for f in fields if hasattr(entry, f)}


def _apply_gap_corrections(formula: str, doc: dict) -> None:
    """Mutate *doc* in‑place so that 'band_gap' is physically calibrated."""
    # 1) exact match (trusted experimental number)
    if formula in CALIBRATED_GAPS:
        doc["band_gap"] = CALIBRATED_GAPS[formula]
        return
    # 2) otherwise apply halide‑specific scissor to PBE gap
    hal = next(h for h in ("I", "Br", "Cl") if h in formula)
    doc["band_gap"] += GAP_OFFSET[hal]


def _optical_weight(Eg: float, lo: float, hi: float) -> float:
    """Strict 0 / 1 merit inside the target window."""
    return 1.0 if lo <= Eg <= hi else 0.0

# ────────────────────────────────────────────────────────────────────
# Screening routines
# -------------------------------------------------------------------

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
    """Binary A–B alloy screen across composition *x* in [0,1]."""
    lo, hi = bg_window

    dA = fetch_mp_data(formula_A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(formula_B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    _apply_gap_corrections(formula_A, dA)
    _apply_gap_corrections(formula_B, dB)

    # Goldschmidt radii for A‑site, B‑site, X‑site (from formula_A prototype)
    comp = Composition(formula_A)
    A_site = next(e.symbol for e in comp.elements if e.symbol in IONIC_RADII)
    B_site = next(e.symbol for e in comp.elements if e.symbol in {"Pb", "Sn"})
    X_site = next(e.symbol for e in comp.elements if e.symbol in {"I", "Br", "Cl"})
    rA, rB, rX = IONIC_RADII[A_site], IONIC_RADII[B_site], IONIC_RADII[X_site]

    rows = []
    for x in np.arange(0, 1 + 1e-6, dx):
        # Vegard + quadratic bowing
        Eg = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bowing * x * (1 - x)
        Ehull = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]

        gap_score = _optical_weight(Eg, lo, hi)
        stability = max(0.0, 1 - Ehull)  # simple linear proxy

        # formability (still uses fixed rX; next patch will interpolate)
        t = (rA + rX) / (np.sqrt(2) * (rB + rX))
        mu = rB / rX
        form_score = np.exp(-0.5 * ((t - 0.90) / 0.07) ** 2) * np.exp(
            -0.5 * ((mu - 0.50) / 0.07) ** 2
        )

        env_pen = 1 + alpha * rh / 100 + beta * temp / 100
        score = form_score * stability * gap_score / env_pen

        rows.append(
            {
                "x": round(x, 3),
                "Eg": round(Eg, 3),
                "Ehull": round(Ehull, 4),
                "score": round(score, 3),
                "formula": f"{formula_A}-{formula_B} x={x:.2f}",
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


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
    n_mc: int = 200,
) -> pd.DataFrame:
    """Monte‑Carlo ternary screen over (x,y) on the A–B–C simplex."""

    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    for f, d in ((A, dA), (B, dB), (C, dC)):
        _apply_gap_corrections(f, d)

    lo, hi = bg
    rows = []
    rng = np.random.default_rng(42)
    for _ in range(n_mc):
        x, y = rng.random(2)
        if x + y > 1:
            x, y = 1 - x, 1 - y
        z = 1 - x - y

        Eg = (
            z * dA["band_gap"]
            + x * dB["band_gap"]
            + y * dC["band_gap"]
            - bows["AB"] * x * z
            - bows["AC"] * y * z
            - bows["BC"] * x * y
        )
        Ehull = (
            z * dA["energy_above_hull"]
            + x * dB["energy_above_hull"]
            + y * dC["energy_above_hull"]
            + bows["AB"] * x * z
            + bows["AC"] * y * z
            + bows["BC"] * x * y
        )
        gap_score = _optical_weight(Eg, lo, hi)
        stability = np.exp(-max(Ehull, 0) / 0.1)
        score = stability * gap_score

        rows.append(
            {
                "x": round(x, 3),
                "y": round(y, 3),
                "Eg": round(Eg, 3),
                "Ehull": round(Ehull, 4),
                "score": round(score, 3),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

# --------------------------------------------------------------------
# legacy alias
_summary = fetch_mp_data
