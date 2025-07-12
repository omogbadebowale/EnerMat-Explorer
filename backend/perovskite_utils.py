
import os
from dotenv import load_dotenv
load_dotenv()

# for secrets fallback on Streamlit Cloud
import streamlit as st

import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Composition

# ── Load Materials Project API key ─────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError(
        "🛑 Please set MP_API_KEY to your 32-character Materials Project API key"
    )
mpr = MPRester(API_KEY)

# ── Supported end-members ──────────────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

# ── Ionic radii (Å) for Goldschmidt tolerance ──────────────────────────
IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81,
}


def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    """Return a dict of the first matching entry's requested fields, or None."""
    docs = mpr.summary.search(formula=formula)
    if not docs:
        return None
    entry = docs[0]
    out: dict = {}
    for f in fields:
        if hasattr(entry, f):
            out[f] = getattr(entry, f)
    return out if out else None


def score_band_gap(bg: float, lo: float, hi: float) -> float:
    """How close bg is to the [lo, hi] window."""
    if bg < lo:
        return max(0.0, 1 - (lo - bg) / (hi - lo))
    if bg > hi:
        return max(0.0, 1 - (bg - hi) / (hi - lo))
    return 1.0


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
    """Binary screening A–B across x from 0→1."""
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
        hull = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        stability = max(0.0, 1 - hull)
        gap_score = score_band_gap(Eg, lo, hi)
        t = (rA + rX) / (np.sqrt(2) * (rB + rX))
        mu = rB / rX
        form_score = np.exp(-0.5 * ((t - 0.90) / 0.07) ** 2) * np.exp(-0.5 * ((mu - 0.50) / 0.07) ** 2)
        env_pen = 1 + alpha * (rh / 100) + beta * (temp / 100)
        score = form_score * stability * gap_score / env_pen
        rows.append({
            "x": round(x, 3),
            "Eg": round(Eg, 3),
            "stability": round(stability, 3),
            "score": round(score, 3),
            "formula": f"{formula_A}-{formula_B} x={x:.2f}",
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


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
    """Ternary screening A–B–C over x,y fractions."""
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
            Eg = (
                (1 - x - y) * dA["band_gap"] + x * dB["band_gap"] + y * dC["band_gap"]
                - bows["AB"] * x * (1 - x - y)
                - bows["AC"] * y * (1 - x - y)
                - bows["BC"] * x * y
            )
            Eh_val = (
                (1 - x - y) * dA["energy_above_hull"] + x * dB["energy_above_hull"] + y * dC["energy_above_hull"]
                + bows["AB"] * x * (1 - x - y)
                + bows["AC"] * y * (1 - x - y)
                + bows["BC"] * x * y
            )
            stability = np.exp(-max(Eh_val, 0) / 0.1)
            gap_score = score_band_gap(Eg, lo, hi)
            score = stability * gap_score
            rows.append({"x": round(x,3), "y": round(y,3), "Eg": round(Eg,3), "score": round(score,3)})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# alias for backwards compatibility
_summary = fetch_mp_data
