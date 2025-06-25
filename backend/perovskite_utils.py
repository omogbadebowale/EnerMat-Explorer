# EnerMat-Explorer  •  Patched 2025-06-25
# ---------------------------------------------------------------
# Works stand-alone or as importable backend for Streamlit.

import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st        # falls back to dummy st when not on Cloud
import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Composition

# ── API key ────────────────────────────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 Set a valid 32-character Materials-Project API key")
mpr = MPRester(API_KEY)

# ── Global physics knobs ───────────────────────────────────────────────
SCISSOR_SHIFT = 0.90          # eV  (PBE  →  HSE06+SOC)
TAU_STABILITY = 0.05          # eV  for  exp(-Ehull/τ)
OPT_WINDOW    = (1.0, 1.4)    # eV  default single-junction; change to (1.6,2.1) for tandem

# ── Ionic radii (Å) for Goldschmidt part (unchanged) ───────────────────
IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81,
}


# ╭─────────────────────────── MP helpers ─────────────────────────────╮
def fetch_mp_data(formula: str, fields: list[str] | None = None) -> dict:
    """
    Return dict of lowest-Ehull polymorph summary fields.
    Cached per formula for speed.
    """
    if "_cache" not in fetch_mp_data.__dict__:
        fetch_mp_data._cache = {}
    cache = fetch_mp_data._cache

    if formula in cache:
        summary = cache[formula]
    else:
        docs = mpr.summary.search(formulas=[formula],
                                  fields=["formula_pretty", "band_gap",
                                          "energy_above_hull", "is_gap_direct"])
        if not docs:
            raise ValueError(f"{formula} not found on Materials Project")
        summary = docs[0]
        cache[formula] = summary

    if fields:
        return {f: getattr(summary, f) for f in fields}
    return summary
# ╰────────────────────────────────────────────────────────────────────╯


# ╭──────────────────── Physics utility functions ──────────────────────╮
def corrected_gap(formula: str) -> float:
    """HSE-quality gap via rigid scissor shift."""
    pbe_gap = fetch_mp_data(formula, ["band_gap"])["band_gap"]
    return pbe_gap + SCISSOR_SHIFT


def stability_weight(ehull: float, tau: float = TAU_STABILITY) -> float:
    """Smooth thermodynamic factor  exp(-Ehull/τ)."""
    return float(np.exp(-max(ehull, 0.0) / tau))


def optical_weight(eg: float,
                   window: tuple[float, float] = OPT_WINDOW,
                   margin: float = 0.20) -> float:
    """
    Trapezoid: 1 inside [lo,hi]; linear fall-off over ±margin; 0 outside.
    """
    lo, hi = window
    if lo <= eg <= hi:
        return 1.0
    if eg < lo - margin or eg > hi + margin:
        return 0.0
    if eg < lo:
        return (eg - (lo - margin)) / margin
    return ((hi + margin) - eg) / margin
# ╰─────────────────────────────────────────────────────────────────────╯


# ╭──────────────────────── Binary alloy screen ────────────────────────╮
def mix_abx3(formula_A: str,
             formula_B: str,
             rh: float,
             temp: float,
             bg_window: tuple[float, float] = OPT_WINDOW,
             bowing: float = 0.0,
             dx: float = 0.05,
             alpha: float = 1.0,
             beta:  float = 1.0) -> pd.DataFrame:
    """
    Screen binary A–B perovskite: returns DataFrame sorted by composite score.
    """
    lo, hi = bg_window
    dA = fetch_mp_data(formula_A, ["energy_above_hull"])
    dB = fetch_mp_data(formula_B, ["energy_above_hull"])

    # Ionic radii for formability filter (unchanged)
    comp = Composition(formula_A)
    A_site = next(e.symbol for e in comp.elements if e.symbol in IONIC_RADII)
    B_site = next(e.symbol for e in comp.elements if e.symbol in {"Pb", "Sn"})
    X_site = next(e.symbol for e in comp.elements if e.symbol in {"I", "Br", "Cl"})
    rA, rB, rX = IONIC_RADII[A_site], IONIC_RADII[B_site], IONIC_RADII[X_site]

    env_pen = 1.0 / (1 + alpha * rh / 100 + beta * temp / 100)
    rows = []
    xs = np.arange(0.0, 1.0 + 1e-9, dx)
    for x in xs:
        # band gap with scissor + bowing
        Eg = ((1 - x) * corrected_gap(formula_A)
              + x * corrected_gap(formula_B)
              - bowing * x * (1 - x))
        stability = stability_weight((1 - x) * dA["energy_above_hull"]
                                     + x * dB["energy_above_hull"])
        gap_score = optical_weight(Eg, (lo, hi))
        t = (rA + rX) / (np.sqrt(2) * (rB + rX))
        mu = rB / rX
        form_score = np.exp(-0.5 * ((t - 0.90) / 0.07) ** 2) \
                   * np.exp(-0.5 * ((mu - 0.50) / 0.07) ** 2)
        score = form_score * stability * gap_score * env_pen
        rows.append(dict(x=round(x, 3), Eg=round(Eg, 3),
                         stability=round(stability, 3),
                         score=round(score, 3)))
    return (pd.DataFrame(rows)
              .sort_values("score", ascending=False)
              .reset_index(drop=True))
# ╰─────────────────────────────────────────────────────────────────────╯


# ╭───────────────────────── Ternary screen ────────────────────────────╮
def screen_ternary(A: str,
                   B: str,
                   C: str,
                   rh: float,
                   temp: float,
                   bg_window: tuple[float, float] = OPT_WINDOW,
                   bows: dict[str, float] | None = None,
                   dx: float = 0.1,
                   dy: float = 0.1) -> pd.DataFrame:
    """
    Simple grid (dx,dy) over the A–B–C triangle.  bow dict keys: "AB","AC","BC".
    """
    if bows is None:
        bows = {"AB": 0.0, "AC": 0.0, "BC": 0.0}
    dA = fetch_mp_data(A, ["energy_above_hull"])
    dB = fetch_mp_data(B, ["energy_above_hull"])
    dC = fetch_mp_data(C, ["energy_above_hull"])

    lo, hi = bg_window
    env_pen = 1.0 / (1 + rh / 100 + temp / 100)   # fast heuristic as before
    rows = []
    for x in np.arange(0, 1 + 1e-9, dx):
        for y in np.arange(0, 1 - x + 1e-9, dy):
            z = 1 - x - y
            Eg = ((1 - x - y) * corrected_gap(A)
                  + x * corrected_gap(B)
                  + y * corrected_gap(C)
                  - bows["AB"] * x * z
                  - bows["AC"] * y * z
                  - bows["BC"] * x * y)
            Eh = ((1 - x - y) * dA["energy_above_hull"]
                  + x * dB["energy_above_hull"]
                  + y * dC["energy_above_hull"])
            stability = stability_weight(Eh)
            score = stability * optical_weight(Eg, (lo, hi)) * env_pen
            rows.append(dict(x=round(x,3), y=round(y,3),
                             Eg=round(Eg,3), Ehull=round(Eh,3),
                             score=round(score,3)))
    return (pd.DataFrame(rows)
              .sort_values("score", ascending=False)
              .reset_index(drop=True))
# ╰─────────────────────────────────────────────────────────────────────╯


# Backwards-compat alias
_summary = fetch_mp_data
