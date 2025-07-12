"""
perovskite_backend.py   •   physics-corrected 2025-07-12
========================================================
Binary and ternary screening utilities for CsSn(Br,Cl,I)₃ alloys.
All fixes discussed with Explore-GPT are included.

Author: Your Name
Licence: MIT
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition

# ──────────────────────────────────────────────────────────────
# 1.  Materials-Project API setup
# ──────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("Please set a valid 32-character MP_API_KEY in .env")
mpr = MPRester(API_KEY)

# ──────────────────────────────────────────────────────────────
# 2.  Empirical PBE → exp band-gap offsets  (eV)
# ──────────────────────────────────────────────────────────────
HALIDE_GAP_CORR = {"I": 0.90, "Br": 0.70, "Cl": 0.80}

# ──────────────────────────────────────────────────────────────
# 3.  Ionic radii (Å) for optional tolerance-factor use
# ──────────────────────────────────────────────────────────────
IONIC_RADII = {"Cs": 1.88, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81}

# ──────────────────────────────────────────────────────────────
# 4.  Helper functions
# ──────────────────────────────────────────────────────────────
def _mp_summary(formula: str) -> dict:
    """
    Return summary dict for the first *cubic* perovskite entry
    of a given formula; fields: band_gap, energy_above_hull.
    """
    docs = mpr.summary.search(
        formula=formula,
        spacegroup_number={"$in": [221, 225]},  # Pm-3m or Fm-3m
        fields=["band_gap", "energy_above_hull"],
    )
    if not docs:
        raise ValueError(f"No cubic MP entry found for {formula}")
    entry = docs[0]
    return {
        "band_gap": entry.band_gap,
        "energy_above_hull": entry.energy_above_hull,
    }


def _correct_gap(formula: str, raw_gap: float) -> float:
    """Add halide-specific empirical offset to PBE gap."""
    hal = next(h for h in HALIDE_GAP_CORR if h in formula)
    return raw_gap + HALIDE_GAP_CORR[hal]


def _optical_weight(Eg: float, lo: float = 1.0, hi: float = 1.4) -> float:
    """Return 1.0 inside [lo, hi]; 0.0 outside that window."""
    return 1.0 if lo <= Eg <= hi else 0.0


def _env_penalty(RH: float, T: float) -> float:
    """Optional uniform RH/T penalty (composition-independent)."""
    return 1.0 / (1 + RH / 100 + T / 100)


# ──────────────────────────────────────────────────────────────
# 5.  Binary alloy screening  A(1-x)B(x)
# ──────────────────────────────────────────────────────────────
def mix_abx3(
    formula_A: str,
    formula_B: str,
    rh: float = 50.0,
    temp: float = 25.0,
    bowing: float = -0.15,          # **negative** for Br→Cl
    dx: float = 0.05,
) -> pd.DataFrame:
    """
    Screen binary join and return DataFrame with columns:
    x, Eg, Ehull, stability, f_Eg, score
    """
    dA = _mp_summary(formula_A)
    dB = _mp_summary(formula_B)

    Eg_A = _correct_gap(formula_A, dA["band_gap"])
    Eg_B = _correct_gap(formula_B, dB["band_gap"])

    rows = []
    for x in np.arange(0.0, 1.0 + 1e-6, dx):
        # band gap with bowing
        Eg = (1 - x) * Eg_A + x * Eg_B - bowing * x * (1 - x)

        # formation energy (include +25 meV x(1-x) mixing penalty)
        Ehull = (
            (1 - x) * dA["energy_above_hull"]
            + x * dB["energy_above_hull"]
            + 0.025 * x * (1 - x)
        )

        stability = np.exp(-Ehull / 0.05)         # 50 meV e-fold
        f_Eg = _optical_weight(Eg)
        score = stability * f_Eg * _env_penalty(rh, temp)

        rows.append(
            {
                "x": round(x, 3),
                "Eg": round(Eg, 3),
                "Ehull": round(Ehull, 3),
                "stability": round(stability, 3),
                "f_Eg": f_Eg,
                "score": round(score, 3),
            }
        )

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# 6.  Ternary screening  A(1-x-y) B(x) C(y)
#     (Latin-hypercube or grid sampling)
# ──────────────────────────────────────────────────────────────
def screen_ternary(
    A: str,
    B: str,
    C: str,
    rh: float = 50.0,
    temp: float = 25.0,
    bows: dict[str, float] | None = None,   # pairwise bowing dict
    n_mc: int = 200,
) -> pd.DataFrame:
    """
    Monte-Carlo sample ternary triangle and return DataFrame with:
    x_B, y_C, Eg, Ehull, score
    """
    bows = bows or {"AB": -0.15, "AC": +0.30, "BC": -0.15}  # example values

    dA = _mp_summary(A)
    dB = _mp_summary(B)
    dC = _mp_summary(C)

    Eg_A = _correct_gap(A, dA["band_gap"])
    Eg_B = _correct_gap(B, dB["band_gap"])
    Eg_C = _correct_gap(C, dC["band_gap"])

    rng = np.random.default_rng(0)
    samples = rng.dirichlet((1, 1, 1), size=n_mc)  # rows (z,x,y)

    rows = []
    for z, x, y in samples:
        Eg = (
            z * Eg_A
            + x * Eg_B
            + y * Eg_C
            - bows["AB"] * x * z
            - bows["AC"] * y * z
            - bows["BC"] * x * y
        )

        Ehull = (
            z * dA["energy_above_hull"]
            + x * dB["energy_above_hull"]
            + y * dC["energy_above_hull"]
            + 0.025 * (x * z + y * z + x * y)  # simple mixing penalty
        )

        stability = np.exp(-Ehull / 0.05)
        f_Eg = _optical_weight(Eg)
        score = stability * f_Eg * _env_penalty(rh, temp)

        rows.append(
            {
                "x_B": round(x, 3),
                "y_C": round(y, 3),
                "Eg": round(Eg, 3),
                "Ehull": round(Ehull, 3),
                "score": round(score, 3),
            }
        )

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# 7.  If run as script, quick demo
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = mix_abx3("CsSnBr3", "CsSnCl3", bowing=-0.15)
    print(df.head())
