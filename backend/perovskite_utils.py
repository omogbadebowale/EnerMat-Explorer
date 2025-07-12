"""
perovskite_backend.py  •  physics-corrected 2025-07-12
======================================================
High-throughput screening utilities for CsSn(Br,Cl,I)₃ alloys.
Implements all fixes discussed with Explore-GPT:

* +0.70/0.80/0.90 eV PBE→exp gap offsets for Br/Cl/I
* negative bowing for Br↔Cl   (default −0.15 eV)
* +25 meV·x(1−x) mixing enthalpy
* exponential stability  exp(−Ehull/0.05 eV)
* optical window 1.0–1.4 eV (strict)
* optional uniform RH/T penalty (doesn’t distort ranking)
* cubic perovskite filter in MP query

Author: <your name>
Licence: MIT
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition

# ───────────────────────────────
# 1.  Materials-Project set-up
# ───────────────────────────────
load_dotenv()
_API = os.getenv("MP_API_KEY")
if not _API or len(_API) != 32:
    raise RuntimeError("Set a valid 32-char MP_API_KEY in .env or Streamlit Secrets.")
mpr = MPRester(_API)

# ───────────────────────────────
# 2.  Empirical gap offsets (eV)
# ───────────────────────────────
GAP_CORR = {"I": 0.90, "Br": 0.70, "Cl": 0.80}

# ───────────────────────────────
# 3.  Helper functions
# ───────────────────────────────
def _mp_summary(formula: str) -> dict:
    """First cubic-phase summary with band_gap & energy_above_hull."""
    docs = mpr.summary.search(
        formula=formula,
        spacegroup_number={"$in": [221, 225]},   # Pm-3m / Fm-3m
        fields=["band_gap", "energy_above_hull"],
    )
    if not docs:
        raise ValueError(f"No cubic MP entry for {formula}")
    d = docs[0]
    return {"band_gap": d.band_gap, "Ehull": d.energy_above_hull}


def _correct_gap(formula: str, Eg_raw: float) -> float:
    hal = next(h for h in GAP_CORR if h in formula)
    return Eg_raw + GAP_CORR[hal]


def _optical_weight(Eg: float, lo: float = 1.0, hi: float = 1.4) -> float:
    """1.0 inside window, 0.0 outside."""
    return 1.0 if lo <= Eg <= hi else 0.0


def _env_penalty(RH: float, T: float) -> float:
    """Uniform environment penalty; composition-independent."""
    return 1.0 / (1 + RH / 100 + T / 100)

# ───────────────────────────────
# 4.  Binary screening
# ───────────────────────────────
def mix_abx3(
    A: str,
    B: str,
    rh: float = 50.0,
    temp: float = 25.0,
    dx: float = 0.05,
    bowing: float = -0.15,            # negative for Br↔Cl
) -> pd.DataFrame:
    """Return DataFrame of x, Eg, Ehull, score for A(1-x)B(x)."""

    dA = _mp_summary(A)
    dB = _mp_summary(B)
    Eg_A = _correct_gap(A, dA["band_gap"])
    Eg_B = _correct_gap(B, dB["band_gap"])

    out = []
    for x in np.arange(0.0, 1.0 + 1e-6, dx):
        # band gap
        Eg = (1 - x) * Eg_A + x * Eg_B - bowing * x * (1 - x)

        # formation energy with +25 meV x(1-x) penalty
        Ehull = (
            (1 - x) * dA["Ehull"]
            + x * dB["Ehull"]
            + 0.025 * x * (1 - x)
        )

        stab = np.exp(-Ehull / 0.05)          # 50 meV e-fold
        f_Eg = _optical_weight(Eg)
        score = stab * f_Eg * _env_penalty(rh, temp)

        out.append(
            dict(
                x=round(x, 3),
                Eg=round(Eg, 3),
                Ehull=round(Ehull, 3),
                score=round(score, 3),
            )
        )

    return pd.DataFrame(out).sort_values("score", ascending=False).reset_index(drop=True)

# ───────────────────────────────
# 5.  Ternary screening (MC)
# ───────────────────────────────
def screen_ternary(
    A: str,
    B: str,
    C: str,
    rh: float = 50.0,
    temp: float = 25.0,
    n_mc: int = 200,
    bows: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Sample A(1-x-y) B(x) C(y) with Latin-hypercube dirichlet.
    bows = {"AB":-0.15, "AC":+0.30, "BC":-0.15} default.
    """
    bows = bows or {"AB": -0.15, "AC": 0.30, "BC": -0.15}

    dA = _mp_summary(A)
    dB = _mp_summary(B)
    dC = _mp_summary(C)
    Eg_A = _correct_gap(A, dA["band_gap"])
    Eg_B = _correct_gap(B, dB["band_gap"])
    Eg_C = _correct_gap(C, dC["band_gap"])

    rng = np.random.default_rng(0)
    samples = rng.dirichlet((1, 1, 1), size=n_mc)  # z,x,y

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
            z * dA["Ehull"]
            + x * dB["Ehull"]
            + y * dC["Ehull"]
            + 0.025 * (x * y + x * z + y * z)   # simple mix penalty
        )

        stab = np.exp(-Ehull / 0.05)
        f_Eg = _optical_weight(Eg)
        score = stab * f_Eg * _env_penalty(rh, temp)

        rows.append(
            dict(
                x_B=round(x, 3),
                y_C=round(y, 3),
                Eg=round(Eg, 3),
                Ehull=round(Ehull, 3),
                score=round(score, 3),
            )
        )

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# ───────────────────────────────
# 6. quick demo if invoked directly
# ───────────────────────────────
if __name__ == "__main__":
    print("Binary CsSnBr3–CsSnCl3 (50 % RH, 25 °C)")
    print(mix_abx3("CsSnBr3", "CsSnCl3").head(), "\n")
    print("Random ternary CsSnBr3–CsSnCl3–CsSnI3 (n=200)")
    print(screen_ternary("CsSnBr3", "CsSnCl3", "CsSnI3").head())
