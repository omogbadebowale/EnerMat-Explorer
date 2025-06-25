"""
EnerMat Perovskite Explorer · backend  (stable 2025-06-25)

• Ground-state selector (lowest energy_above_hull)
• Gap = hse_gap if present, else PBE + halide-weighted scissor
• Boltzmann metastability: exp(−E_hull / kT_eff)
• Optional pair-specific bowing via backend/bowing.yaml
• Ternary rows now include stability + formula (no blank cells)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition
import streamlit as st  # only to fetch the key from st.secrets on Cloud

# ────────────────────────────── API key ──────────────────────────────────────
load_dotenv()
_API = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not _API or len(_API) != 32:
    raise RuntimeError("🛑  Please set a valid 32-character Materials-Project key.")
mpr = MPRester(_API)

# ─────────────────── presets & ionic radii ───────────────────────────────────
END_MEMBERS: List[str] = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

IONIC_RADII: Dict[str, float] = {
    "Cs": 1.88,
    "Rb": 1.72,
    "MA": 2.17,
    "FA": 2.53,
    "Pb": 1.19,
    "Sn": 1.18,
    "I": 2.20,
    "Br": 1.96,
    "Cl": 1.81,
}

SCISSOR = {"I": 0.60, "Br": 0.90, "Cl": 1.30}    # eV added to PBE gaps
K_T_EFF = 0.06                                    # eV  (≈700 K)
DEFAULT_BOW = 0.30                                # eV

# ─────────────────── optional bowing-table loader ────────────────────────────
def _load_bowing(path: Path | str = Path(__file__).with_name("bowing.yaml")) -> Dict[str, float]:
    if Path(path).is_file():
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}

BOW_TABLE: Dict[str, float] = _load_bowing()

# ───────────────────────── helper functions ──────────────────────────────────
def _scissor(formula: str) -> float:
    comp = Composition(formula).get_el_amt_dict()
    return sum(comp.get(X, 0) * SCISSOR[X] for X in SCISSOR) / 3.0

def _bow(pair_key: str, fallback: float = DEFAULT_BOW) -> float:
    return float(BOW_TABLE.get(pair_key, fallback))

# ────────────────── Materials-Project pull with corrections ──────────────────
def fetch_mp_data(formula: str, fields: List[str]) -> Dict | None:
    extra = {"band_gap", "energy_above_hull"}        # guaranteed fields
    docs = mpr.summary.search(formula=formula, fields=list(set(fields) | extra))
    if not docs:
        return None
    entry = min(docs, key=lambda d: d.energy_above_hull)            # ✅ lowest-hull

    Eg = getattr(entry, "hse_gap", None)
    if Eg is None:
        Eg = entry.band_gap + _scissor(formula)

    out = {f: getattr(entry, f, None) for f in fields if hasattr(entry, f)}
    out["Eg"] = Eg
    return out

def score_band_gap(Eg: float, lo: float, hi: float) -> float:
    if Eg < lo:
        return max(0.0, 1 - (lo - Eg) / (hi - lo))
    if Eg > hi:
        return max(0.0, 1 - (Eg - hi) / (hi - lo))
    return 1.0

# ───────────────────────────── binary screen ─────────────────────────────────
def mix_abx3(
    formula_A: str,
    formula_B: str,
    rh: float,
    temp: float,
    bg_window: Tuple[float, float],
    bowing: float = DEFAULT_BOW,
    dx: float = 0.05,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> pd.DataFrame:
    lo, hi = bg_window
    dA = fetch_mp_data(formula_A, ["energy_above_hull"])
    dB = fetch_mp_data(formula_B, ["energy_above_hull"])
    if not dA or not dB:
        return pd.DataFrame()

    compA = Composition(formula_A)
    X_site = next(e.symbol for e in compA.elements if e.symbol in SCISSOR)
    rA = IONIC_RADII[next(e.symbol for e in compA.elements if e.symbol in IONIC_RADII)]
    rB = IONIC_RADII[next(e.symbol for e in compA.elements if e.symbol in {"Pb", "Sn"})]
    rX = IONIC_RADII[X_site]

    rows = []
    for x in np.arange(0.0, 1.0 + 1e-6, dx):
        b_eff = _bow("AB", bowing)
        Eg = (1 - x) * dA["Eg"] + x * dB["Eg"] - b_eff * x * (1 - x)
        hull = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]

        stability = np.exp(-hull / K_T_EFF)
        gap_score = score_band_gap(Eg, lo, hi)

        t = (rA + rX) / (np.sqrt(2) * (rB + rX))
        mu = rB / rX
        form_score = np.exp(-0.5 * ((t - 0.90) / 0.07) ** 2) * np.exp(
            -0.5 * ((mu - 0.50) / 0.07) ** 2
        )

        env_pen = 1 + alpha * (rh / 100) + beta * (temp / 100)
        score = form_score * stability * gap_score / env_pen

        rows.append(
            dict(
                x=round(x, 3),
                Eg=round(Eg, 3),
                stability=round(stability, 3),
                score=round(score, 3),
                formula=f"{formula_A}-{formula_B} x={x:.2f}",
            )
        )

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

# ───────────────────────────── ternary screen ────────────────────────────────
def screen_ternary(
    A: str,
    B: str,
    C: str,
    rh: float,
    temp: float,
    bg: Tuple[float, float],
    bows: Dict[str, float],
    dx: float = 0.1,
    dy: float = 0.1,
    n_mc: int = 200,
) -> pd.DataFrame:
    dA = fetch_mp_data(A, ["energy_above_hull"])
    dB = fetch_mp_data(B, ["energy_above_hull"])
    dC = fetch_mp_data(C, ["energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    lo, hi = bg
    rows = []
    for x in np.arange(0.0, 1.0 + 1e-6, dx):
        for y in np.arange(0.0, 1.0 - x + 1e-6, dy):
            z = 1 - x - y
            bAB = _bow("AB", bows.get("AB", DEFAULT_BOW))
            bAC = _bow("AC", bows.get("AC", DEFAULT_BOW))
            bBC = _bow("BC", bows.get("BC", DEFAULT_BOW))

            Eg = (
                z * dA["Eg"]
                + x * dB["Eg"]
                + y * dC["Eg"]
                - bAB * x * z
                - bAC * y * z
                - bBC * x * y
            )
            hull = (
                z * dA["energy_above_hull"]
                + x * dB["energy_above_hull"]
                + y * dC["energy_above_hull"]
            )

            stability = np.exp(-hull / K_T_EFF)
            gap_score = score_band_gap(Eg, lo, hi)
            env_pen = 1 + (rh / 100) + (temp / 100)
            score = stability * gap_score / env_pen

            rows.append(
                dict(
                    x=round(x, 3),
                    y=round(y, 3),
                    Eg=round(Eg, 3),
                    stability=round(stability, 3),      # ← ensures no blank
                    score=round(score, 3),
                    formula=f"{A}-{B}-{C} x={x:.2f} y={y:.2f}",
                )
            )

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

# legacy alias
_summary = fetch_mp_data
