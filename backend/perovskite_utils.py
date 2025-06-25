# ──────────────────────────────────────────────────────────────────────────────
#  backend/perovskite_utils.py   (patched 2025-06-25)
# ──────────────────────────────────────────────────────────────────────────────
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition

# ─── API key ──────────────────────────────────────────────────────────────────
load_dotenv()
import streamlit as st

API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError(
        "🛑 Please set MP_API_KEY to your 32-character Materials Project key."
    )
mpr = MPRester(API_KEY)

# ─── End-member presets & ionic radii ─────────────────────────────────────────
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

# ─── Constants for quick “scissor” correction of PBE gaps (eV) ────────────────
SCISSOR = {"I": 0.60, "Br": 0.90, "Cl": 1.30}  # add to PBE gap

# ─── Boltzmann metastability parameter (effective synthesis temperature) ─────
K_T_EFF = 0.06  # eV  (≈ 700 K)

# ─── Optional pair-specific bowing table  (backend/bowing.yaml)  --------------
def _load_bowing(path: Path | str = Path(__file__).with_name("bowing.yaml")) -> Dict[str, float]:
    if not Path(path).is_file():
        return {}  # fall back on default 0.30 eV everywhere
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}

BOW_TABLE: Dict[str, float] = _load_bowing()


# ─── Helper: fetch lowest-energy MP entry & corrected gap ─────────────────────
def fetch_mp_data(formula: str, fields: List[str]) -> Dict | None:
    """
    Return a dict of requested fields **for the lowest E_hull entry**.
    Adds 'Eg' (gap with HSE or scissor-corrected PBE) automatically.
    """
    docs = mpr.summary.search(
        formula=formula,
        fields=list(set(fields) | {"band_gap", "hse_gap", "energy_above_hull"}),
    )
    if not docs:
        return None

    entry = min(docs, key=lambda d: d.energy_above_hull)  # ground state
    raw_gap = getattr(entry, "hse_gap", None)  # may be None
    if raw_gap is None:
        raw_gap = entry.band_gap
        # apply halide-weighted scissor
        comp = Composition(formula).get_el_amt_dict()
        delta = sum(comp.get(X, 0) * SCISSOR[X] for X in SCISSOR) / 3.0
        raw_gap += delta

    out = {f: getattr(entry, f) for f in fields if hasattr(entry, f)}
    out["Eg"] = raw_gap
    return out


# ─── Helper: pair-specific bowing lookup ──────────────────────────────────────
def _get_bow(key: str, default: float = 0.30) -> float:
    """Return bowing parameter for a given pair key ('AB', 'AC', 'BC')."""
    return float(BOW_TABLE.get(key, default))


# ─── Gap-window scoring function ──────────────────────────────────────────────
def score_band_gap(Eg: float, lo: float, hi: float) -> float:
    """Return 1.0 inside [lo, hi], tapering linearly to 0 at 0.6 or 1.8 eV."""
    if Eg < lo:
        return max(0.0, 1 - (lo - Eg) / (hi - lo))
    if Eg > hi:
        return max(0.0, 1 - (Eg - hi) / (hi - lo))
    return 1.0


# ─── Binary screening ────────────────────────────────────────────────────────
def mix_abx3(
    formula_A: str,
    formula_B: str,
    rh: float,
    temp: float,
    bg_window: Tuple[float, float],
    bowing: float = 0.30,
    dx: float = 0.05,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> pd.DataFrame:
    lo, hi = bg_window
    dA = fetch_mp_data(formula_A, ["energy_above_hull"])
    dB = fetch_mp_data(formula_B, ["energy_above_hull"])
    if not dA or not dB:
        return pd.DataFrame()

    compA, compB = Composition(formula_A), Composition(formula_B)
    X_site = next(e.symbol for e in compA.elements if e.symbol in SCISSOR)
    rA = IONIC_RADII[next(e.symbol for e in compA.elements if e.symbol in IONIC_RADII)]
    rB = IONIC_RADII[next(e.symbol for e in compA.elements if e.symbol in {"Pb", "Sn"})]
    rX = IONIC_RADII[X_site]

    rows = []
    for x in np.arange(0, 1 + 1e-6, dx):
        # bowing for this specific pair overrides slider if present in YAML
        b_eff = _get_bow("AB", bowing)

        Eg = (1 - x) * dA["Eg"] + x * dB["Eg"] - b_eff * x * (1 - x)
        hull = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]

        stability = np.exp(-hull / K_T_EFF)  # 0–1 Boltzmann weight
        gap_score = score_band_gap(Eg, lo, hi)

        t = (rA + rX) / (np.sqrt(2) * (rB + rX))
        mu = rB / rX
        form_score = np.exp(-0.5 * ((t - 0.90) / 0.07) ** 2) * np.exp(
            -0.5 * ((mu - 0.50) / 0.07) ** 2
        )

        env_pen = 1 + alpha * (rh / 100) + beta * (temp / 100)
        score = form_score * stability * gap_score / env_pen

        rows.append(
            {
                "x": round(x, 3),
                "Eg": round(Eg, 3),
                "stability": round(stability, 3),
                "score": round(score, 3),
                "formula": f"{formula_A}-{formula_B} x={x:.2f}",
            }
        )
    return (
        pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    )


# ─── Ternary screening ───────────────────────────────────────────────────────
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
    for x in np.arange(0, 1 + 1e-6, dx):
        for y in np.arange(0, 1 - x + 1e-6, dy):
            z = 1 - x - y
            bAB = _get_bow("AB", bows.get("AB", 0.30))
            bAC = _get_bow("AC", bows.get("AC", 0.30))
            bBC = _get_bow("BC", bows.get("BC", 0.30))

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

            env_pen = 1 + (rh / 100) + (temp / 100)  # same Γ for ternary
            score = stability * gap_score / env_pen

            rows.append(
                {
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "Eg": round(Eg, 3),
                    "score": round(score, 3),
                }
            )

    return (
        pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    )


# alias for backwards compatibility
_summary = fetch_mp_data
