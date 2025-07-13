"""
EnerMat Perovskite Explorer – backend/perovskite_utils.py
Clean build • 2025‑07‑13 🟢

* calibrated experimental gaps
* convex‑hull stability filter
* **composition‑resolved Sn²⁺→Sn⁴⁺ oxidation energy ΔEox**
  (weighted by mix‑fraction so the column is **not** constant!)
* binary and ternary screen helpers consumed by the Streamlit front‑end.

Install deps:  `pip install mp-api pymatgen python-dotenv`  (Streamlit only
needed on the UI side).
"""
from __future__ import annotations

import math, os, functools, logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition

# Optional – Streamlit may not exist inside a pure‑backend unit‑test env
try:
    import streamlit as st  # type: ignore
except ModuleNotFoundError:  # no Streamlit = no UI warnings
    class _Dummy:  # pylint: disable=too-few-public-methods
        def __getattr__(self, _name):
            return None
    st = _Dummy()  # type: ignore

# ──────────────────────────────────────────── logging helpers
_log = logging.getLogger("perovskite_utils")

# ──────────────────────────────────── API key & Materials Project client
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or getattr(st.secrets, "MP_API_KEY", None)
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 Please supply your 32‑character MP_API_KEY in .env or Streamlit secrets!")
_mpr = MPRester(API_KEY)

# ───────────────────────────────────────────── reference data

# ── common end‑member library used by the Streamlit front‑end
END_MEMBERS = [
    "CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3",
]
# Legacy code in the UI expects singular name – keep as alias
END_MEMBER = END_MEMBERS

CALIBRATED_GAPS: Dict[str, float] = {
    "CsSnBr3": 1.79, "CsSnCl3": 2.83, "CsSnI3": 1.30,
    "CsPbBr3": 2.30, "CsPbI3": 1.73,
}
# PBE → exp offsets when no calibrated value available
_GAP_OFFSET = {"I": 0.90, "Br": 0.70, "Cl": 0.80}

IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

# fallback per‑atom energy for molecular O2 (PBEsol, 2025 mail talk)
_FALLBACK_E_O2 = -4.93  # eV / atom

# ──────────────────────────────────────────────── utilities

def _find_halide(formula: str) -> str:
    """Return first matching X ∈ {I,Br,Cl} in the formula."""
    try:
        return next(h for h in ("I", "Br", "Cl") if h in formula)
    except StopIteration as exc:  # pragma: no cover – non‑halide caller bug
        raise ValueError(f"No halide found in {formula}") from exc


def _fetch_mp(formula: str, fields: List[str]) -> Dict[str, Any] | None:
    """Wrapper around MP summary search that patches band_gap calibration."""
    docs = _mpr.summary.search(formula=formula, fields=tuple(fields))
    if not docs:
        _log.warning("MP entry for %s not found", formula)
        return None

    d = {f: getattr(docs[0], f, None) for f in fields}

    # patch gap if needed
    if "band_gap" in fields:
        if formula in CALIBRATED_GAPS:
            d["band_gap"] = CALIBRATED_GAPS[formula]
        else:
            hal = _find_halide(formula)
            d["band_gap"] = (d["band_gap"] or 0.0) + _GAP_OFFSET[hal]
    return d


def _safe_energy(formula: str) -> float | None:
    """Return total energy per atom if found, else None."""
    d = _fetch_mp(formula, ["energy_per_atom"])
    return None if d is None else d["energy_per_atom"]

# ────────────────────────────────── optical filter (strict 0/1 score)

def _within_band_gap(Eg: float, lo: float, hi: float) -> float:
    return 1.0 if lo <= Eg <= hi else 0.0

# ───────────────────────────────────────── Sn²⁺ → Sn⁴⁺ oxidation energy
#   CsSnX3 + ½ O2  →  ½(Cs2SnX6 + SnO2)

@functools.lru_cache(maxsize=None)
def oxidation_energy(formula_sn2: str) -> float:
    """Positive ΔEox ⇒ harder to oxidise Sn²⁺.

    Uses MP total energies. If the O₂ molecule entry is missing, fall back to a
    constant PBEsol value so the function never raises inside a screening run.
    """
    hal = _find_halide(formula_sn2)

    energies = {
        "reac": _safe_energy(f"CsSn{hal}3"),
        "prod1": _safe_energy(f"Cs2Sn{hal}6"),
        "prod2": _safe_energy("SnO2"),
        "o2": _safe_energy("O2") or _FALLBACK_E_O2,
    }
    if any(v is None for v in energies.values()):
        _log.warning("Oxidation‑energy incomplete for %s – returning 0.0", formula_sn2)
        return 0.0

    reac, prod1, prod2, e_o2 = energies.values()
    return 0.5 * prod1 + 0.5 * prod2 - reac - 0.5 * e_o2  # eV per Sn atom

# ───────────────────────────────────────────── Binary screening helper

def mix_abx3(
    formula_A: str,
    formula_B: str,
    rh: float,
    temp: float,
    bg_window: Tuple[float, float],
    bowing: float = 0.0,
    dx: float = 0.05,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> pd.DataFrame:

    lo, hi = bg_window
    dA = _fetch_mp(formula_A, ["band_gap", "energy_above_hull"])
    dB = _fetch_mp(formula_B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    # pre‑compute oxidation energies of endpoints
    eox_A = oxidation_energy(formula_A)
    eox_B = oxidation_energy(formula_B)

    # geometric tolerance factor – use A‑site radii as proxy
    comp_A = Composition(formula_A)
    A_site = next(e.symbol for e in comp_A.elements if e.symbol in IONIC_RADII)
    B_site = next(e.symbol for e in comp_A.elements if e.symbol in {"Pb", "Sn"})
    X_site = _find_halide(formula_A)
    rA, rB, rX = IONIC_RADII[A_site], IONIC_RADII[B_site], IONIC_RADII[X_site]

    rows: List[Dict[str, Any]] = []
    for x in np.arange(0.0, 1.0 + 1e-6, dx):
        Eg = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bowing * x * (1 - x)
        Eh = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        stab = max(0.0, 1 - Eh / 0.025)  # ≥1 inside 25 meV/atom,
        gap  = _within_band_gap(Eg, lo, hi)

        t  = (rA + rX) / (math.sqrt(2) * (rB + rX))
        mu = rB / rX
        form = math.exp(-0.5 * ((t - 0.90) / 0.07) ** 2) * math.exp(-0.5 * ((mu - 0.50) / 0.07) ** 2)

        # composition‑weighted oxidation penalty (NEW – varies with x!)
        eox = (1 - x) * eox_A + x * eox_B
        ox_pen = math.exp(eox / 0.20)  # +ve ΔEox favours, −ve penalises

        env = 1 + alpha * rh / 100 + beta * temp / 100
        score = form * stab * gap * ox_pen / env

        rows.append({
            "x": round(x, 3),
            "Eg": round(Eg, 3),
            "Ehull": round(Eh, 4),
            "Eox": round(eox, 3),
            "score": round(score, 3),
            "formula": f"{formula_A}-{formula_B} x={x:.2f}",
        })

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# ────────────────────────────────────────────── Ternary screening

def screen_ternary(
    A: str,
    B: str,
    C: str,
    rh: float,
    temp: float,
    bg: Tuple[float, float],
    bows: Dict[str, float] | None = None,
    dx: float = 0.10,
    dy: float = 0.10,
) -> pd.DataFrame:

    bows = bows or {"AB": 0.0, "AC": 0.0, "BC": 0.0}
    dA = _fetch_mp(A, ["band_gap", "energy_above_hull"])
    dB = _fetch_mp(B, ["band_gap", "energy_above_hull"])
    dC = _fetch_mp(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    eox_A, eox_B, eox_C = oxidation_energy(A), oxidation_energy(B), oxidation_energy(C)
    lo, hi = bg

    rows: List[Dict[str, Any]] = []
    for x in np.arange(0.0, 1.0 + 1e-6, dx):
        for y in np.arange(0.0, 1.0 - x + 1e-6, dy):
            z = 1 - x - y
            Eg = (z * dA["band_gap"] + x * dB["band_gap"] + y * dC["band_gap"]
                  - bows["AB"] * x * z - bows["AC"] * y * z - bows["BC"] * x * y)

            Eh = (z * dA["energy_above_hull"] + x * dB["energy_above_hull"] + y * dC["energy_above_hull"]
                  + bows["AB"] * x * z + bows["AC"] * y * z + bows["BC"] * x * y)

            stab = math.exp(-max(Eh, 0) / 0.025)
            gap  = _within_band_gap(Eg, lo, hi)

            eox = z * eox_A + x * eox_B + y * eox_C
            score = stab * gap * math.exp(eox / 0.20)

            rows.append({
                "x": round(x, 3),
                "y": round(y, 3),
                "Eg": round(Eg, 3),
                "Ehull": round(Eh, 4),
                "Eox": round(eox, 3),
                "score": round(score, 3),
                "formula": f"{A}-{B}-{C} x={x:.2f} y={y:.2f}",
            })

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# backward compat alias (legacy notebooks)
_summary = _fetch_mp
