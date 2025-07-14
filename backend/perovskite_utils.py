from __future__ import annotations
import math, os
from functools import lru_cache

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from mp_api.client import MPRester
from pymatgen.core import Composition

# ─────────────────────────  API key  ──────────────────────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32‑character MP_API_KEY missing in env or secrets")

mpr = MPRester(API_KEY)

# ────────────────────── reference tables ─────────────────────
END_MEMBERS = [
    "CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3",
]

CALIBRATED_GAPS = {
    "CsSnBr3": 1.79,
    "CsSnCl3": 2.83,
    "CsPbI3": 1.46,
    "CsPbBr3": 2.32,
}

GAP_OFFSET = {"I": +0.52, "Br": +0.88, "Cl": +1.10}

IONIC_RADII = {
    "Cs": 1.88, "Sn": 1.18, "Pb": 1.19,
    "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

K_T_EFF = 0.20

# -----------------------------------------------------------------------------
def fetch_mp_data(formula: str, fields: list[str]):
    docs = mpr.summary.search(formula=formula, fields=tuple(fields))
    if not docs:
        return None
    entry = docs[0]
    out = {f: getattr(entry, f, None) for f in fields}
    if "band_gap" in fields:
        if formula in CALIBRATED_GAPS:
            out["band_gap"] = CALIBRATED_GAPS[formula]
        else:
            hal = next(h for h in ("I", "Br", "Cl") if h in formula)
            out["band_gap"] = (out["band_gap"] or 0.0) + GAP_OFFSET[hal]
    return out

@lru_cache(maxsize=64)
def oxidation_energy(formula_sn2: str) -> float:
    if "Sn" not in formula_sn2:
        return 0.0

    hal = next((h for h in ("I", "Br", "Cl") if h in formula_sn2), None)
    if hal is None:
        return 0.0

    def formation_energy_fu(formula: str) -> float:
        doc = fetch_mp_data(formula, ["formation_energy_per_atom"])
        if not doc or doc["formation_energy_per_atom"] is None:
            raise ValueError(f"Missing formation-energy for {formula}")
        comp = Composition(formula)
        return doc["formation_energy_per_atom"] * comp.num_atoms

    try:
        H_reac = formation_energy_fu(formula_sn2)
        H_prod1 = formation_energy_fu(f"Cs2Sn{hal}6")
        H_prod2 = formation_energy_fu("SnO2")
        return 0.5 * (H_prod1 + H_prod2) - H_reac
    except Exception:
        return 0.0

score_band_gap = lambda Eg, lo, hi: 1.0 if lo <= Eg <= hi else 0.0

# -----------------------------------------------------------------------------
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
    **kwargs,
) -> pd.DataFrame:

    lo, hi = bg_window
    dA = fetch_mp_data(formula_A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(formula_B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    hal = next((h for h in ("I", "Br", "Cl") if h in formula_A), None)
    if hal is None:
        return pd.DataFrame()

    rA, rB, rX = (IONIC_RADII.get(k, 1.5) for k in ("Cs", "Sn", hal))
    dEox_A = oxidation_energy(formula_A)
    dEox_B = oxidation_energy(formula_B)

    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        Eg = (1 - x) * dA.get("band_gap", 0) + x * dB.get("band_gap", 0) - bowing * x * (1 - x)
        Eh = (1 - x) * dA.get("energy_above_hull", 0) + x * dB.get("energy_above_hull", 0)
        dEox = (1 - x) * dEox_A + x * dEox_B
        ox_pen = math.exp(dEox / K_T_EFF)
        stab = math.exp(-Eh / (alpha * K_T_EFF))
        tfac = (rA + rX) / (math.sqrt(2) * (rB + rX))
        fit = math.exp(-beta * abs(tfac - 0.95))
        form = score_band_gap(Eg, lo, hi)
        score = form * stab * fit * ox_pen

        rows.append({
            "x": round(x, 3),
            "Eg": round(Eg, 3),
            "Ehull": round(Eh, 4),
            "Eox": round(dEox, 3),
            "score": round(score, 3),
            "formula": f"{formula_A}-{formula_B} x={x:.2f}",
        })

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# -----------------------------------------------------------------------------
def screen_ternary(
    A: str,
    B: str,
    C: str,
    rh: float,
    temp: float,
    bg: tuple[float, float],
    bows: dict[str, float],
    dx: float = 0.10,
    dy: float = 0.10,
    n_mc: int = 200,
) -> pd.DataFrame:

    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull", "Eox_e"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull", "Eox_e"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull", "Eox_e"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    lo, hi = bg
    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        for y in np.arange(0.0, 1.0 - x + 1e-9, dy):
            z = 1 - x - y

            Eg = (
                z * dA.get("band_gap", 0)
                + x * dB.get("band_gap", 0)
                + y * dC.get("band_gap", 0)
                - bows["AB"] * x * z
                - bows["AC"] * y * z
                - bows["BC"] * x * y
            )

            Eh = (
                z * dA.get("energy_above_hull", 0)
                + x * dB.get("energy_above_hull", 0)
                + y * dC.get("energy_above_hull", 0)
            )

            dEox = (
                z * dA.get("Eox_e", 0)
                + x * dB.get("Eox_e", 0)
                + y * dC.get("Eox_e", 0)
            )

            ox_pen = math.exp(dEox / K_T_EFF)
            form = score_band_gap(Eg, lo, hi)
            stab = math.exp(-Eh / (0.0259 * 2.0))
            score = form * stab * ox_pen

            rows.append({
                "x": round(x, 3), "y": round(y, 3),
                "Eg": round(Eg, 3), "Ehull": round(Eh, 4), "Eox": round(dEox, 3),
                "score": round(score, 3),
                "formula": f"{A}-{B}-{C} x={x:.2f} y={y:.2f}",
            })

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

_summary = fetch_mp_data
