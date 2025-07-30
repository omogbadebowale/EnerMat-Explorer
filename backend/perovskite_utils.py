from __future__ import annotations
import math
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from mp_api.client import MPRester
from pymatgen.core import Composition

# ─────────── Shockley–Queisser helper ───────────
from backend.sq import sq_efficiency

# ─────────── API key ───────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32-character MP_API_KEY missing")

mpr = MPRester(API_KEY)

# ─────────── reference data ───────────
END_MEMBERS = ["CsSnI3", "CsSnBr3", "CsSnCl3", "CsGeBr3", "CsGeCl3", "CsPbCl3", "CsPbBr3", "CsPbI3", 
               "CsSnSe3", "CsSnTe3", "CsGeI3", "CsSnF3", "CsGeF3", "CsPbF3", "CsPb(SCN)3", "CsPb(Br1-xIx)3"]

CALIBRATED_GAPS = {
    "CsSnBr3": 1.30,
    "CsSnCl3": 2.40,
    "CsSnI3":  1.00,
    "CsGeBr3": 2.20,
    "CsGeCl3": 2.7,
    "CsPbI3": 1.73,
    "CsPbBr3": 2.30,
    "CsPbCl3": 2.32,
    "CsSnSe3": 1.10,
    "CsSnTe3": 1.20,
    "CsPbF3": 2.00,
}

GAP_OFFSET = {"I": +0.52, "Br": +0.88, "Cl": +1.10, "Pb": 1.31, }

# ─────────── helpers ───────────
def fetch_mp_data(formula: str, fields: list[str]):
    docs = mpr.summary.search(formula=formula, fields=tuple(fields))
    if not docs:
        return None
    ent = docs[0]
    out = {f: getattr(ent, f, None) for f in fields}
    return out

@lru_cache(maxsize=64)
def oxidation_energy(formula_sn2: str) -> float:
    """ΔEₒₓ per Sn for CsSnX₃ + ½ O₂ → ½ (Cs₂SnX₆ + SnO₂)."""
    if "Sn" not in formula_sn2:
        return 0.0
    hal = next((h for h in ("I", "Br", "Cl") if h in formula_sn2), None)
    if hal is None:
        return 0.0

    def formation_energy_fu(formula: str) -> float:
        doc = fetch_mp_data(formula, ["formation_energy_per_atom"])
        if not doc or doc.get("formation_energy_per_atom") is None:
            raise ValueError(f"Missing formation-energy for {formula}")
        comp = Composition(formula)
        return doc["formation_energy_per_atom"] * comp.num_atoms

    H_reac  = formation_energy_fu(formula_sn2)
    H_prod1 = formation_energy_fu(f"Cs2Sn{hal}6")
    H_prod2 = formation_energy_fu("SnO2")
    return 0.5 * (H_prod1 + H_prod2) - H_reac

# ─────────── binary screen ───────────
def screen_binary(
    A: str,
    B: str,
    rh: float,
    temp: float,
    bg: tuple[float, float],
    bow: float,
    dx: float,
    *,
    z: float = 0.0,
    application: str | None = None,
    substitution_element: str = "Ge",  # New parameter for substitution
) -> pd.DataFrame:
    lo, hi = bg
    center = sigma = None
    if application in APPLICATION_CONFIG:
        cfg = APPLICATION_CONFIG[application]
        lo, hi = cfg["range"]
        center, sigma = cfg["center"], cfg["sigma"]

    # Substitution logic
    A_sub = A.replace("Sn", substitution_element)  # Handle substitution
    B_sub = B.replace("Sn", substitution_element)  # Handle substitution

    dA = fetch_mp_data(A_sub, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B_sub, ["band_gap", "energy_above_hull"])

    if not (dA and dB):
        return pd.DataFrame()

    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        # Sn branch
        Eg_Sn   = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bow * x * (1 - x)
        Eh_Sn   = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        # Calculate oxidation energy
        dEox_Sn = (1 - x) * oxidation_energy(A) + x * oxidation_energy(B)

        sbg = _score_band_gap(Eg_Sn, lo, hi, center, sigma)
        raw = (
            sbg
            * math.exp(-Eh_Sn / 0.0259)
            * math.exp(dEox_Sn / K_T_EFF)
        )

        # Shockley–Queisser PCE limit
        pce = sq_efficiency(Eg_Sn)

        rows.append({
            "x":           round(x, 3),
            "Eg":          round(Eg_Sn, 3),
            "Ehull":       round(Eh_Sn, 4),
            "Eox":         round(dEox_Sn, 3),
            "raw":         raw,
            "formula":     f"{A}-{B} x={x:.2f}",
            "PCE_max (%)": round(pce * 100, 1),
        })

    if not rows:
        return pd.DataFrame()

    m = max(r["raw"] for r in rows) or 1.0
    for r in rows:
        r["score"] = round(r.pop("raw") / m, 3)

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

# ─────────── ternary screen ───────────
def screen_ternary(
    A: str,
    B: str,
    C: str,
    rh: float,
    temp: float,
    bg: tuple[float, float],
    bows: dict[str, float],
    *,
    dx: float = 0.10,
    dy: float = 0.10,
    z: float = 0.0,
    application: str | None = None,
    substitution_element: str = "Ge",  # New parameter for substitution
) -> pd.DataFrame:
    lo, hi = bg
    center = sigma = None
    if application in APPLICATION_CONFIG:
        cfg = APPLICATION_CONFIG[application]
        lo, hi = cfg["range"]
        center, sigma = cfg["center"], cfg["sigma"]

    # Substitution logic for ternary
    A_sub = A.replace("Sn", substitution_element)  # Handle substitution
    B_sub = B.replace("Sn", substitution_element)  # Handle substitution
    C_sub = C.replace("Sn", substitution_element)  # Handle substitution

    dA = fetch_mp_data(A_sub, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B_sub, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C_sub, ["band_gap", "energy_above_hull"])

    if not (dA and dB and dC):
        return pd.DataFrame()

    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        for y in np.arange(0.0, 1.0 - x + 1e-9, dy):
            w = 1.0 - x - y
            # Sn gap
            Eg_Sn = (
                w * dA["band_gap"] + x * dB["band_gap"] + y * dC["band_gap"]
                - bows["AB"] * x * w - bows["AC"] * y * w - bows["BC"] * x * y
            )

            Eh_Sn = (
                w * dA["energy_above_hull"] + x * dB["energy_above_hull"] + y * dC["energy_above_hull"]
            )
            sbg = _score_band_gap(Eg_Sn, lo, hi, center, sigma)
            raw = sbg * math.exp(-Eh_Sn / 0.0518)

            # Shockley–Queisser PCE limit
            pce = sq_efficiency(Eg_Sn)

            rows.append({
                "x": round(x, 3),
                "y": round(y, 3),
                "z": round(z, 2),
                "Eg": round(Eg_Sn, 3),
                "Ehull": round(Eh_Sn, 4),
                "PCE_max (%)": round(pce * 100, 1),
                "raw": raw,
                "formula": f"{A}-{B}-{C} x={x:.2f} y={y:.2f} z={z:.2f}",
            })

    if not rows:
        return pd.DataFrame()

    m = max(r["raw"] for r in rows) or 1.0
    for r in rows:
        r["score"] = round(r.pop("raw") / m, 3)

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
