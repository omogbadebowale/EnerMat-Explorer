from __future__ import annotations
import math, os
from functools import lru_cache

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from mp_api.client import MPRester
from pymatgen.core import Composition

# ─────────────────────────────── API key ───────────────────────────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🚩 32‑character MP_API_KEY missing in env or secrets")

mpr = MPRester(API_KEY)

# ─────────────────────────────── Reference Tables ───────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

CALIBRATED_GAPS = {
    "CsSnBr3": 1.79,
    "CsSnCl3": 2.83,
    "CsPbI3": 1.46,
    "CsPbBr3": 2.32,
}

GAP_OFFSET = {"I": +0.52, "Br": +0.88, "Cl": +1.10}
IONIC_RADII = {"Cs": 1.88, "Sn": 1.18, "Pb": 1.19, "I": 2.20, "Br": 1.96, "Cl": 1.81}
K_T_EFF = 0.20

# ─────────────────────────────── Utilities ───────────────────────────────
def fetch_mp_data(formula: str, fields: list[str]):
    if "Eox_e" in fields:
        fields = [f for f in fields if f != "Eox_e"]
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

# ─────────────────────────────── Binary Screen ───────────────────────────────
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
    z: float | None = None,             # ← slider arrives here
) -> pd.DataFrame:
    """
    Binary CsSnX3  (A)  <—>  CsSnX3  (B)
    Optional B-site mixing:  CsSn(1−z)Ge_zX3
    """
    # ---------------------------------------------------------------------
    lo, hi = bg_window
    dA = fetch_mp_data(formula_A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(formula_B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    hal = next(h for h in ("I", "Br", "Cl") if h in formula_A)
    rA, rB, rX = (IONIC_RADII.get(k, 1.5) for k in ("Cs", "Sn", hal))

    # --- B-site Ge data (only loaded if z is not None) -------------------
    if z and z > 0:
        geA = formula_A.replace("Sn", "Ge")
        geB = formula_B.replace("Sn", "Ge")
        dGA = fetch_mp_data(geA, ["band_gap", "energy_above_hull"])
        dGB = fetch_mp_data(geB, ["band_gap", "energy_above_hull"])
        if not (dGA and dGB):
            st.warning("⛔ No Ge end-member data on MP – Ge mixing disabled.")
            z = 0.0
    else:
        z = 0.0  # ensure numeric

    dEox_A = oxidation_energy(formula_A)
    dEox_B = oxidation_energy(formula_B)

    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        # ---------- band gap ------------------------------------------------
        Eg_sn = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bowing * x * (1 - x)
        Eg    = (1 - z) * Eg_sn + z * ((1 - x) * dGA["band_gap"] + x * dGB["band_gap"])
        # ---------- Ehull ---------------------------------------------------
        Eh_sn = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        Eh    = (1 - z) * Eh_sn + z * ((1 - x) * dGA["energy_above_hull"] + x * dGB["energy_above_hull"])
        # ---------- oxidation ----------------------------------------------
        dEox_sn = (1 - x) * dEox_A + x * dEox_B
        dEox    = (1 - z) * dEox_sn  # assume Ge²⁺ oxidation neutral
        # ---------- scoring -------------------------------------------------
        ox_pen = math.exp(dEox / K_T_EFF)
        stab   = math.exp(-Eh / (alpha * K_T_EFF))
        tfac   = (rA + rX) / (math.sqrt(2) * (rB + rX))
        fit    = math.exp(-beta * abs(tfac - 0.95))
        form   = score_band_gap(Eg, lo, hi)
        score  = form * stab * fit * ox_pen
        # ---------- row -----------------------------------------------------
        rows.append({
            "x": round(x, 3), "z": round(z, 3),
            "Eg": round(Eg, 3), "Ehull": round(Eh, 4), "Eox": round(dEox, 3),
            "score": round(score, 3),
            "formula": f"CsSn(1-{z:.2f})Ge{z:.2f}I{3*(1-x):.2f}Br{3*x:.2f}",
        })

    return (pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True))
# ─────────────────────────────── Ternary Screen ───────────────────────────────
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

    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    oxA = oxidation_energy(A)
    oxB = oxidation_energy(B)
    oxC = oxidation_energy(C)

    if not (dA and dB and dC):
        return pd.DataFrame()

    lo, hi = bg
    rows = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        for y in np.arange(0.0, 1.0 - x + 1e-9, dy):
            z = 1 - x - y
            Eg = (z * dA["band_gap"] + x * dB["band_gap"] + y * dC["band_gap"]
                  - bows["AB"] * x * z - bows["AC"] * y * z - bows["BC"] * x * y)
            Eh = (z * dA["energy_above_hull"] + x * dB["energy_above_hull"] + y * dC["energy_above_hull"])
            dEox = z * oxA + x * oxB + y * oxC
            ox_pen = math.exp(dEox / K_T_EFF)
            stab = math.exp(-Eh / (0.0259 * 2.0))
            form = score_band_gap(Eg, lo, hi)
            score = 1e3 * form * stab * ox_pen      # rescale for readability
            rows.append({
                "x": round(x, 3), "y": round(y, 3),
                "Eg": round(Eg, 3), "Ehull": round(Eh, 4),
                "Eox": round(dEox, 3), "score": round(score, 3),
                "formula": f"{A}-{B}-{C} x={x:.2f} y={y:.2f}"
            })

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# ─────────────────────────────── Legacy Export ───────────────────────────────
_summary = fetch_mp_data
