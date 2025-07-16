# backend/perovskite_utils.py
# EnerMat utilities  v9.6  (2025-07-15, Ge-ready)

from __future__ import annotations
import math, os
from functools import lru_cache

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from mp_api.client import MPRester
from pymatgen.core import Composition

# ─────────── API key ───────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32-character MP_API_KEY missing")

mpr = MPRester(API_KEY)

# ─────────── reference data ───────────
END_MEMBERS = ["CsSnI3", "CsSnBr3", "CsSnCl3", "CsGeBr3", "CsGeCl3"]

CALIBRATED_GAPS = {
    "CsSnBr3": 1.79,
    "CsSnCl3": 2.83,
    "CsSnI3":  1.00,
    "CsGeBr3": 2.20,
    "CsGeCl3": 3.30,
}

GAP_OFFSET = {"I": +0.52, "Br": +0.88, "Cl": +1.10}
IONIC_RADII = {"Cs": 1.88, "Sn": 1.18, "Ge": 0.73,
               "I": 2.20, "Br": 1.96, "Cl": 1.81}

K_T_EFF = 0.20          # soft-penalty “kT” (eV)
score_band_gap = lambda Eg, lo, hi: 1.0 if lo <= Eg <= hi else 0.0

# ─────────── helpers ───────────
def fetch_mp_data(formula: str, fields: list[str]):
    docs = mpr.summary.search(formula=formula, fields=tuple(fields))
    if not docs:
        return None
    ent = docs[0]
    out = {f: getattr(ent, f, None) for f in fields}

    if "band_gap" in fields:
        if formula in CALIBRATED_GAPS:
            out["band_gap"] = CALIBRATED_GAPS[formula]
        else:
            hal = next(h for h in ("I", "Br", "Cl") if h in formula)
            out["band_gap"] = (out["band_gap"] or 0.0) + GAP_OFFSET[hal]
    return out


@lru_cache(maxsize=64)
def oxidation_energy(formula_sn2: str) -> float:
    """ΔEₒₓ per Sn for   CsSnX₃ + ½ O₂ → ½ (Cs₂SnX₆ + SnO₂).
       Positive ⇒ Sn(II) oxidation is uphill (good)."""
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

    H_reac  = formation_energy_fu(formula_sn2)       # CsSnX3
    H_prod1 = formation_energy_fu(f"Cs2Sn{hal}6")    # Cs2SnX6
    H_prod2 = formation_energy_fu("SnO2")            # SnO2
    return 0.5 * (H_prod1 + H_prod2) - H_reac


# ─────────── binary screen ───────────
def screen_binary(A, B, rh, temp, bg, bow, dx, *, z: float = 0.0):
    return mix_abx3(A, B, rh, temp, bg, bow, dx, z=z)


def mix_abx3(A, B, rh, temp, bg, bow, dx,
             z: float = 0.0, alpha: float = 1.0, beta: float = 1.0) -> pd.DataFrame:

    lo, hi = bg
    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    hal = next(h for h in ("I", "Br", "Cl") if h in A)
    rA, rB, rX = (IONIC_RADII[k] for k in ("Cs", "Sn", hal))
    oxA, oxB = oxidation_energy(A), oxidation_energy(B)

    rows = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        Eg = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bow * x * (1 - x)
        Eh = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        dEox = (1 - x) * oxA + x * oxB
        raw = (
            score_band_gap(Eg, lo, hi)
            * math.exp(-Eh / (alpha * 0.0259))
            * math.exp(dEox / K_T_EFF)
            * math.exp(-beta * abs((rA + rX) / (math.sqrt(2) * (rB + rX)) - 0.95))
        )
        rows.append(
            {
                "x": round(x, 3),
                "z": round(z, 2),
                "Eg": round(Eg, 3),
                "Ehull": round(Eh, 4),
                "Eox": round(dEox, 3),
                "raw": raw,
                "formula": f"{A}-{B} x={x:.2f} z={z:.2f}",
            }
        )

    if not rows:
        return pd.DataFrame()
    m = max(r["raw"] for r in rows) or 1.0
    for r in rows:
        r["score"] = round(r["raw"] / m, 3)
        del r["raw"]
    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

# ───────────────────── ternary screen  (Sn ⇄ Ge ready) ──────────────────────
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
    z: float = 0.0,          # ← NEW keyword – Ge fraction on the B-site
) -> pd.DataFrame:
    """Score the ternary CsSnX₃ win–lose triangle with optional Ge-fraction *z*.

    A + B + C are the three vertex formulas (Sn–halide perovskites).
    If z>0, each Sn vertex gets an implicit Ge analogue (CsGeX₃) and
    the electronic/stability terms are linearly interpolated:
        Eg   = (1-z)·Eg_Sn   + z·Eg_Ge
        Eh   = (1-z)·Eh_Sn   + z·Eh_Ge
    Oxidation energy is kept for the Sn branch only (Ge²⁺↔Ge⁴⁺ is negligible).
    """
    # --- fetch Sn branch ----------------------------------------------------
    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    # --- optional Ge branch -------------------------------------------------
    if z > 0:
        A_Ge = A.replace("Sn", "Ge")
        B_Ge = B.replace("Sn", "Ge")
        C_Ge = C.replace("Sn", "Ge")
        dA_Ge = fetch_mp_data(A_Ge, ["band_gap", "energy_above_hull"]) or dA
        dB_Ge = fetch_mp_data(B_Ge, ["band_gap", "energy_above_hull"]) or dB
        dC_Ge = fetch_mp_data(C_Ge, ["band_gap", "energy_above_hull"]) or dC
    else:
        dA_Ge, dB_Ge, dC_Ge = dA, dB, dC    # dummy aliases

    # --- oxidation (Sn only) ------------------------------------------------
    oxA, oxB, oxC = (oxidation_energy(f) for f in (A, B, C))
    lo, hi = bg
    rows: list[dict] = []

    # --- composition grid ---------------------------------------------------
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        for y in np.arange(0.0, 1.0 - x + 1e-9, dy):
            w = 1.0 - x - y                     # w == zSn in literature, but we keep z for Ge
            # ---------- band gap -------------
            Eg_Sn = (
                w * dA["band_gap"]
              + x * dB["band_gap"]
              + y * dC["band_gap"]
              - bows["AB"] * x * w
              - bows["AC"] * y * w
              - bows["BC"] * x * y
            )
            Eg_Ge = (
                w * dA_Ge["band_gap"]
              + x * dB_Ge["band_gap"]
              + y * dC_Ge["band_gap"]
              - bows["AB"] * x * w
              - bows["AC"] * y * w
              - bows["BC"] * x * y
            )
            Eg = (1.0 - z) * Eg_Sn + z * Eg_Ge

            # ---------- Ehull ---------------
            Eh_Sn = (
                w * dA["energy_above_hull"]
              + x * dB["energy_above_hull"]
              + y * dC["energy_above_hull"]
            )
            Eh_Ge = (
                w * dA_Ge["energy_above_hull"]
              + x * dB_Ge["energy_above_hull"]
              + y * dC_Ge["energy_above_hull"]
            )
            Eh = (1.0 - z) * Eh_Sn + z * Eh_Ge

            # ---------- oxidation -----------
            dEox = w * oxA + x * oxB + y * oxC    # keep Sn only

            # ---------- score ---------------
            raw = (
                score_band_gap(Eg, lo, hi)
                * math.exp(-Eh / 0.0518)
                * math.exp(dEox / K_T_EFF)
            )
            rows.append(
                {
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "z": round(z, 2),
                    "Eg": round(Eg, 3),
                    "Ehull": round(Eh, 4),
                    "Eox": round(dEox, 3),
                    "raw": raw,
                    "formula": f"{A}-{B}-{C} x={x:.2f} y={y:.2f} z={z:.2f}",
                }
            )

    if not rows:
        return pd.DataFrame()

    raw_max = max(r["raw"] for r in rows) or 1.0
    for r in rows:
        r["score"] = round(r["raw"] / raw_max, 3)
        del r["raw"]

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
)
