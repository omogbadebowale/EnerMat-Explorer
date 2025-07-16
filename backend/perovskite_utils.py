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
        r["score"] = round(r.pop("raw") / m, 3)
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


# ─────────── ternary screen ─────────── (Sn ⇄ Ge ready)
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
) -> pd.DataFrame:
    # [existing ternary code unchanged]
    # …
    pass  # your existing implementation


# ─────────── NEW: Quad-mix PEA/FA-Sn/Ge + I/Br ───────────
def screen_pea_fa_sn_ge_halide(
    rh: float,
    temp: float,
    bg: tuple[float,float],
    bow_halide: float,
    dx: float = 0.05,
    dz: float = 0.05,
) -> pd.DataFrame:
    """
    Scores PEA0.15FA0.85Sn1−zGez(I1−nBrn)3 over (n,z) grid;
    returns columns [n, z, Eg, Ehull, Eox, score].
    """
    lo, hi = bg

    # endmembers for Sn/Ge binary at each halide
    sn_i = "FASnI3"
    sn_b = "FASnBr3"
    ge_i = "FAGeI3"
    ge_b = "FAGeBr3"

    # precompute halide curves
    df_sn = mix_abx3(sn_i, sn_b, rh, temp, bg, bow_halide, dx, z=0)
    df_ge = mix_abx3(ge_i, ge_b, rh, temp, bg, bow_halide, dx, z=0)

    ox_sn = oxidation_energy(sn_i)
    ox_ge = oxidation_energy(ge_i)

    rows = []
    for idx, r_sn in df_sn.iterrows():
        n = r_sn["x"]
        r_ge = df_ge.iloc[idx]
        for z_val in np.arange(0.0, 1.0+1e-9, dz):
            Eg   = (1-z_val)*r_sn["Eg"]   + z_val*r_ge["Eg"]
            Eh   = (1-z_val)*r_sn["Ehull"]+ z_val*r_ge["Ehull"]
            Eox  = (1-z_val)*ox_sn        + z_val*ox_ge
            raw  = (
                score_band_gap(Eg, lo, hi)
                * math.exp(-Eh/0.0259)
                * math.exp(Eox/K_T_EFF)
            )
            rows.append({
                "n": round(n,3),
                "z": round(z_val,3),
                "Eg": round(Eg,3),
                "Ehull": round(Eh,4),
                "Eox": round(Eox,3),
                "raw": raw
            })

    if not rows:
        return pd.DataFrame()

    m = max(r["raw"] for r in rows) or 1.0
    for r in rows:
        r["score"] = round(r.pop("raw")/m,3)

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
