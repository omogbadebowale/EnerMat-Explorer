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

# ─────────── application-based band-gap targets ───────────
APPLICATION_CONFIG = {
    "single": {"range": (1.10, 1.40), "center": 1.25, "sigma": 0.10},
    "tandem": {"range": (1.60, 1.90), "center": 1.75, "sigma": 0.10},
    "indoor": {"range": (1.70, 2.20), "center": 1.95, "sigma": 0.15},
    "detector": {"range": (0.80, 3.00), "center": None,  "sigma": None},
}

# ─────────── reference data ───────────
END_MEMBERS = ["CsSnI3", "CsSnBr3", "CsSnCl3", "CsGeBr3", "CsGeCl3", "CsPbCl3", "CsPbBr3", "CsPbI3", "CsSbBr3", "CsSbCl3", "CsCuBr3", "CsCuCl3", "CsMgBr3", "CsMgCl3"
              "CsCaBr3", "CsCaCl3", "CsBaBr3", "CsBaCl3", "CsNiBr3", "CsNiCl3", "CsZnBr3", "CsZnCl3"]

CALIBRATED_GAPS = {
     "CsSnBr3": 1.30,
    "CsSnCl3": 2.40,
    "CsSnI3": 1.00,
    "CsGeBr3": 2.20,
    "CsGeCl3": 2.7,
    "CsPbI3": 1.73,
    "CsPbBr3": 2.30,
    "CsPbCl3": 2.32,
    "CsSbBr3": 2.05,  # Example value for Antimony doping, check literature
    "CsSbCl3": 2.15,  # Example value for Antimony doping, check literature
    "CsCuBr3": 2.40,  # Example value for Copper doping, check literature
    "CsCuCl3": 2.45,  # Example value for Copper doping, check literature
    "CsMgBr3": 2.55,  # Example value for Magnesium doping, check literature
    "CsMgCl3": 2.60,  # Example value for Magnesium doping, check literature
    "CsCaBr3": 2.50,  # Example value for Calcium doping, check literature
    "CsCaCl3": 2.55,  # Example value for Calcium doping, check literature
    "CsBaBr3": 2.45,  # Example value for Barium doping, check literature
    "CsBaCl3": 2.50,  # Example value for Barium doping, check literature
    "CsNiBr3": 2.70,  # Example value for Nickel doping, check literature
    "CsNiCl3": 2.75,  # Example value for Nickel doping, check literature
    "CsZnBr3": 2.50,  # Example value for Zinc doping, check literature
    "CsZnCl3": 2.55,  # Example value for Zinc doping, check literature
}

GAP_OFFSET = {
    "I": +0.52,   # Iodine offset, typically consistent
    "Br": +0.88,  # Bromine offset, typically consistent
    "Cl": +1.10,  # Chlorine offset, typically consistent
    "Pb": 1.31,   # Lead-specific offset (perovskites involving Pb)
    "Sb": 0.60,   # Antimony-specific offset (check literature)
    "Cu": 0.65,   # Copper-specific offset (check literature)
    "Mg": 0.70,   # Magnesium-specific offset (check literature)
    "Ca": 0.75,   # Calcium-specific offset (check literature)
    "Ba": 0.80,   # Barium-specific offset (check literature)
    "Ni": 0.85,   # Nickel-specific offset (check literature)
    "Zn": 0.90,   # Zinc-specific offset (check literature)
}

IONIC_RADII = {
    IONIC_RADII = {
    "Cs": 1.88,
    "Sn": 1.18,
    "Ge": 0.73,
    "I": 2.20,  # Iodine Ion radius
    "Br": 1.96,
    "Cl": 1.81,
    "Pb": 1.31,
    "Sb": 0.76,  # If Sb is used in calculations
    "Cu": 0.77,  # If Cu is used in calculations
    "Mg": 0.72,  # If Mg is used in calculations
    "Ca": 0.99,  # If Ca is used in calculations
    "Ba": 1.61,  # If Ba is used in calculations
    "Ni": 0.69,  # If Ni is used in calculations
    "Zn": 0.74   # If Zn is used in calculations
}

K_T_EFF = 0.20  # soft-penalty “kT” (eV)

# ─────────── band-gap scoring ───────────
def _score_band_gap(Eg: float, lo: float, hi: float, center: float | None, sigma: float | None) -> float:
    if Eg < lo or Eg > hi:
        return 0.0
    if center is None or sigma is None:
        return 1.0
    # Gaussian weighting
    return math.exp(-((Eg - center) ** 2) / (2 * sigma * sigma))

score_band_gap = _score_band_gap  # alias

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
            out["band_gap"] = (out.get("band_gap", 0.0) or 0.0) + GAP_OFFSET[hal]
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
    doping_element: str = "Ge",  # New doping element parameter
) -> pd.DataFrame:
    lo, hi = bg
    center = sigma = None
    if application in APPLICATION_CONFIG:
        cfg = APPLICATION_CONFIG[application]
        lo, hi = cfg["range"]
        center, sigma = cfg["center"], cfg["sigma"]

    return mix_abx3(A, B, rh, temp, (lo, hi), bow, dx,
                    z=z, center=center, sigma=sigma, doping_element=doping_element)


def mix_abx3(
    A: str,
    B: str,
    rh: float,
    temp: float,
    bg: tuple[float, float],
    bow: float,
    dx: float,
    *,
    z: float = 0.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    center: float | None = None,
    sigma: float | None = None,
    doping_element: str = "Ge",  # Handle doping element in the mix_abx3 function
) -> pd.DataFrame:
    lo, hi = bg
    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    # Apply doping element logic if z > 0
    if doping_element != "Ge":
        A = A.replace("Sn", doping_element)
        B = B.replace("Sn", doping_element)

    # Continue with normal computation for binary mix
    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        Eg_Sn = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bow * x * (1 - x)
        Eh_Sn = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        rA, rB, rX = (IONIC_RADII[k] for k in ("Cs", "Sn", "I"))
        oxA, oxB = oxidation_energy(A), oxidation_energy(B)

        # Compute doping and band gap score
        sbg = _score_band_gap(Eg_Sn, lo, hi, center, sigma)
        raw = sbg * math.exp(-Eh_Sn / (alpha * 0.0259)) * math.exp(oxA / K_T_EFF)

        pce = sq_efficiency(Eg_Sn)
        rows.append({
            "x": round(x, 3),
            "z": round(z, 2),
            "Eg": round(Eg_Sn, 3),
            "Ehull": round(Eh_Sn, 4),
            "Eox": round(oxA, 3),
            "raw": raw,
            "formula": f"{A}-{B} x={x:.2f} z={z:.2f}",
            "PCE_max (%)": round(pce * 100, 1),
        })

    if not rows:
        return pd.DataFrame()
    m = max(r["raw"] for r in rows) or 1.0
    for r in rows:
        r["score"] = round(r.pop("raw") / m, 3)

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


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
    doping_element: str = "Ge",  # Added doping element here
) -> pd.DataFrame:
    lo, hi = bg
    center = sigma = None
    if application in APPLICATION_CONFIG:
        cfg = APPLICATION_CONFIG[application]
        lo, hi = cfg["range"]
        center, sigma = cfg["center"], cfg["sigma"]

    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    # Apply doping element logic if z > 0
    if doping_element != "Ge":
        A = A.replace("Sn", doping_element)
        B = B.replace("Sn", doping_element)
        C = C.replace("Sn", doping_element)

    oxA, oxB, oxC = (oxidation_energy(f) for f in (A, B, C))
    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        for y in np.arange(0.0, 1.0 - x + 1e-9, dy):
            w = 1.0 - x - y
            # Sn gap
            Eg_Sn = (
                w * dA["band_gap"] + x * dB["band_gap"] + y * dC["band_gap"]
                - bows["AB"] * x * w - bows["AC"] * y * w - bows["BC"] * x * y
            )
            # Ge gap
            Eg_Ge = (
                w * dA["band_gap"] + x * dB["band_gap"] + y * dC["band_gap"]
                - bows["AB"] * x * w - bows["AC"] * y * w - bows["BC"] * x * y
            )
            Eg = (1.0 - z) * Eg_Sn + z * Eg_Ge

            Eh_Sn = (
                w * dA["energy_above_hull"] + x * dB["energy_above_hull"] + y * dC["energy_above_hull"]
            )
            Eh_Ge = (
                w * dA["energy_above_hull"] + x * dB["energy_above_hull"] + y * dC["energy_above_hull"]
            )
            Eh = (1.0 - z) * Eh_Sn + z * Eh_Ge

            dEox = w * oxA + x * oxB + y * oxC
            sbg = _score_band_gap(Eg, lo, hi, center, sigma)
            raw = sbg * math.exp(-Eh / 0.0518) * math.exp(dEox / K_T_EFF)

            # Shockley–Queisser PCE limit
            pce = sq_efficiency(Eg)

            rows.append({
                "x": round(x, 3),
                "y": round(y, 3),
                "z": round(z, 2),
                "Eg": round(Eg, 3),
                "Ehull": round(Eh, 4),
                "Eox": round(dEox, 3),
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

