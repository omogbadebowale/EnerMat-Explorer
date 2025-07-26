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
# Make sure you have backend/sq.py with `def sq_efficiency(Eg: float) -> float: ...`
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
END_MEMBERS = ["CsSnI3", "CsSnBr3", "CsSnCl3", "CsGeBr3", "CsGeCl3",  "CsPbCl3", "CsPbBr3", "CsPbI3"]

CALIBRATED_GAPS = {
    "CsSnBr3": 1.75,
    "CsSnCl3": 2.98,
    "CsSnI3":  1.00,
    "CsGeBr3": 2.32,
    "CsGeCl3": 3.67,
    "CsPbI3": 1.68,
    "CsPbBr3": 2.36,
    "CsPbCl3": 3.03,

}

GAP_OFFSET = {"I": 1.3-0.45, "Br": 1.75-0.97, "Cl": 2.98-0.98, "Pb": 1.31, }
IONIC_RADII = {"Cs": 1.88, "Sn": 1.18, "Ge": 0.73,
               "I": 2.20, "Br": 1.96, "Cl": 1.81, "Pb": 1.31, }

K_T_EFF = 0.20  # soft-penalty “kT” (eV)

# ─────────── NEW: surface‑passivation constants ───────────
K_B_EV: float = 8.617_333e-5  # eV K⁻¹
T_REF  : float = 300.0        # reference temperature (K)

def s_oxsurf(eox_sn: float, eox_ge: float, T: float = T_REF) -> float:
    """Boltzmann factor favouring Ge‑first oxidation."""
    return math.exp(-(eox_ge - eox_sn) / (K_B_EV * T))

# ─────────── band-gap scoring ───────────
def _score_band_gap(
    Eg: float,
    lo: float, hi: float,
    center: float | None,
    sigma: float | None
) -> float:
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
    """ΔEₒₓ per Sn for CsSnX₃ + ½ O₂ → ½ (Cs₂SnX₆ + SnO₂)."""
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

# ─────────── binary screen wrappers ───────────

def screen_binary(A: str, B: str, rh: float, temp: float,
                  bg: tuple[float, float], bow: float, dx: float,
                  *, z: float = 0.0,
                  application: str | None = None) -> pd.DataFrame:
    lo, hi = bg
    center = sigma = None
    if application in APPLICATION_CONFIG:
        cfg = APPLICATION_CONFIG[application]
        lo, hi = cfg["range"]
        center, sigma = cfg["center"], cfg["sigma"]

    return mix_abx3(A, B, rh, temp, (lo, hi), bow, dx,
                    z=z, center=center, sigma=sigma)

# ─────────── core binary mix function ───────────

def mix_abx3(A: str, B: str, rh: float, temp: float,
             bg: tuple[float, float], bow: float, dx: float,
             *, z: float = 0.0,
             alpha: float = 1.0, beta: float = 1.0,
             center: float | None = None, sigma: float | None = None) -> pd.DataFrame:
    lo, hi = bg
    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    # optional Ge branch (for z > 0)
    if z > 0:
        A_Ge = A.replace("Sn", "Ge")
        B_Ge = B.replace("Sn", "Ge")
        dA_Ge = fetch_mp_data(A_Ge, ["band_gap", "energy_above_hull"]) or dA
        dB_Ge = fetch_mp_data(B_Ge, ["band_gap", "energy_above_hull"]) or dB
        oxA_Ge = oxidation_energy(A_Ge)
        oxB_Ge = oxidation_energy(B_Ge)
    else:
        dA_Ge, dB_Ge = dA, dB
        oxA_Ge, oxB_Ge = oxidation_energy(A), oxidation_energy(B)

    hal = next(h for h in ("I", "Br", "Cl") if h in A)
    rA, rB, rX = (IONIC_RADII[k] for k in ("Cs", "Sn", hal))
    oxA, oxB = oxidation_energy(A), oxidation_energy(B)

    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        # Sn branch properties
        Eg_Sn   = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bow * x * (1 - x)
        Eh_Sn   = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        dEox_Sn = (1 - x) * oxA + x * oxB

        # Ge branch properties
        Eg_Ge   = (1 - x) * dA_Ge["band_gap"] + x * dB_Ge["band_gap"] - bow * x * (1 - x)
        Eh_Ge   = (1 - x) * dA_Ge["energy_above_hull"] + x * dB_Ge["energy_above_hull"]
        dEox_Ge = (1 - x) * oxA_Ge + x * oxB_Ge

        # Interpolate Sn↔Ge according to z
        Eg   = (1.0 - z) * Eg_Sn   + z * Eg_Ge
        Eh   = (1.0 - z) * Eh_Sn   + z * Eh_Ge
        dEox = (1.0 - z) * dEox_Sn + z * dEox_Ge

        # New surface‑passivation factor
        S_oxsurf = s_oxsurf(dEox_Sn, dEox_Ge)

        sbg = _score_band_gap(Eg, lo, hi, center, sigma)
        raw = (
            sbg
            * math.exp(-Eh / (alpha * 0.0259))
            * math.exp(dEox / K_T_EFF)
            * math.exp(-beta * abs((rA + rX) / (math.sqrt(2) * (rB + rX)) - 0.95))
            * S_oxsurf
        )

        pce = sq_efficiency(Eg)

        rows.append({
            "x": round(x, 3),
            "z": round(z, 2),
            "Eg": round(Eg, 3),
            "Ehull": round(Eh, 4),
            "Eox": round(dEox, 3),
            "S_oxsurf": round(S_oxsurf, 2),
            "raw": raw,
            "formula": f"{A}-{B} x={x:.2f} z={z:.2f}",
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

    # Ge-branch setup
    if z > 0:
        A_Ge = A.replace("Sn", "Ge")
        B_Ge = B.replace("Sn", "Ge")
        C_Ge = C.replace("Sn", "Ge")
        dA_Ge = fetch_mp_data(A_Ge, ["band_gap", "energy_above_hull"]) or dA
        dB_Ge = fetch_mp_data(B_Ge, ["band_gap", "energy_above_hull"]) or dB
        dC_Ge = fetch_mp_data(C_Ge, ["band_gap", "energy_above_hull"]) or dC
    else:
        dA_Ge, dB_Ge, dC_Ge = dA, dB, dC

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
                w * dA_Ge["band_gap"] + x * dB_Ge["band_gap"] + y * dC_Ge["band_gap"]
                - bows["AB"] * x * w - bows["AC"] * y * w - bows["BC"] * x * y
            )
            Eg = (1.0 - z) * Eg_Sn + z * Eg_Ge

            Eh_Sn = (
                w * dA["energy_above_hull"] + x * dB["energy_above_hull"] + y * dC["energy_above_hull"]
            )
            Eh_Ge = (
                w * dA_Ge["energy_above_hull"] + x * dB_Ge["energy_above_hull"] + y * dC_Ge["energy_above_hull"]
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
