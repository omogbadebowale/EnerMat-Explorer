# ===============================
# File: backend.py
# ===============================
from __future__ import annotations
import math
import os
from functools import lru_cache
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from mp_api.client import MPRester
from pymatgen.core import Composition

# ─────────── Shockley–Queisser helper ───────────
# Make sure you have backend/sq.py with `def sq_efficiency(Eg: float) -> float: ...`
from backend.sq import sq_efficiency

# ─────────── Constants ───────────
# Boltzmann constant in eV/K
K_B = 8.617333262e-5

# Base scale (soft) for oxidation weighting in the score (eV)
K_T_EFF_BASE = 0.20

# Humidity coupling [eV per Sn] that shifts ΔE_ox downward at high RH
# (so high humidity reduces oxidation advantage and increases penalty)
LAMBDA_RH = 0.30

# Ionic radii (Shannon-style) used for tolerance factor evaluation
IONIC_RADII = {
    # A-site cations
    "Cs": 1.88, "FA": 2.79, "MA": 2.70,
    # B-site cations
    "Sn": 1.18, "Ge": 0.73, "Pb": 1.31,
    "Bi": 1.03, "Sb": 0.76, "Ag": 1.15, "In": 0.81,
    # X-site anions
    "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

# Gap calibration offsets to align MP/PBE-like gaps with experiment (eV)
# (Aligned with manuscript text)
GAP_OFFSET = {"I": 0.52, "Br": 0.88, "Cl": 1.10}

# Application presets (can be overridden by user if `manual_eg=True`)
APPLICATION_CONFIG = {
    "single":  {"range": (1.10, 1.40), "center": 1.25, "sigma": 0.10},
    "tandem":  {"range": (1.60, 1.90), "center": 1.75, "sigma": 0.10},
    "indoor":  {"range": (1.70, 2.20), "center": 1.95, "sigma": 0.15},
    "detector": {"range": (0.80, 3.00), "center": None,  "sigma": None},
}

# Reference end-members (formula strings)
END_MEMBERS = [
    # classic Sn / Ge
    "CsSnI3", "CsSnBr3", "CsSnCl3",
    "CsGeBr3", "CsGeCl3",
    # organic Sn
    "FASnI3", "MASnBr3",
    # vacancy-ordered
    "Cs2SnI6",
    # layered Bi / Sb
    "Cs3Bi2Br9", "Cs3Sb2I9",
    # double perovskites
    "Cs2AgBiBr6", "Cs2AgInCl6",
    # Pb references
    "CsPbCl3", "CsPbBr3", "CsPbI3",
]

# Optional curated experimental gaps: set when available; otherwise offsets will be used.
CALIBRATED_GAPS: Dict[str, float] = {
    # Example (fill if you have curated values):
    # "CsSnI3": 1.30,
    # "CsSnBr3": 1.75,
    # "CsSnCl3": 3.10,
}

# ─────────── API key ───────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32-character MP_API_KEY missing")

mpr = MPRester(API_KEY)

# ─────────── Utility: parsing sites from formula ───────────
A_SITE_TAGS = ("Cs", "FA", "MA")
HALS = ("I", "Br", "Cl")
B_SITE_TAGS = ("Sn", "Ge", "Pb")


def parse_abx3_sites(formula: str) -> Tuple[str, str, str]:
    """Best-effort parse for simple ABX3-like formulas to extract (A, B, X).
    Returns (A, B, X). If not found, raises ValueError.
    """
    A = next((a for a in A_SITE_TAGS if a in formula), None)
    B = next((b for b in B_SITE_TAGS if b in formula), None)
    X = next((x for x in HALS if x in formula), None)
    if not (A and B and X):
        raise ValueError(f"Cannot parse A/B/X from {formula}")
    return A, B, X


# ─────────── Data fetch & provenance ───────────

def _apply_gap_correction(formula: str, band_gap: Optional[float]) -> Tuple[float, str]:
    """Return corrected gap and a short note about the correction route used."""
    if formula in CALIBRATED_GAPS:
        return CALIBRATED_GAPS[formula], "calibrated"
    # Fall back to halide offset
    _, _, X = parse_abx3_sites(formula)
    corrected = (band_gap or 0.0) + GAP_OFFSET.get(X, 0.0)
    return corrected, f"offset+{X}"


def fetch_mp_data(formula: str, fields: List[str]):
    docs = mpr.summary.search(formula=formula, fields=tuple(set(fields + ["material_id", "last_updated", "band_gap"])))
    if not docs:
        return None
    ent = docs[0]
    out = {f: getattr(ent, f, None) for f in fields if hasattr(ent, f)}
    # store provenance
    out["material_id"] = getattr(ent, "material_id", None)
    out["last_updated"] = getattr(ent, "last_updated", None)

    if "band_gap" in fields:
        corrected, route = _apply_gap_correction(formula, getattr(ent, "band_gap", None))
        out["band_gap"] = corrected
        out["gap_route"] = route
    return out


@lru_cache(maxsize=64)
def oxidation_energy(formula_sn2: str) -> float:
    """ΔE_ox per Sn for CsSnX3 + 1/2 O2 -> 1/2 (Cs2SnX6 + SnO2). Positive → harder to oxidize.
    Uses 0 K formation energies from Materials Project.
    """
    if "Sn" not in formula_sn2:
        return 0.0
    hal = next((h for h in HALS if h in formula_sn2), None)
    if hal is None:
        return 0.0

    def formation_energy_fu(formula: str) -> float:
        doc = fetch_mp_data(formula, ["formation_energy_per_atom"])
        if not doc or doc.get("formation_energy_per_atom") is None:
            raise ValueError(f"Missing formation-energy for {formula}")
        comp = Composition(formula)
        return float(doc["formation_energy_per_atom"]) * comp.num_atoms

    H_reac = formation_energy_fu(formula_sn2)
    H_prod1 = formation_energy_fu(f"Cs2Sn{hal}6")
    H_prod2 = formation_energy_fu("SnO2")
    # 1/2 * [(Cs2SnX6) + (SnO2)] - (CsSnX3)
    return 0.5 * (H_prod1 + H_prod2) - H_reac


# ─────────── Scoring helpers ───────────

def _score_band_gap(Eg: float, lo: float, hi: float, center: Optional[float], sigma: Optional[float]) -> float:
    if Eg < lo or Eg > hi:
        return 0.0
    if center is None or sigma is None:
        return 1.0
    # Gaussian weighting
    return math.exp(-((Eg - center) ** 2) / (2 * sigma * sigma))


score_band_gap = _score_band_gap  # alias


# Tolerance factor and structural penalty

def tolerance_factor(rA: float, rB: float, rX: float) -> float:
    return (rA + rX) / (math.sqrt(2.0) * (rB + rX))


def structural_penalty(t: float, t0: float = 0.95, beta: float = 1.0) -> float:
    # Exponential penalty around a preferred t0 (beta controls sharpness)
    return math.exp(-beta * abs(t - t0))


# Resolve band-gap window (manual override vs preset)

def resolve_bg_window(application: Optional[str], user_lohi: Tuple[float, float], manual_eg: bool) -> Tuple[float, float, Optional[float], Optional[float]]:
    if manual_eg or not application or application not in APPLICATION_CONFIG:
        lo, hi = user_lohi
        return lo, hi, None, None  # no Gaussian center/sigma unless you want to set them manually too
    cfg = APPLICATION_CONFIG[application]
    lo, hi = cfg["range"]
    return lo, hi, cfg["center"], cfg["sigma"]


# ─────────── Binary screen ───────────

def screen_binary(
    A: str,
    B: str,
    rh: float,
    temp_c: float,
    bg: Tuple[float, float],
    bow: float,
    dx: float,
    *,
    z: float = 0.0,
    application: Optional[str] = None,
    manual_eg: bool = False,
    alpha: float = 1.0,
    beta_struct: float = 1.0,
) -> pd.DataFrame:
    # Resolve Eg window and Gaussian settings
    lo, hi, center, sigma = resolve_bg_window(application, bg, manual_eg)

    # Fetch end-member data
    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])  # also returns material_id, gap_route
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])  # also returns material_id, gap_route
    if not (dA and dB):
        return pd.DataFrame()

    # Optional Ge branch (binary)
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

    # Parse sites and radii for structural factor
    A_A, A_B, A_X = parse_abx3_sites(A)
    B_A, B_B, B_X = parse_abx3_sites(B)

    # Composition weighting for radii (binary along x)
    rB_SnGe = (1.0 - z) * IONIC_RADII["Sn"] + z * IONIC_RADII["Ge"]

    rows: List[Dict] = []
    T = temp_c + 273.15
    kT = K_B * T

    # Pre-compute oxidation for Sn branches
    oxA = oxidation_energy(A)
    oxB = oxidation_energy(B)

    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        # Linear + bowing for gaps (Sn and Ge branches)
        Eg_Sn = (1 - x) * float(dA["band_gap"]) + x * float(dB["band_gap"]) - bow * x * (1 - x)
        Eh_Sn = (1 - x) * float(dA["energy_above_hull"]) + x * float(dB["energy_above_hull"])
        dEox_Sn = (1 - x) * oxA + x * oxB

        Eg_Ge = (1 - x) * float(dA_Ge["band_gap"]) + x * float(dB_Ge["band_gap"]) - bow * x * (1 - x)
        Eh_Ge = (1 - x) * float(dA_Ge["energy_above_hull"]) + x * float(dB_Ge["energy_above_hull"])
        dEox_Ge = (1 - x) * oxA_Ge + x * oxB_Ge

        # Interpolate with Ge fraction z
        Eg = (1.0 - z) * Eg_Sn + z * Eg_Ge
        Eh = (1.0 - z) * Eh_Sn + z * Eh_Ge
        dEox = (1.0 - z) * dEox_Sn + z * dEox_Ge

        # Environment coupling for oxidation (humidity shifts ΔE_ox downward)
        RH_norm = float(rh) / 100.0
        dEox_env = dEox - LAMBDA_RH * RH_norm

        # Scores
        sbg = _score_band_gap(Eg, lo, hi, center, sigma)
        sEh = math.exp(-Eh / max(1e-9, (alpha * kT)))
        sOx = math.exp(dEox_env / K_T_EFF_BASE)

        # Structural factor via composition-weighted radii
        rA = (1 - x) * IONIC_RADII[A_A] + x * IONIC_RADII[B_A]
        rX = (1 - x) * IONIC_RADII[A_X] + x * IONIC_RADII[B_X]
        t = tolerance_factor(rA, rB_SnGe, rX)
        sStr = structural_penalty(t, t0=0.95, beta=beta_struct)

        raw = sbg * sEh * sOx * sStr
        pce = sq_efficiency(Eg)

        rows.append({
            "x": round(x, 3),
            "z": round(z, 2),
            "Eg": round(Eg, 3),
            "Ehull": round(Eh, 4),
            "Eox": round(dEox, 3),
            "Eox_env": round(dEox_env, 3),
            "t_factor": round(t, 3),
            "PCE_max (%)": round(pce * 100, 1),
            "raw": raw,
            "formula": f"{A}-{B} x={x:.2f} z={z:.2f}",
            # provenance (same for all rows in a run but useful in a Provenance tab)
            "A_mpid": dA.get("material_id"),
            "B_mpid": dB.get("material_id"),
            "A_gap_route": dA.get("gap_route"),
            "B_gap_route": dB.get("gap_route"),
            "T_K": round(T, 1),
            "RH_%": int(rh),
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


# ─────────── Ternary screen ───────────

def screen_ternary(
    A: str,
    B: str,
    C: str,
    rh: float,
    temp_c: float,
    bg: Tuple[float, float],
    bows: Dict[str, float],
    *,
    dx: float = 0.10,
    dy: float = 0.10,
    z: float = 0.0,
    application: Optional[str] = None,
    manual_eg: bool = False,
    alpha: float = 1.0,
    beta_struct: float = 1.0,
) -> pd.DataFrame:
    lo, hi, center, sigma = resolve_bg_window(application, bg, manual_eg)

    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    # Ge branches if requested
    if z > 0:
        A_Ge = A.replace("Sn", "Ge")
        B_Ge = B.replace("Sn", "Ge")
        C_Ge = C.replace("Sn", "Ge")
        dA_Ge = fetch_mp_data(A_Ge, ["band_gap", "energy_above_hull"]) or dA
        dB_Ge = fetch_mp_data(B_Ge, ["band_gap", "energy_above_hull"]) or dB
        dC_Ge = fetch_mp_data(C_Ge, ["band_gap", "energy_above_hull"]) or dC
    else:
        dA_Ge, dB_Ge, dC_Ge = dA, dB, dC

    # Precompute oxidations
    oxA, oxB, oxC = (oxidation_energy(f) for f in (A, B, C))

    # Parse sites
    A_A, A_B, A_X = parse_abx3_sites(A)
    B_A, B_B, B_X = parse_abx3_sites(B)
    C_A, C_B, C_X = parse_abx3_sites(C)

    T = temp_c + 273.15
    kT = K_B * T
    RH_norm = float(rh) / 100.0

    # B-site mixed radius (Sn/Ge)
    rB_SnGe = (1.0 - z) * IONIC_RADII["Sn"] + z * IONIC_RADII["Ge"]

    rows: List[Dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        for y in np.arange(0.0, 1.0 - x + 1e-9, dy):
            w = 1.0 - x - y

            # Gaps
            Eg_Sn = (
                w * float(dA["band_gap"]) + x * float(dB["band_gap"]) + y * float(dC["band_gap"])
                - bows["AB"] * x * w - bows["AC"] * y * w - bows["BC"] * x * y
            )
            Eg_Ge = (
                w * float(dA_Ge["band_gap"]) + x * float(dB_Ge["band_gap"]) + y * float(dC_Ge["band_gap"])
                - bows["AB"] * x * w - bows["AC"] * y * w - bows["BC"] * x * y
            )
            Eg = (1.0 - z) * Eg_Sn + z * Eg_Ge

            # Ehull
            Eh_Sn = w * float(dA["energy_above_hull"]) + x * float(dB["energy_above_hull"]) + y * float(dC["energy_above_hull"])
            Eh_Ge = w * float(dA_Ge["energy_above_hull"]) + x * float(dB_Ge["energy_above_hull"]) + y * float(dC_Ge["energy_above_hull"])
            Eh = (1.0 - z) * Eh_Sn + z * Eh_Ge

            # Oxidation (composition-weighted, then humidity-shifted)
            dEox = w * oxA + x * oxB + y * oxC
            dEox_env = dEox - LAMBDA_RH * RH_norm

            # Scores
            sbg = _score_band_gap(Eg, lo, hi, center, sigma)
            sEh = math.exp(-Eh / max(1e-9, (alpha * kT)))  # temperature-coupled
            sOx = math.exp(dEox_env / K_T_EFF_BASE)

            # Structural factor: composition-weighted radii for A and X
            rA = w * IONIC_RADII[A_A] + x * IONIC_RADII[B_A] + y * IONIC_RADII[C_A]
            rX = w * IONIC_RADII[A_X] + x * IONIC_RADII[B_X] + y * IONIC_RADII[C_X]
            t = tolerance_factor(rA, rB_SnGe, rX)
            sStr = structural_penalty(t, t0=0.95, beta=beta_struct)

            pce = sq_efficiency(Eg)
            raw = sbg * sEh * sOx * sStr

            rows.append({
                "x": round(x, 3),
                "y": round(y, 3),
                "z": round(z, 2),
                "Eg": round(Eg, 3),
                "Ehull": round(Eh, 4),
                "Eox": round(dEox, 3),
                "Eox_env": round(dEox_env, 3),
                "t_factor": round(t, 3),
                "PCE_max (%)": round(pce * 100, 1),
                "raw": raw,
                "formula": f"{A}-{B}-{C} x={x:.2f} y={y:.2f} z={z:.2f}",
                # provenance
                "A_mpid": dA.get("material_id"),
                "B_mpid": dB.get("material_id"),
                "C_mpid": dC.get("material_id"),
                "A_gap_route": dA.get("gap_route"),
                "B_gap_route": dB.get("gap_route"),
                "C_gap_route": dC.get("gap_route"),
                "T_K": round(T, 1),
                "RH_%": int(rh),
            })

    if not rows:
        return pd.DataFrame()

    m = max(r["raw"] for r in rows) or 1.0
    for r in rows:
        r["score"] = round(r.pop("raw") / m, 3)

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
