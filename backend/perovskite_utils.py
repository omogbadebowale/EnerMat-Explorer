from __future__ import annotations
import math
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# streamlit may not be present on some backends; make optional
try:
    import streamlit as st
except Exception:  # pragma: no cover
    class _Dummy:  # minimal stub
        def __getattr__(self, k): return {}
    st = _Dummy()  # type: ignore

# Materials Project (optional; if key missing we fall back gracefully)
try:
    from mp_api.client import MPRester
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

from pymatgen.core import Composition

# ─────────── Shockley–Queisser helper ───────────
# Ensure you have backend/sq.py with: def sq_efficiency(Eg: float) -> float
from backend.sq import sq_efficiency

# ─────────── API key / MPRester ───────────
load_dotenv()
API_KEY = (os.getenv("MP_API_KEY") or getattr(st, "secrets", {}).get("MP_API_KEY") or "").strip()

mpr = None
if MP_AVAILABLE and len(API_KEY) == 32:
    try:
        mpr = MPRester(API_KEY)
    except Exception:
        mpr = None

# ─────────── application-based band-gap targets ───────────
# Presets control the *preference* around Eg. Single has no Gaussian (SQ alone decides).
APPLICATION_CONFIG = {
    "single": {"range": (1.10, 1.40), "center": None,  "sigma": None},   # ← reproduces your Ge≈0.45 optimum
    "tandem": {"range": (1.60, 1.90), "center": 1.75,  "sigma": 0.10},
    "indoor": {"range": (1.70, 2.20), "center": 1.95,  "sigma": 0.15},
    "detector": {"range": (0.80, 3.00), "center": None, "sigma": None},
}

# ─────────── reference data ───────────
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
    # Pb references (kept for comparisons)
    "CsPbCl3", "CsPbBr3", "CsPbI3",
]

# If MP returns PBE gaps, we apply simple offsets when no calibrated value is known.
CALIBRATED_GAPS = {
    # put any curated values you have here; otherwise offsets will be used
    "CsSnI3": 1.30, "CsSnBr3": 1.80, "CsSnCl3": 3.00,
    "Cs2SnI6": 1.60, "FASnI3": 1.30, "Cs2AgBiBr6": 2.00, "Cs3Bi2Br9": 2.20,
}
GAP_OFFSET = {"I": 0.85, "Br": 0.78, "Cl": 2.00, "Pb": 1.31}

IONIC_RADII = {
    # A / B cations
    "Cs": 1.88, "FA": 2.79, "MA": 2.70,
    "Sn": 1.18, "Ge": 0.73, "Pb": 1.31, "Si": 0.54, "Mn": 0.83, "Zn": 0.74,
    "Bi": 1.03, "Sb": 0.76, "Ag": 1.15, "In": 0.81,
    # X anions
    "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

# Softness scales
K_T_EFF  = 0.20     # oxidation softness (eV)
K_T_HULL = 0.0259   # Ehull softness (~kT at 300 K per atom)

# ─────────── band-gap preference (Gaussian, optional) ───────────
def _score_band_gap(Eg: float, lo: float, hi: float, center: float | None, sigma: float | None) -> float:
    # If a hard window is given and Eg sits outside, return 0 to de-emphasize those points
    if lo is not None and hi is not None and (Eg < lo or Eg > hi):
        return 0.0
    if center is None or sigma is None or sigma <= 0:
        return 1.0
    return math.exp(-((Eg - center) ** 2) / (2.0 * sigma * sigma))

# ─────────── helpers ───────────
def _infer_halide(formula: str) -> str | None:
    for h in ("I", "Br", "Cl"):
        if h in formula:
            return h
    return None

def fetch_mp_data(formula: str, fields: list[str]):
    """
    Pull minimal data from MP; if missing, return calibrated/offset values.
    """
    out = {f: None for f in fields}
    # Online path
    if mpr is not None:
        try:
            docs = mpr.summary.search(formula=formula, fields=tuple(set(fields)))
            if docs:
                ent = docs[0]
                for f in fields:
                    out[f] = getattr(ent, f, None)
        except Exception:
            pass

    # Gap handling
    if "band_gap" in fields:
        if CALIBRATED_GAPS.get(formula) is not None:
            out["band_gap"] = CALIBRATED_GAPS[formula]
        else:
            hal = _infer_halide(formula)
            base = float(out.get("band_gap") or 0.0)
            out["band_gap"] = base + (GAP_OFFSET.get(hal, 0.0) if hal else 0.0)

    return out

@lru_cache(maxsize=64)
def oxidation_energy(formula_sn2: str) -> float:
    """ΔEₒₓ per Sn for CsSnX₃ + ½ O₂ → ½ (Cs₂SnX₆ + SnO₂)."""
    if "Sn" not in formula_sn2:
        return 0.0
    hal = _infer_halide(formula_sn2)
    if hal is None:
        return 0.0

    def formation_energy_fu(formula: str) -> float:
        doc = fetch_mp_data(formula, ["formation_energy_per_atom"])
        if not doc or doc.get("formation_energy_per_atom") is None:
            return 0.0  # fall back gracefully
        comp = Composition(formula)
        return float(doc["formation_energy_per_atom"]) * comp.num_atoms

    H_reac  = formation_energy_fu(formula_sn2)
    H_prod1 = formation_energy_fu(f"Cs2Sn{hal}6")
    H_prod2 = formation_energy_fu("SnO2")
    return 0.5 * (H_prod1 + H_prod2) - H_reac

# ─────────── geometry / tolerance penalty ───────────
def _tolerance_penalty(rB: float, halide: str, t0: float = 0.95, beta: float = 1.0, A: str = "Cs") -> float:
    rA = IONIC_RADII.get(A, 1.88)
    rX = IONIC_RADII.get(halide, 2.0)
    t  = (rA + rX) / (math.sqrt(2.0) * (rB + rX))
    return math.exp(-beta * abs(t - t0))

def _b_radius(z: float, dopant: str | None = "Ge") -> float:
    r_sn = IONIC_RADII["Sn"]
    if not dopant or dopant == "None" or z <= 0.0:
        return r_sn
    r_d  = IONIC_RADII.get(dopant, r_sn)
    return (1.0 - z) * r_sn + z * r_d

# ─────────── Binary screen ───────────
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
    dopant_element: str | None = "Ge",
    t0: float = 0.95,
    beta: float = 1.0,
    **kwargs,  # swallow extra args to avoid UI/backend mismatches
) -> pd.DataFrame:
    # Resolve application preset (overrides slider to keep results consistent)
    app = (application or "single").lower()
    lo, hi = bg
    center = sigma = None
    if app in APPLICATION_CONFIG:
        cfg = APPLICATION_CONFIG[app]
        if cfg["range"]:
            lo, hi = cfg["range"]
        center, sigma = cfg["center"], cfg["sigma"]

    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    # Optional dopant branch (isovalent assumption for gap trend)
    dA_D = fetch_mp_data(A.replace("Sn", dopant_element or "Ge"), ["band_gap", "energy_above_hull"]) if z > 0 else dA
    dB_D = fetch_mp_data(B.replace("Sn", dopant_element or "Ge"), ["band_gap", "energy_above_hull"]) if z > 0 else dB

    hal = _infer_halide(A) or _infer_halide(B) or "I"
    rB  = _b_radius(z, dopant_element or "Ge")
    tol_pen_const = _tolerance_penalty(rB, hal, t0=t0, beta=beta)

    oxA, oxB = oxidation_energy(A), oxidation_energy(B)

    rows: list[dict] = []
    EA, EB = float(dA["band_gap"]), float(dB["band_gap"])
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        # Sn branch
        Eg_Sn   = (1 - x) * EA + x * float(dB["band_gap"]) - bow * x * (1 - x)
        Eh_Sn   = (1 - x) * float(dA["energy_above_hull"]) + x * float(dB["energy_above_hull"])
        dEox_Sn = (1 - x) * oxA + x * oxB

        # Dopant branch (gap/hull only)
        Eg_D    = (1 - x) * float(dA_D["band_gap"]) + x * float(dB_D["band_gap"]) - bow * x * (1 - x)
        Eh_D    = (1 - x) * float(dA_D["energy_above_hull"]) + x * float(dB_D["energy_above_hull"])

        # Interpolate with z
        Eg   = (1.0 - z) * Eg_Sn + z * Eg_D
        Eh   = (1.0 - z) * Eh_Sn + z * Eh_D
        dEox = dEox_Sn  # oxidation proxy from Sn branch

        # Performance: SQ × (optional Gaussian preference by application)
        perf = sq_efficiency(Eg)
        g    = _score_band_gap(Eg, lo, hi, center, sigma)

        raw = perf * g * math.exp(-Eh / K_T_HULL) * math.exp(dEox / K_T_EFF) * tol_pen_const

        rows.append({
            "x": round(x, 3),
            "z": round(z, 2),
            "Eg": round(Eg, 3),
            "Ehull": round(Eh, 4),
            "Eox_e": round(dEox, 3),
            "PCE_max (%)": round(perf * 100.0, 1),
            "raw_score": raw,  # keep raw
            "formula": f"{A}-{B} x={x:.2f} z={z:.2f}",
        })

    if not rows:
        return pd.DataFrame()

    m = max(r["raw_score"] for r in rows) or 1.0
    for r in rows:
        r["score"] = round(r["raw_score"] / m, 3)

    keep = ["formula","x","z","Eg","Ehull","Eox_e","PCE_max (%)","raw_score","score"]
    return pd.DataFrame(rows)[keep].sort_values("score", ascending=False).reset_index(drop=True)

# ─────────── Ternary screen ───────────
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
    dopant_element: str | None = "Ge",
    t0: float = 0.95,
    beta: float = 1.0,
    **kwargs,  # swallow extra args
) -> pd.DataFrame:
    app = (application or "single").lower()
    lo, hi = bg
    center = sigma = None
    if app in APPLICATION_CONFIG:
        cfg = APPLICATION_CONFIG[app]
        if cfg["range"]:
            lo, hi = cfg["range"]
        center, sigma = cfg["center"], cfg["sigma"]

    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    # Optional dopant (gap/hull) branches
    dA_D = fetch_mp_data(A.replace("Sn", dopant_element or "Ge"), ["band_gap", "energy_above_hull"]) if z > 0 else dA
    dB_D = fetch_mp_data(B.replace("Sn", dopant_element or "Ge"), ["band_gap", "energy_above_hull"]) if z > 0 else dB
    dC_D = fetch_mp_data(C.replace("Sn", dopant_element or "Ge"), ["band_gap", "energy_above_hull"]) if z > 0 else dC

    halA = _infer_halide(A) or "I"
    rB   = _b_radius(z, dopant_element or "Ge")
    tol_pen_const = _tolerance_penalty(rB, halA, t0=t0, beta=beta)

    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        for y in np.arange(0.0, 1.0 - x + 1e-9, dy):
            w = 1.0 - x - y
            # gaps
            Eg_Sn = (
                w * float(dA["band_gap"]) + x * float(dB["band_gap"]) + y * float(dC["band_gap"])
                - bows["AB"] * x * w - bows["AC"] * y * w - bows["BC"] * x * y
            )
            Eg_D = (
                w * float(dA_D["band_gap"]) + x * float(dB_D["band_gap"]) + y * float(dC_D["band_gap"])
                - bows["AB"] * x * w - bows["AC"] * y * w - bows["BC"] * x * y
            )
            Eg = (1.0 - z) * Eg_Sn + z * Eg_D

            # hull
            Eh_Sn = w * float(dA["energy_above_hull"]) + x * float(dB["energy_above_hull"]) + y * float(dC["energy_above_hull"])
            Eh_D  = w * float(dA_D["energy_above_hull"]) + x * float(dB_D["energy_above_hull"]) + y * float(dC_D["energy_above_hull"])
            Eh = (1.0 - z) * Eh_Sn + z * Eh_D

            # oxidation proxy (from end-members)
            oxA, oxB, oxC = oxidation_energy(A), oxidation_energy(B), oxidation_energy(C)
            dEox = w * oxA + x * oxB + y * oxC

            perf = sq_efficiency(Eg)
            g    = _score_band_gap(Eg, lo, hi, center, sigma)
            raw  = perf * g * math.exp(-Eh / K_T_HULL) * math.exp(dEox / K_T_EFF) * tol_pen_const

            rows.append({
                "x": round(x, 3), "y": round(y, 3),
                "z": round(z, 2),
                "Eg": round(Eg, 3),
                "Ehull": round(Eh, 4),
                "Eox_e": round(dEox, 3),
                "PCE_max (%)": round(perf * 100.0, 1),
                "raw_score": raw,
                "formula": f"{A}-{B}-{C} x={x:.2f} y={y:.2f} z={z:.2f}",
            })

    if not rows:
        return pd.DataFrame()

    m = max(r["raw_score"] for r in rows) or 1.0
    for r in rows:
        r["score"] = round(r["raw_score"] / m, 3)

    keep = ["formula","x","y","z","Eg","Ehull","Eox_e","PCE_max (%)","raw_score","score"]
    return pd.DataFrame(rows)[keep].sort_values("score", ascending=False).reset_index(drop=True)
