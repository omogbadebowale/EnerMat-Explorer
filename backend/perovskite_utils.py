from __future__ import annotations
import json
import math
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Streamlit is optional on backend; guard imports for non-UI contexts
try:
    import streamlit as st
except Exception:  # pragma: no cover
    class _Dummy:
        def __getattr__(self, k): return {}
    st = _Dummy()  # type: ignore

# ─────────── Shockley–Queisser helper ───────────
# Keep your existing backend/sq.py implementation
from backend.sq import sq_efficiency

# ─────────── API key / MP client (graceful) ───────────
load_dotenv()
_API_KEY = (os.getenv("MP_API_KEY") or getattr(st, "secrets", {}).get("MP_API_KEY") or "").strip()

mpr = None
offline_mode = False
try:
    if len(_API_KEY) == 32:
        # Lazy import only when key is available
        from mp_api.client import MPRester
        mpr = MPRester(_API_KEY)
    else:
        offline_mode = True
except Exception:
    offline_mode = True

# ─────────── Configuration & reference data ───────────
APPLICATION_CONFIG = {
    "single":   {"range": (1.10, 1.40), "center": 1.25, "sigma": 0.10},
    "tandem":   {"range": (1.60, 1.90), "center": 1.75, "sigma": 0.10},
    "indoor":   {"range": (1.70, 2.20), "center": 1.95, "sigma": 0.15},
    "detector": {"range": (0.80, 3.00), "center": None,  "sigma": None},
}

# Presets for convenience (users may still type custom formulas)
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
    # Pb references (optional)
    "CsPbCl3", "CsPbBr3", "CsPbI3",
]

# Optional: load calibrated gaps (e.g., experimental) from secrets or file
def _load_calibrated_gaps() -> dict[str, float]:
    # 1) Streamlit secrets stringified JSON
    j = getattr(st, "secrets", {}).get("CALIBRATED_GAPS_JSON")
    if j:
        try:
            return json.loads(j)
        except Exception:
            pass
    # 2) Local JSON file path via env
    path = os.getenv("CALIBRATED_GAPS_PATH")
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # 3) Fallback minimal internal examples (extend as needed)
    return {
        "CsSnI3": 1.30,
        "CsSnBr3": 1.80,
        "CsSnCl3": 3.00,
        "FASnI3": 1.30,
        "Cs2SnI6": 1.60,
        "Cs3Bi2Br9": 2.20,
        "Cs2AgBiBr6": 2.00,
    }

CALIBRATED_GAPS = _load_calibrated_gaps()

# Simple halide-based offsets (used when CALIBRATED_GAPS lacks a value)
GAP_OFFSET = {"I": 0.85, "Br": 0.78, "Cl": 2.00, "Pb": 1.31}

# Shannon-like radii (coarse) for tolerance-factor penalty
IONIC_RADII = {
    # A / B cations
    "Cs": 1.88, "FA": 2.79, "MA": 2.70,
    "Sn": 1.18, "Ge": 0.73, "Pb": 1.31,
    "Bi": 1.03, "Sb": 0.76, "Ag": 1.15, "In": 0.81,
    # X anions
    "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

# “Soft” energy scales for score exponents
K_T_EFF = 0.20  # eV for oxidation reward
K_T_HULL = 0.0259  # ~room-temp kT used in stability penalty

# Default pseudo-uncertainties (used for score bounds)
DEFAULT_ERRS = {
    "Eg": 0.10,     # eV
    "Ehull": 0.01,  # eV/atom
}

# ─────────── Band-gap score ───────────
def _score_band_gap(Eg: float, lo: float, hi: float, center: float | None, sigma: float | None) -> float:
    if Eg < lo or Eg > hi:
        return 0.0
    if center is None or sigma is None:
        return 1.0
    return math.exp(-((Eg - center) ** 2) / (2 * sigma * sigma))

score_band_gap = _score_band_gap  # alias

# ─────────── Materials Project helpers (offline-safe) ───────────
# Minimal offline “summary” for demos if MP is unavailable
_OFFLINE_SUMMARY: dict[str, dict] = {
    "CsSnI3":     {"band_gap": CALIBRATED_GAPS.get("CsSnI3", 1.3), "energy_above_hull": 0.02},
    "CsSnBr3":    {"band_gap": CALIBRATED_GAPS.get("CsSnBr3", 1.8), "energy_above_hull": 0.01},
    "Cs2SnI6":    {"band_gap": CALIBRATED_GAPS.get("Cs2SnI6", 1.6), "energy_above_hull": 0.00},
    "Cs3Bi2Br9":  {"band_gap": CALIBRATED_GAPS.get("Cs3Bi2Br9", 2.2), "energy_above_hull": 0.04},
    "Cs2AgBiBr6": {"band_gap": CALIBRATED_GAPS.get("Cs2AgBiBr6", 2.0), "energy_above_hull": 0.00},
}

def _infer_halide(formula: str) -> str | None:
    for h in ("I", "Br", "Cl"):
        if h in formula:
            return h
    return None

def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    """
    Returns a dict with requested fields or None if nothing found.
    Applies CALIBRATED_GAPS first; otherwise adds GAP_OFFSET by halide.
    """
    # 1) Try online MP
    if (mpr is not None) and (not offline_mode):
        try:
            docs = mpr.summary.search(formula=formula, fields=tuple(set(fields)))
            if docs:
                ent = docs[0]
                out = {f: getattr(ent, f, None) for f in fields}
                # gap calibration
                if "band_gap" in fields:
                    if formula in CALIBRATED_GAPS:
                        out["band_gap"] = CALIBRATED_GAPS[formula]
                    else:
                        hal = _infer_halide(formula)
                        base = out.get("band_gap", 0.0) or 0.0
                        out["band_gap"] = base + (GAP_OFFSET.get(hal, 0.0) if hal else 0.0)
                return out
        except Exception:
            pass

    # 2) Offline fallback
    doc = _OFFLINE_SUMMARY.get(formula)
    if not doc:
        return None
    out = {f: doc.get(f) for f in fields}
    if "band_gap" in fields and out.get("band_gap") is None:
        hal = _infer_halide(formula)
        out["band_gap"] = (0.0) + (GAP_OFFSET.get(hal, 0.0) if hal else 0.0)
    return out

@lru_cache(maxsize=128)
def oxidation_energy(formula_sn2: str) -> float:
    """
    ΔE_ox per Sn for CsSnX3 + 1/2 O2 → 1/2 (Cs2SnX6 + SnO2).
    For non-Sn systems (Bi/Sb/double perovskites), returns 0.0 (neutral)
    until extended reactions are implemented.
    """
    if "Sn" not in formula_sn2:
        # TODO: extend with Bi/Sb/double perovskite oxidation routes.
        return 0.0

    hal = _infer_halide(formula_sn2)
    if hal is None:
        return 0.0

    def formation_energy_per_fu(formula: str) -> float:
        doc = fetch_mp_data(formula, ["formation_energy_per_atom"])
        if not doc or doc.get("formation_energy_per_atom") is None:
            # conservative neutral contribution
            return 0.0
        from pymatgen.core import Composition
        comp = Composition(formula)
        return float(doc["formation_energy_per_atom"]) * comp.num_atoms

    H_reac  = formation_energy_per_fu(formula_sn2)
    H_prod1 = formation_energy_per_fu(f"Cs2Sn{hal}6")
    H_prod2 = formation_energy_per_fu("SnO2")
    return 0.5 * (H_prod1 + H_prod2) - H_reac

# ─────────── Scoring utilities ───────────
def _environment_penalty(rh: float, temp_c: float, *, gamma_h: float, gamma_t: float) -> float:
    """
    Returns multiplicative penalty ∈ (0, 1] based on RH and Temp.
    If gammas are 0, penalty = 1 (no effect). Bounded & smooth.
    """
    # Normalize RH ∈ [0,1], Temp delta above 25°C (don’t penalize below)
    rh_n = max(0.0, min(1.0, rh / 100.0))
    dt   = max(0.0, temp_c - 25.0)
    pen_h = math.exp(-gamma_h * rh_n)
    pen_t = math.exp(-gamma_t * (dt / 50.0))  # ~50°C scale
    return max(0.0, min(1.0, pen_h * pen_t))

def _tolerance_penalty(A: str, B_site_radius: float, X: str, *, t0: float, beta: float) -> float:
    """
    Goldschmidt-like tolerance factor penalty centered at t0 with stiffness beta.
    Uses coarse ionic radii for simplicity.
    """
    rA = IONIC_RADII.get("Cs", 1.88)  # assume Cs unless user encodes MA/FA in formula
    rX = IONIC_RADII.get(X, 2.0)
    t  = (rA + rX) / (math.sqrt(2.0) * (B_site_radius + rX))
    return math.exp(-beta * abs(t - t0))

def _score_raw(Eg, Eh, dEox, sbg, env_pen, tol_pen, *, alpha=1.0) -> float:
    return (
        sbg
        * math.exp(-Eh / (alpha * K_T_HULL))
        * math.exp(dEox / K_T_EFF)
        * env_pen
        * tol_pen
    )

def _score_bounds(Eg, Eh, dEox, sbg_lo, sbg_hi, env_pen, tol_pen) -> tuple[float, float]:
    raw_lo = _score_raw(Eg, Eh + DEFAULT_ERRS["Ehull"], dEox, sbg_lo, env_pen, tol_pen)
    raw_hi = _score_raw(Eg, Eh - DEFAULT_ERRS["Ehull"], dEox, sbg_hi, env_pen, tol_pen)
    return raw_lo, raw_hi

# ─────────── Helpers ───────────
def _suggest_bowing(EA: float, EB: float, *, center: float | None) -> float | None:
    """
    Heuristic bowing suggestion so Eg(0.5) nudges toward target center.
    b ≈ ((EA+EB)/2 - center) / (0.25)
    """
    if center is None:
        return None
    return ((EA + EB) * 0.5 - center) / 0.25

def _b_site_radius(z: float) -> float:
    """Interpolated B-site radius for Sn/Ge mixing."""
    return (1.0 - z) * IONIC_RADII["Sn"] + z * IONIC_RADII["Ge"]

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
    use_bowing_suggestion: bool = False,
    gamma_h: float = 0.0,
    gamma_t: float = 0.0,
    t0: float = 0.95,
    beta: float = 1.0,
) -> pd.DataFrame:
    lo, hi = bg
    center = sigma = None
    if application in APPLICATION_CONFIG:
        cfg = APPLICATION_CONFIG[application]
        lo, hi = cfg["range"]
        center, sigma = cfg["center"], cfg["sigma"]

    return mix_abx3(
        A, B, rh, temp, (lo, hi), bow, dx,
        z=z, center=center, sigma=sigma,
        use_bowing_suggestion=use_bowing_suggestion,
        gamma_h=gamma_h, gamma_t=gamma_t,
        t0=t0, beta=beta
    )

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
    use_bowing_suggestion: bool = False,
    gamma_h: float = 0.0,
    gamma_t: float = 0.0,
    t0: float = 0.95,
) -> pd.DataFrame:
    lo, hi = bg
    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    EA, EB = float(dA["band_gap"]), float(dB["band_gap"])

    # Optional bowing suggestion
    if use_bowing_suggestion:
        b_suggest = _suggest_bowing(EA, EB, center=center)
        if b_suggest is not None:
            bow = float(np.clip(b_suggest, -1.0, 1.0))

    # Optional Ge branch
    if z > 0:
        A_Ge, B_Ge = A.replace("Sn", "Ge"), B.replace("Sn", "Ge")
        dA_Ge = fetch_mp_data(A_Ge, ["band_gap", "energy_above_hull"]) or dA
        dB_Ge = fetch_mp_data(B_Ge, ["band_gap", "energy_above_hull"]) or dB
        oxA_Ge = oxidation_energy(A_Ge)
        oxB_Ge = oxidation_energy(B_Ge)
    else:
        dA_Ge, dB_Ge = dA, dB
        oxA_Ge, oxB_Ge = oxidation_energy(A), oxidation_energy(B)

    hal = _infer_halide(A) or _infer_halide(B) or "I"
    rB = _b_site_radius(z)

    oxA, oxB = oxidation_energy(A), oxidation_energy(B)
    env_pen = _environment_penalty(rh, temp, gamma_h=gamma_h, gamma_t=gamma_t)
    tol_pen_const = _tolerance_penalty("Cs", rB, hal, t0=t0, beta=beta)

    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        # Sn branch
        Eg_Sn   = (1 - x) * EA + x * EB - bow * x * (1 - x)
        Eh_Sn   = (1 - x) * float(dA["energy_above_hull"]) + x * float(dB["energy_above_hull"])
        dEox_Sn = (1 - x) * oxA + x * oxB
        # Ge branch
        Eg_Ge   = (1 - x) * float(dA_Ge["band_gap"]) + x * float(dB_Ge["band_gap"]) - bow * x * (1 - x)
        Eh_Ge   = (1 - x) * float(dA_Ge["energy_above_hull"]) + x * float(dB_Ge["energy_above_hull"])
        dEox_Ge = (1 - x) * oxA_Ge + x * oxB_Ge

        # interpolate Sn/Ge
        Eg   = (1.0 - z) * Eg_Sn   + z * Eg_Ge
        Eh   = (1.0 - z) * Eh_Sn   + z * Eh_Ge
        dEox = (1.0 - z) * dEox_Sn + z * dEox_Ge

        # band-gap score & bounds
        sbg    = _score_band_gap(Eg, lo, hi, center, sigma)
        sbg_lo = _score_band_gap(Eg - DEFAULT_ERRS["Eg"], lo, hi, center, sigma)
        sbg_hi = _score_band_gap(Eg + DEFAULT_ERRS["Eg"], lo, hi, center, sigma)

        raw  = _score_raw(Eg, Eh, dEox, sbg, env_pen, tol_pen_const, alpha=alpha)
        rawL, rawH = _score_bounds(Eg, Eh, dEox, sbg_lo, sbg_hi, env_pen, tol_pen_const)

        pce = sq_efficiency(Eg)

        rows.append({
            "x":           round(x, 3),
            "z":           round(z, 2),
            "Eg":          round(Eg, 3),
            "Eg_err":      DEFAULT_ERRS["Eg"],
            "Ehull":       round(Eh, 4),
            "Ehull_err":   DEFAULT_ERRS["Ehull"],
            "Eox_e":       round(dEox, 3),
            "raw":         raw,
            "raw_low":     rawL,
            "raw_high":    rawH,
            "formula":     f"{A}-{B} x={x:.2f} z={z:.2f}",
            "PCE_max (%)": round(pce * 100, 1),
        })

    if not rows:
        return pd.DataFrame()

    m = max(r["raw"] for r in rows) or 1.0
    mL = max(r["raw_low"] for r in rows) or 1.0
    mH = max(r["raw_high"] for r in rows) or 1.0
    for r in rows:
        r["score"]      = round(r.pop("raw") / m, 3)
        r["score_low"]  = round(r.pop("raw_low") / mL, 3)
        r["score_high"] = round(r.pop("raw_high") / mH, 3)

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

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
    gamma_h: float = 0.0,
    gamma_t: float = 0.0,
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

    # Ge-branch
    if z > 0:
        A_Ge, B_Ge, C_Ge = A.replace("Sn", "Ge"), B.replace("Sn", "Ge"), C.replace("Sn", "Ge")
        dA_Ge = fetch_mp_data(A_Ge, ["band_gap", "energy_above_hull"]) or dA
        dB_Ge = fetch_mp_data(B_Ge, ["band_gap", "energy_above_hull"]) or dB
        dC_Ge = fetch_mp_data(C_Ge, ["band_gap", "energy_above_hull"]) or dC
    else:
        dA_Ge, dB_Ge, dC_Ge = dA, dB, dC

    hal = _infer_halide(A) or _infer_halide(B) or _infer_halide(C) or "I"
    env_pen = _environment_penalty(rh, temp, gamma_h=gamma_h, gamma_t=gamma_t)

    oxA, oxB, oxC = (oxidation_energy(f) for f in (A, B, C))
    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        for y in np.arange(0.0, 1.0 - x + 1e-9, dy):
            w = 1.0 - x - y
            # Sn gaps with pairwise bowing
            Eg_Sn = (
                w * float(dA["band_gap"]) + x * float(dB["band_gap"]) + y * float(dC["band_gap"])
                - bows["AB"] * x * w - bows["AC"] * y * w - bows["BC"] * x * y
            )
            # Ge gaps
            Eg_Ge = (
                w * float(dA_Ge["band_gap"]) + x * float(dB_Ge["band_gap"]) + y * float(dC_Ge["band_gap"])
                - bows["AB"] * x * w - bows["AC"] * y * w - bows["BC"] * x * y
            )
            Eg = (1.0 - z) * Eg_Sn + z * Eg_Ge

            Eh_Sn = w * float(dA["energy_above_hull"]) + x * float(dB["energy_above_hull"]) + y * float(dC["energy_above_hull"])
            Eh_Ge = w * float(dA_Ge["energy_above_hull"]) + x * float(dB_Ge["energy_above_hull"]) + y * float(dC_Ge["energy_above_hull"])
            Eh = (1.0 - z) * Eh_Sn + z * Eh_Ge

            dEox = w * oxA + x * oxB + y * oxC

            sbg    = _score_band_gap(Eg, lo, hi, center, sigma)
            sbg_lo = _score_band_gap(Eg - DEFAULT_ERRS["Eg"], lo, hi, center, sigma)
            sbg_hi = _score_band_gap(Eg + DEFAULT_ERRS["Eg"], lo, hi, center, sigma)

            tol_pen = 1.0  # (optional) can add ternary tolerance handling if desired
            raw  = _score_raw(Eg, Eh, dEox, sbg, env_pen, tol_pen)
            rawL, rawH = _score_bounds(Eg, Eh, dEox, sbg_lo, sbg_hi, env_pen, tol_pen)

            pce = sq_efficiency(Eg)

            rows.append({
                "x": round(x, 3), "y": round(y, 3), "z": round(z, 2),
                "Eg": round(Eg, 3), "Eg_err": DEFAULT_ERRS["Eg"],
                "Ehull": round(Eh, 4), "Ehull_err": DEFAULT_ERRS["Ehull"],
                "Eox_e": round(dEox, 3),
                "PCE_max (%)": round(pce * 100, 1),
                "raw": raw, "raw_low": rawL, "raw_high": rawH,
                "formula": f"{A}-{B}-{C} x={x:.2f} y={y:.2f} z={z:.2f}",
            })

    if not rows:
        return pd.DataFrame()
    m  = max(r["raw"] for r in rows) or 1.0
    mL = max(r["raw_low"] for r in rows) or 1.0
    mH = max(r["raw_high"] for r in rows) or 1.0
    for r in rows:
        r["score"]      = round(r.pop("raw") / m, 3)
        r["score_low"]  = round(r.pop("raw_low") / mL, 3)
        r["score_high"] = round(r.pop("raw_high") / mH, 3)

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
