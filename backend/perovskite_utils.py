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

# ─────────── Shockley–Queisser helper (unchanged expectation) ───────────
from backend.sq import sq_efficiency

# ─────────── API key / MP client (graceful) ───────────
load_dotenv()
_API_KEY = (os.getenv("MP_API_KEY") or getattr(st, "secrets", {}).get("MP_API_KEY") or "").strip()

mpr = None
offline_mode = False
try:
    if len(_API_KEY) == 32:
        from mp_api.client import MPRester
        mpr = MPRester(_API_KEY)
    else:
        offline_mode = True
except Exception:
    offline_mode = True

# ─────────── Application presets (unchanged) ───────────
APPLICATION_CONFIG = {
    "single":   {"range": (1.10, 1.40), "center": 1.25, "sigma": 0.10},
    "tandem":   {"range": (1.60, 1.90), "center": 1.75, "sigma": 0.10},
    "indoor":   {"range": (1.70, 2.20), "center": 1.95, "sigma": 0.15},
    "detector": {"range": (0.80, 3.00), "center": None,  "sigma": None},
}

# Presets for convenience (you can still type custom formulas)
END_MEMBERS = [
    "CsSnI3", "CsSnBr3", "CsSnCl3",
    "CsGeBr3", "CsGeCl3",
    "FASnI3", "MASnBr3",
    "Cs2SnI6",
    "Cs3Bi2Br9", "Cs3Sb2I9",
    "Cs2AgBiBr6", "Cs2AgInCl6",
    "CsPbCl3", "CsPbBr3", "CsPbI3",
]

# Optional: load calibrated gaps (same logic as before)
def _load_calibrated_gaps() -> dict[str, float]:
    j = getattr(st, "secrets", {}).get("CALIBRATED_GAPS_JSON")
    if j:
        try:
            return json.loads(j)
        except Exception:
            pass
    path = os.getenv("CALIBRATED_GAPS_PATH")
    if path and os.path.exists(path):
        try:
            import json as _json
            with open(path, "r", encoding="utf-8") as f:
                return _json.load(f)
        except Exception:
            pass
    return {
        "CsSnI3": 1.30, "CsSnBr3": 1.80, "CsSnCl3": 3.00,
        "FASnI3": 1.30, "Cs2SnI6": 1.60, "Cs3Bi2Br9": 2.20, "Cs2AgBiBr6": 2.00,
    }

CALIBRATED_GAPS = _load_calibrated_gaps()

# Halide-based offsets as before
GAP_OFFSET = {"I": 0.85, "Br": 0.78, "Cl": 2.00, "Pb": 1.31}

# Ionic radii (coarse; added a few dopants)
IONIC_RADII = {
    "Cs": 1.88, "FA": 2.79, "MA": 2.70,
    "Sn": 1.18, "Ge": 0.73, "Pb": 1.31,
    "Si": 0.54, "Mn": 0.83, "Zn": 0.74,
    "Bi": 1.03, "Sb": 0.76, "Ag": 1.15, "In": 0.81,
    "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

# Soft “energy scales”
K_T_EFF  = 0.20   # eV for oxidation reward
K_T_HULL = 0.0259 # ~kT

# Default pseudo-uncertainties for score bands
DEFAULT_ERRS = {"Eg": 0.10, "Ehull": 0.01}

# Supported isovalent dopants (won’t trigger scope warning)
ISOVALENT_B_SITE = {"Ge", "Si", "Pb"}

# ─────────── Helpers ───────────
def _infer_halide(formula: str) -> str | None:
    for h in ("I", "Br", "Cl"):
        if h in formula:
            return h
    return None

def _b_site_radius(dopant: str | None, z: float) -> float:
    r_sn = IONIC_RADII["Sn"]
    if not dopant or dopant == "None" or z <= 0:
        return r_sn
    r_d = IONIC_RADII.get(dopant, r_sn)
    return (1.0 - z) * r_sn + z * r_d

def _score_band_gap(Eg: float, lo: float, hi: float, center: float | None, sigma: float | None) -> float:
    if Eg < lo or Eg > hi:
        return 0.0
    if center is None or sigma is None:
        return 1.0
    return math.exp(-((Eg - center) ** 2) / (2 * sigma * sigma))

score_band_gap = _score_band_gap

# ─────────── MP data (offline-safe) ───────────
_OFFLINE_SUMMARY: dict[str, dict] = {
    "CsSnI3":     {"band_gap": CALIBRATED_GAPS.get("CsSnI3", 1.3), "energy_above_hull": 0.02},
    "CsSnBr3":    {"band_gap": CALIBRATED_GAPS.get("CsSnBr3", 1.8), "energy_above_hull": 0.01},
    "Cs2SnI6":    {"band_gap": CALIBRATED_GAPS.get("Cs2SnI6", 1.6), "energy_above_hull": 0.00},
    "Cs3Bi2Br9":  {"band_gap": CALIBRATED_GAPS.get("Cs3Bi2Br9", 2.2), "energy_above_hull": 0.04},
    "Cs2AgBiBr6": {"band_gap": CALIBRATED_GAPS.get("Cs2AgBiBr6", 2.0), "energy_above_hull": 0.00},
    # include some Ge/Si/Pb analogues for demo fallback if needed
    "CsGeBr3": {"band_gap": 2.40, "energy_above_hull": 0.03},
    "CsGeCl3": {"band_gap": 3.20, "energy_above_hull": 0.03},
}

def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    if (mpr is not None) and (not offline_mode):
        try:
            docs = mpr.summary.search(formula=formula, fields=tuple(set(fields)))
            if docs:
                ent = docs[0]
                out = {f: getattr(ent, f, None) for f in fields}
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
    doc = _OFFLINE_SUMMARY.get(formula)
    if not doc:
        return None
    out = {f: doc.get(f) for f in fields}
    if "band_gap" in fields and out.get("band_gap") is None:
        hal = _infer_halide(formula)
        out["band_gap"] = (0.0) + (GAP_OFFSET.get(hal, 0.0) if hal else 0.0)
    return out

# ─────────── Oxidation proxy ───────────
@lru_cache(maxsize=128)
def oxidation_energy(formula_sn2: str) -> float:
    """
    ΔE_ox per Sn for: CsSnX3 + 1/2 O2 → 1/2 (Cs2SnX6 + SnO2)
    If no Sn present (e.g., heterovalent or other B-site), return neutral 0.0.
    """
    if "Sn" not in formula_sn2:
        return 0.0
    hal = _infer_halide(formula_sn2)
    if hal is None:
        return 0.0

    def formation_energy_per_fu(formula: str) -> float:
        doc = fetch_mp_data(formula, ["formation_energy_per_atom"])
        if not doc or doc.get("formation_energy_per_atom") is None:
            return 0.0
        from pymatgen.core import Composition
        comp = Composition(formula)
        return float(doc["formation_energy_per_atom"]) * comp.num_atoms

    H_reac  = formation_energy_per_fu(formula_sn2)
    H_prod1 = formation_energy_per_fu(f"Cs2Sn{hal}6")
    H_prod2 = formation_energy_per_fu("SnO2")
    return 0.5 * (H_prod1 + H_prod2) - H_reac

# ─────────── Penalties & bounds ───────────
def _environment_penalty(rh: float, temp_c: float, *, gamma_h: float, gamma_t: float) -> float:
    rh_n = max(0.0, min(1.0, rh / 100.0))
    dt   = max(0.0, temp_c - 25.0)
    pen_h = math.exp(-gamma_h * rh_n)
    pen_t = math.exp(-gamma_t * (dt / 50.0))
    return max(0.0, min(1.0, pen_h * pen_t))

def _tolerance_penalty(rB: float, X: str, *, t0: float, beta: float, A: str = "Cs") -> float:
    rA = IONIC_RADII.get(A, 1.88)
    rX = IONIC_RADII.get(X, 2.0)
    t  = (rA + rX) / (math.sqrt(2.0) * (rB + rX))
    return math.exp(-beta * abs(t - t0))

def _score_raw(Eg, Eh, dEox, sbg, env_pen, tol_pen, *, alpha=1.0) -> float:
    return sbg * math.exp(-Eh / (alpha * K_T_HULL)) * math.exp(dEox / K_T_EFF) * env_pen * tol_pen

def _score_bounds(Eg, Eh, dEox, sbg_lo, sbg_hi, env_pen, tol_pen) -> tuple[float, float]:
    raw_lo = _score_raw(Eg, Eh + DEFAULT_ERRS["Ehull"], dEox, sbg_lo, env_pen, tol_pen)
    raw_hi = _score_raw(Eg, Eh - DEFAULT_ERRS["Ehull"], dEox, sbg_hi, env_pen, tol_pen)
    return raw_lo, raw_hi

def _suggest_bowing(EA: float, EB: float, *, center: float | None) -> float | None:
    if center is None:
        return None
    return ((EA + EB) * 0.5 - center) / 0.25

# ─────────── Doping helpers ───────────
def _dopant_scope(dopant_element: str | None) -> tuple[bool, str]:
    if not dopant_element or dopant_element == "None":
        return True, "No B-site doping"
    if dopant_element in ISOVALENT_B_SITE:
        return True, "Isovalent B-site doping"
    return False, "Heterovalent/exploratory mode"

def _fetch_sn_and_dopant_data(A: str, B: str, dopant: str | None) -> tuple[dict, dict, dict, dict]:
    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    if not dopant or dopant == "None":
        return dA, dB, dA, dB  # dopant branch = Sn branch (z=0)
    A_D = A.replace("Sn", dopant)
    B_D = B.replace("Sn", dopant)
    dA_D = fetch_mp_data(A_D, ["band_gap", "energy_above_hull"]) or dA
    dB_D = fetch_mp_data(B_D, ["band_gap", "energy_above_hull"]) or dB
    return dA, dB, dA_D, dB_D

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
    z: float = 0.0,                          # kept for backward-compat (interpreted as dopant fraction)
    application: str | None = None,
    use_bowing_suggestion: bool = False,
    gamma_h: float = 0.0,
    gamma_t: float = 0.0,
    t0: float = 0.95,
    beta: float = 1.0,
    dopant_element: str | None = "Ge",       # NEW
    dopant_fraction: float | None = None,    # NEW (if None, uses z)
) -> pd.DataFrame:
    lo, hi = bg
    center = sigma = None
    if application in APPLICATION_CONFIG:
        cfg = APPLICATION_CONFIG[application]
        lo, hi = cfg["range"]
        center, sigma = cfg["center"], cfg["sigma"]

    # Backward compatibility: if dopant_fraction is None, use z; if element None -> "Ge"
    if dopant_fraction is None:
        dopant_fraction = float(z or 0.0)
    dopant = dopant_element or "Ge"

    in_scope, scope_msg = _dopant_scope(dopant if dopant_fraction > 0 else None)

    dA, dB, dA_D, dB_D = _fetch_sn_and_dopant_data(A, B, dopant if dopant_fraction > 0 else None)
    if not (dA and dB):
        return pd.DataFrame()

    EA, EB = float(dA["band_gap"]), float(dB["band_gap"])
    if use_bowing_suggestion:
        b_suggest = _suggest_bowing(EA, EB, center=center)
        if b_suggest is not None:
            bow = float(np.clip(b_suggest, -1.0, 1.0))

    hal = _infer_halide(A) or _infer_halide(B) or "I"
    rB = _b_site_radius(dopant, dopant_fraction)

    oxA_sn, oxB_sn = oxidation_energy(A), oxidation_energy(B)
    # If dopant path is unknown (non-Sn), keep oxidation neutral for the dopant branch
    oxA_d = oxidation_energy(A.replace("Sn", dopant)) if (dopant and "Sn" in A and dopant in ISOVALENT_B_SITE) else 0.0
    oxB_d = oxidation_energy(B.replace("Sn", dopant)) if (dopant and "Sn" in B and dopant in ISOVALENT_B_SITE) else 0.0

    env_pen = _environment_penalty(rh, temp, gamma_h=gamma_h, gamma_t=gamma_t)
    tol_pen_const = _tolerance_penalty(rB, hal, t0=t0, beta=beta)

    rows: list[dict] = []
    zf = float(max(0.0, min(1.0, dopant_fraction)))
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        # Sn branch
        Eg_Sn   = (1 - x) * EA + x * float(dB["band_gap"]) - bow * x * (1 - x)
        Eh_Sn   = (1 - x) * float(dA["energy_above_hull"]) + x * float(dB["energy_above_hull"])
        dEox_Sn = (1 - x) * oxA_sn + x * oxB_sn
        # Dopant branch (falls back to Sn if missing)
        Eg_D    = (1 - x) * float(dA_D["band_gap"]) + x * float(dB_D["band_gap"]) - bow * x * (1 - x)
        Eh_D    = (1 - x) * float(dA_D["energy_above_hull"]) + x * float(dB_D["energy_above_hull"])
        dEox_D  = (1 - x) * oxA_d + x * oxB_d

        Eg   = (1.0 - zf) * Eg_Sn + zf * Eg_D
        Eh   = (1.0 - zf) * Eh_Sn + zf * Eh_D
        dEox = (1.0 - zf) * dEox_Sn + zf * dEox_D

        sbg    = _score_band_gap(Eg, lo, hi, center, sigma)
        sbg_lo = _score_band_gap(Eg - DEFAULT_ERRS["Eg"], lo, hi, center, sigma)
        sbg_hi = _score_band_gap(Eg + DEFAULT_ERRS["Eg"], lo, hi, center, sigma)

        raw  = _score_raw(Eg, Eh, dEox, sbg, env_pen, tol_pen_const)
        rawL, rawH = _score_bounds(Eg, Eh, dEox, sbg_lo, sbg_hi, env_pen, tol_pen_const)

        pce = sq_efficiency(Eg)

        rows.append({
            "x": round(x, 3),
            "z": round(zf, 2),
            "dopant": dopant if zf > 0 else "None",
            "Eg": round(Eg, 3),
            "Eg_err": DEFAULT_ERRS["Eg"],
            "Ehull": round(Eh, 4),
            "Ehull_err": DEFAULT_ERRS["Ehull"],
            "Eox_e": round(dEox, 3),
            "PCE_max (%)": round(pce * 100, 1),
            "raw": raw, "raw_low": rawL, "raw_high": rawH,
            "formula": f"{A}-{B} x={x:.2f} z={zf:.2f} ({dopant if zf>0 else 'no dopant'})",
            "scope": scope_msg, "in_scope": bool(in_scope),
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

# ─────────── Ternary screen (accepts t0/beta; uses tolerance penalty) ───────────
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
    z: float = 0.0,                          # backward-compat (dopant fraction)
    application: str | None = None,
    gamma_h: float = 0.0,
    gamma_t: float = 0.0,
    dopant_element: str | None = "Ge",
    dopant_fraction: float | None = None,
    # NEW: accept structural penalty args (UI sends these)
    t0: float = 0.95,
    beta: float = 1.0,
) -> pd.DataFrame:
    lo, hi = bg
    center = sigma = None
    if application in APPLICATION_CONFIG:
        cfg = APPLICATION_CONFIG[application]
        lo, hi = cfg["range"]
        center, sigma = cfg["center"], cfg["sigma"]

    if dopant_fraction is None:
        dopant_fraction = float(z or 0.0)
    dopant = dopant_element or "Ge"

    in_scope, scope_msg = _dopant_scope(dopant if dopant_fraction > 0 else None)

    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    # Dopant branches
    if dopant_fraction > 0:
        A_D, B_D, C_D = A.replace("Sn", dopant), B.replace("Sn", dopant), C.replace("Sn", dopant)
        dA_D = fetch_mp_data(A_D, ["band_gap", "energy_above_hull"]) or dA
        dB_D = fetch_mp_data(B_D, ["band_gap", "energy_above_hull"]) or dB
        dC_D = fetch_mp_data(C_D, ["band_gap", "energy_above_hull"]) or dC
    else:
        dA_D, dB_D, dC_D = dA, dB, dC

    # penalties
    env_pen = _environment_penalty(rh, temp, gamma_h=gamma_h, gamma_t=gamma_t)
    zf = float(max(0.0, min(1.0, dopant_fraction)))
    hal = _infer_halide(A) or _infer_halide(B) or _infer_halide(C) or "I"
    rB = _b_site_radius(dopant, zf)
    tol_pen_const = _tolerance_penalty(rB, hal, t0=t0, beta=beta)

    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        for y in np.arange(0.0, 1.0 - x + 1e-9, dy):
            w = 1.0 - x - y
            # Eg with pairwise bowing
            Eg_Sn = (
                w * float(dA["band_gap"]) + x * float(dB["band_gap"]) + y * float(dC["band_gap"])
                - bows["AB"] * x * w - bows["AC"] * y * w - bows["BC"] * x * y
            )
            Eg_D = (
                w * float(dA_D["band_gap"]) + x * float(dB_D["band_gap"]) + y * float(dC_D["band_gap"])
                - bows["AB"] * x * w - bows["AC"] * y * w - bows["BC"] * x * y
            )
            Eg = (1.0 - zf) * Eg_Sn + zf * Eg_D

            # Ehull interpolation
            Eh_Sn = w * float(dA["energy_above_hull"]) + x * float(dB["energy_above_hull"]) + y * float(dC["energy_above_hull"])
            Eh_D  = w * float(dA_D["energy_above_hull"]) + x * float(dB_D["energy_above_hull"]) + y * float(dC_D["energy_above_hull"])
            Eh = (1.0 - zf) * Eh_Sn + zf * Eh_D

            # Oxidation: keep neutral for dopant branch; rely on Sn path
            oxA, oxB, oxC = oxidation_energy(A), oxidation_energy(B), oxidation_energy(C)
            dEox = w * oxA + x * oxB + y * oxC

            # scoring + bounds
            sbg    = _score_band_gap(Eg, lo, hi, center, sigma)
            sbg_lo = _score_band_gap(Eg - DEFAULT_ERRS["Eg"], lo, hi, center, sigma)
            sbg_hi = _score_band_gap(Eg + DEFAULT_ERRS["Eg"], lo, hi, center, sigma)

            raw  = _score_raw(Eg, Eh, dEox, sbg, env_pen, tol_pen_const)
            rawL, rawH = _score_bounds(Eg, Eh, dEox, sbg_lo, sbg_hi, env_pen, tol_pen_const)

            pce = sq_efficiency(Eg)

            rows.append({
                "x": round(x, 3), "y": round(y, 3),
                "z": round(zf, 2), "dopant": dopant if zf > 0 else "None",
                "Eg": round(Eg, 3), "Eg_err": DEFAULT_ERRS["Eg"],
                "Ehull": round(Eh, 4), "Ehull_err": DEFAULT_ERRS["Ehull"],
                "Eox_e": round(dEox, 3),
                "PCE_max (%)": round(pce * 100, 1),
                "raw": raw, "raw_low": rawL, "raw_high": rawH,
                "formula": f"{A}-{B}-{C} x={x:.2f} y={y:.2f} z={zf:.2f} ({dopant if zf>0 else 'no dopant'})",
                "scope": scope_msg, "in_scope": bool(in_scope),
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
