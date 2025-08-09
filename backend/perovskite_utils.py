from __future__ import annotations
import json
import math
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Streamlit (optional on backend)
try:
    import streamlit as st
except Exception:
    class _Dummy:
        def __getattr__(self, k): return {}
    st = _Dummy()  # type: ignore

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

# ─────────── Application profiles ───────────
# For plotting convenience, a 'range' is provided where relevant.
APPLICATION_CONFIG = {
    "single":   {"range": (1.10, 1.40), "center": None,  "sigma": None,  "spectrum": "AM1.5G"},
    "tandem":   {"range": (1.60, 1.90), "center": 1.75,  "sigma": 0.10,  "spectrum": "AM1.5G"},
    "indoor":   {"range": (1.80, 2.10), "center": 1.95,  "sigma": 0.15,  "spectrum": "AM1.5G"},  # using AM1.5G SQ as proxy
    "detector": {"range": None,         "center": None,  "sigma": None,  "spectrum": None},
    # 'custom' will be provided by caller (center/sigma), spectrum=AM1.5G by default
}

# ─────────── End-members list ───────────
END_MEMBERS = [
    "CsSnI3", "CsSnBr3", "CsSnCl3",
    "CsGeBr3", "CsGeCl3",
    "FASnI3", "MASnBr3",
    "Cs2SnI6",
    "Cs3Bi2Br9", "Cs3Sb2I9",
    "Cs2AgBiBr6", "Cs2AgInCl6",
    "CsPbCl3", "CsPbBr3", "CsPbI3",
]

# ─────────── Calibrated gaps (lightweight fallback) ───────────
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
GAP_OFFSET = {"I": 0.85, "Br": 0.78, "Cl": 2.00, "Pb": 1.31}

IONIC_RADII = {
    "Cs": 1.88, "FA": 2.79, "MA": 2.70,
    "Sn": 1.18, "Ge": 0.73, "Pb": 1.31, "Si": 0.54, "Mn": 0.83, "Zn": 0.74,
    "Bi": 1.03, "Sb": 0.76, "Ag": 1.15, "In": 0.81,
    "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

# Softness scales
K_T_EFF  = 0.20    # oxidation softness
K_T_HULL = 0.0259  # hull softness (~kT at 300K)

ISOVALENT_B_SITE = {"Ge", "Si", "Pb"}

# ─────────── Small chemistry helpers ───────────
def _infer_halide(formula: str) -> str | None:
    for h in ("I", "Br", "Cl"):
        if h in formula:
            return h
    return None

def _parse_abx3(formula: str) -> tuple[str | None, str | None]:
    for X in ("I", "Br", "Cl"):
        tag = f"Sn{X}3"
        if formula.endswith(tag):
            A = formula[:-len(tag)]
            return (A, X)
    return (None, None)

SUBS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
def _subnum(x: float, nd=2) -> str:
    s = f"{x:.{nd}f}".rstrip("0").rstrip(".")
    return s.translate(SUBS)

def _fmt_bsite(sn_frac: float, dopant: str | None) -> tuple[str, str]:
    sn = max(0.0, min(1.0, sn_frac))
    z  = 1.0 - sn
    if not dopant or dopant == "None" or z <= 1e-6:
        return ("Sn", "Sn")
    u = f"Sn{_subnum(sn)}{dopant}{_subnum(z)}"
    a = f"Sn{sn:.2f}{dopant}{z:.2f}"
    return (u, a)

def _fmt_xmix_binary(XA: str, XB: str, x: float) -> tuple[str, str]:
    x = max(0.0, min(1.0, x))
    if XA == XB:
        return (f"{XA}₃", f"{XA}3")
    fA, fB = 1.0 - x, x
    parts_u, parts_a = [], []
    if fA > 1e-6:
        parts_u.append(f"{XA}{_subnum(fA)}")
        parts_a.append(f"{XA}{fA:.2f}")
    if fB > 1e-6:
        parts_u.append(f"{XB}{_subnum(fB)}")
        parts_a.append(f"{XB}{fB:.2f}")
    return (f"({''.join(parts_u)})₃", f"({''.join(parts_a)})3")

def _fmt_xmix_ternary(Ax: str, Bx: str, Cx: str, w: float, x: float, y: float) -> tuple[str, str]:
    fracs = {"I": 0.0, "Br": 0.0, "Cl": 0.0}
    for frac, X in ((w, Ax), (x, Bx), (y, Cx)):
        if X in fracs: fracs[X] += frac
    parts_u, parts_a = [], []
    for X in ("I", "Br", "Cl"):
        if fracs[X] > 1e-6:
            parts_u.append(f"{X}{_subnum(fracs[X])}")
            parts_a.append(f"{X}{fracs[X]:.2f}")
    if not parts_u:
        return ("X₃", "X3")
    if len(parts_u) == 1:
        return (f"{parts_u[0][0]}₃", f"{parts_a[0][0]}3")
    return (f"({''.join(parts_u)})₃", f"({''.join(parts_a)})3")

def _b_site_radius(dopant: str | None, z: float) -> float:
    r_sn = IONIC_RADII["Sn"]
    if not dopant or dopant == "None" or z <= 0:
        return r_sn
    r_d = IONIC_RADII.get(dopant, r_sn)
    return (1.0 - z) * r_sn + z * r_d

# ─────────── MP data (offline-safe) ───────────
_OFFLINE_SUMMARY: dict[str, dict] = {
    "CsSnI3":     {"band_gap": CALIBRATED_GAPS.get("CsSnI3", 1.3), "energy_above_hull": 0.02},
    "CsSnBr3":    {"band_gap": CALIBRATED_GAPS.get("CsSnBr3", 1.8), "energy_above_hull": 0.01},
    "Cs2SnI6":    {"band_gap": CALIBRATED_GAPS.get("Cs2SnI6", 1.6), "energy_above_hull": 0.00},
    "Cs3Bi2Br9":  {"band_gap": CALIBRATED_GAPS.get("Cs3Bi2Br9", 2.2), "energy_above_hull": 0.04},
    "Cs2AgBiBr6": {"band_gap": CALIBRATED_GAPS.get("Cs2AgBiBr6", 2.0), "energy_above_hull": 0.00},
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

# ─────────── penalties & composite score ───────────
def _tolerance_penalty(rB: float, X: str, *, t0: float, beta: float, A: str = "Cs") -> float:
    rA = IONIC_RADII.get(A, 1.88)
    rX = IONIC_RADII.get(X, 2.0)
    t  = (rA + rX) / (math.sqrt(2.0) * (rB + rX))
    return math.exp(-beta * abs(t - t0))

def _gaussian_weight(Eg: float, center: float | None, sigma: float | None) -> float:
    if center is None or sigma is None or sigma <= 0:
        return 1.0
    return math.exp(-((Eg - center) ** 2) / (2 * sigma * sigma))

def _performance_term(Eg: float, spectrum: str | None = "AM1.5G") -> float:
    if spectrum is None:
        return 1.0
    # Using AM1.5G SQ for single/tandem/indoor (proxy for indoor)
    return max(0.0, float(sq_efficiency(Eg)))

def _composite_score(Eg: float, Eh: float, dEox: float, *, center: float | None,
                     sigma: float | None, spectrum: str | None, tol_pen: float) -> float:
    perf = _performance_term(Eg, spectrum=spectrum)
    g    = _gaussian_weight(Eg, center, sigma)
    thermo = math.exp(-Eh / K_T_HULL)
    oxstab = math.exp(dEox / K_T_EFF)
    return perf * g * thermo * oxstab * tol_pen

# ─────────── scaffolding for dopant scope ───────────
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
        return dA, dB, dA, dB
    A_D = A.replace("Sn", dopant)
    B_D = B.replace("Sn", dopant)
    dA_D = fetch_mp_data(A_D, ["band_gap", "energy_above_hull"]) or dA
    dB_D = fetch_mp_data(B_D, ["band_gap", "energy_above_hull"]) or dB
    return dA, dB, dA_D, dB_D

# ─────────── Binary screen ───────────
def screen_binary(
    A: str,
    B: str,
    rh_unused: float,      # retained for API stability; currently unused
    temp_unused: float,    # retained
    bg_unused: tuple[float, float],  # retained
    bow: float,
    dx: float,
    *,
    z: float = 0.0,
    application: str | None = None,
    t0: float = 0.95,
    beta: float = 1.0,
    dopant_element: str | None = "Ge",
    dopant_fraction: float | None = None,
    custom_center: float | None = None,  # for 'custom' mode
    custom_sigma: float | None = None,   # for 'custom' mode
    **kwargs,  # ← NEW: swallow any extra keywords
) -> pd.DataFrame:
    # Resolve application config
    app = (application or "single").lower()
    if app == "custom":
        cfg = {"center": custom_center, "sigma": custom_sigma, "spectrum": "AM1.5G", "range": None}
    else:
        cfg = APPLICATION_CONFIG.get(app, APPLICATION_CONFIG["single"])

    if dopant_fraction is None:
        dopant_fraction = float(z or 0.0)
    dopant = dopant_element or "Ge"
    in_scope, scope_msg = _dopant_scope(dopant if dopant_fraction > 0 else None)

    dA, dB, dA_D, dB_D = _fetch_sn_and_dopant_data(A, B, dopant if dopant_fraction > 0 else None)
    if not (dA and dB):
        return pd.DataFrame()

    Ax, XA = _parse_abx3(A)
    Bx, XB = _parse_abx3(B)
    halA = XA or _infer_halide(A) or "I"
    halB = XB or _infer_halide(B) or halA
    A_site = Ax or Bx or "A"

    rB = _b_site_radius(dopant, dopant_fraction)
    tol_pen_const = _tolerance_penalty(rB, halA, t0=t0, beta=beta)

    EA, EB = float(dA["band_gap"]), float(dB["band_gap"])

    rows: list[dict] = []
    zf = float(max(0.0, min(1.0, dopant_fraction)))
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        # Sn branch
        Eg_Sn   = (1 - x) * EA + x * float(dB["band_gap"]) - bow * x * (1 - x)
        Eh_Sn   = (1 - x) * float(dA["energy_above_hull"]) + x * float(dB["energy_above_hull"])
        dEox_Sn = (1 - x) * oxidation_energy(A) + x * oxidation_energy(B)

        # Dopant branch
        Eg_D    = (1 - x) * float(dA_D["band_gap"]) + x * float(dB_D["band_gap"]) - bow * x * (1 - x)
        Eh_D    = (1 - x) * float(dA_D["energy_above_hull"]) + x * float(dB_D["energy_above_hull"])
        dEox_D  = 0.0  # neutral for dopant proxy; rely on Sn oxidation proxy

        # Interpolate by dopant fraction
        Eg   = (1.0 - zf) * Eg_Sn   + zf * Eg_D
        Eh   = (1.0 - zf) * Eh_Sn   + zf * Eh_D
        dEox = (1.0 - zf) * dEox_Sn + zf * dEox_D

        raw = _composite_score(Eg, Eh, dEox,
                               center=cfg["center"], sigma=cfg["sigma"],
                               spectrum=cfg["spectrum"], tol_pen=tol_pen_const)
        pce = sq_efficiency(Eg)

        # pretty formula
        sn_frac = 1.0 - zf
        b_u, b_a = _fmt_bsite(sn_frac, dopant if zf > 0 else None)
        x_u, x_a = _fmt_xmix_binary(halA, halB, x)
        f_u = f"{A_site}{b_u}{x_u}"
        f_a = f"{A_site}{b_a}{x_a}"

        rows.append({
            "x": round(x, 3),
            "z": round(zf, 2),
            "dopant": dopant if zf > 0 else "None",
            "Eg": round(Eg, 3),
            "Ehull": round(Eh, 4),
            "Eox_e": round(dEox, 3),
            "PCE_max (%)": round(pce * 100, 1),
            "raw": raw,
            "formula": f_u,
            "formula_ascii": f_a,
            "scope": scope_msg, "in_scope": bool(in_scope),
        })

    if not rows:
        return pd.DataFrame()
    m = max(r["raw"] for r in rows) or 1.0
    for r in rows:
        r["score"] = round(r.pop("raw") / m, 3)

    keep = ["formula","formula_ascii","x","z","dopant","Eg","Ehull","Eox_e","PCE_max (%)","score","scope","in_scope"]
    return pd.DataFrame(rows)[keep].sort_values("score", ascending=False).reset_index(drop=True)

# ─────────── Ternary screen ───────────
def screen_ternary(
    A: str,
    B: str,
    C: str,
    rh_unused: float,
    temp_unused: float,
    bg_unused: tuple[float, float],
    bows: dict[str, float],
    *,
    dx: float = 0.10,
    dy: float = 0.10,
    z: float = 0.0,
    application: str | None = None,
    t0: float = 0.95,
    beta: float = 1.0,
    dopant_element: str | None = "Ge",
    dopant_fraction: float | None = None,
    custom_center: float | None = None,   # for 'custom'
    custom_sigma: float | None = None,    # for 'custom'
    **kwargs,  # ← NEW
) -> pd.DataFrame:
    app = (application or "single").lower()
    if app == "custom":
        cfg = {"center": custom_center, "sigma": custom_sigma, "spectrum": "AM1.5G", "range": None}
    else:
        cfg = APPLICATION_CONFIG.get(app, APPLICATION_CONFIG["single"])

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

    halA = _infer_halide(A) or "I"
    halB = _infer_halide(B) or halA
    halC = _infer_halide(C) or halA

    Ax, _ = _parse_abx3(A)
    Bx, _ = _parse_abx3(B)
    Cx, _ = _parse_abx3(C)
    A_site = Ax or Bx or Cx or "A"

    zf = float(max(0.0, min(1.0, dopant_fraction)))
    rB = _b_site_radius(dopant, zf)
    tol_pen_const = _tolerance_penalty(rB, halA, t0=t0, beta=beta)

    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        for y in np.arange(0.0, 1.0 - x + 1e-9, dy):
            w = 1.0 - x - y
            Eg_Sn = (
                w * float(dA["band_gap"]) + x * float(dB["band_gap"]) + y * float(dC["band_gap"])
                - bows["AB"] * x * w - bows["AC"] * y * w - bows["BC"] * x * y
            )
            Eg_D = (
                w * float(dA_D["band_gap"]) + x * float(dB_D["band_gap"]) + y * float(dC_D["band_gap"])
                - bows["AB"] * x * w - bows["AC"] * y * w - bows["BC"] * x * y
            )
            Eg = (1.0 - zf) * Eg_Sn + zf * Eg_D

            Eh_Sn = w * float(dA["energy_above_hull"]) + x * float(dB["energy_above_hull"]) + y * float(dC["energy_above_hull"])
            Eh_D  = w * float(dA_D["energy_above_hull"]) + x * float(dB_D["energy_above_hull"]) + y * float(dC_D["energy_above_hull"])
            Eh = (1.0 - zf) * Eh_Sn + zf * Eh_D

            oxA, oxB, oxC = oxidation_energy(A), oxidation_energy(B), oxidation_energy(C)
            dEox = w * oxA + x * oxB + y * oxC

            raw = _composite_score(Eg, Eh, dEox,
                                   center=cfg["center"], sigma=cfg["sigma"],
                                   spectrum=cfg["spectrum"], tol_pen=tol_pen_const)
            pce = sq_efficiency(Eg)

            # pretty formula
            sn_frac = 1.0 - zf
            b_u, b_a = _fmt_bsite(sn_frac, dopant if zf > 0 else None)
            x_u, x_a = _fmt_xmix_ternary(halA, halB, halC, w, x, y)
            f_u = f"{A_site}{b_u}{x_u}"
            f_a = f"{A_site}{b_a}{x_a}"

            rows.append({
                "x": round(x, 3), "y": round(y, 3),
                "z": round(zf, 2), "dopant": dopant if zf > 0 else "None",
                "Eg": round(Eg, 3),
                "Ehull": round(Eh, 4),
                "Eox_e": round(dEox, 3),
                "PCE_max (%)": round(pce * 100, 1),
                "raw": raw,
                "formula": f_u,
                "formula_ascii": f_a,
                "scope": scope_msg, "in_scope": bool(in_scope),
            })

    if not rows:
        return pd.DataFrame()
    m = max(r["raw"] for r in rows) or 1.0
    for r in rows:
        r["score"] = round(r.pop("raw") / m, 3)

    keep = ["formula","formula_ascii","x","y","z","dopant","Eg","Ehull","Eox_e","PCE_max (%)","score","scope","in_scope"]
    return pd.DataFrame(rows)[keep].sort_values("score", ascending=False).reset_index(drop=True)
