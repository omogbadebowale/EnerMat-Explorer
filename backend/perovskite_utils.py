from __future__ import annotations
import math, os
from functools import lru_cache

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from mp_api.client import MPRester
from pymatgen.core import Composition

# ───────────────────────── API key ─────────────────────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🚩 32-character MP_API_KEY missing in env or secrets")

mpr = MPRester(API_KEY)

# ───────────────────── reference data ─────────────────────
END_MEMBERS = [
    "CsSnI3", "CsSnBr3", "CsSnCl3",
    "CsGeBr3", "CsGeCl3",
]

CALIBRATED_GAPS = {          # experimental PL (eV)
    "CsSnI3" : 1.00,
    "CsSnBr3": 1.79,
    "CsSnCl3": 2.83,
    "CsGeBr3": 2.20,
    "CsGeCl3": 3.30,
}

GAP_OFFSET = {"I": +0.52, "Br": +0.88, "Cl": +1.10}   # DFT→exp shift

IONIC_RADII = {              # Å, 6-coord
    "Cs": 1.88, "Sn": 1.18, "Ge": 0.73,
    "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

K_T_EFF = 0.20               # soft-penalty scale (≈ 8 kT)

# ───────────────────── helper functions ───────────────────

def fetch_mp_data(formula: str, fields: list[str]):
    """Return `{field: value}` for a formula (patched band_gap where possible)."""
    docs = mpr.summary.search(formula=formula, fields=tuple(fields))
    if not docs:
        return None
    e = docs[0]
    out = {f: getattr(e, f, None) for f in fields}
    if "band_gap" in fields:
        if formula in CALIBRATED_GAPS:
            out["band_gap"] = CALIBRATED_GAPS[formula]
        else:
            hal = next(h for h in ("I", "Br", "Cl") if h in formula)
            out["band_gap"] = (out["band_gap"] or 0.0) + GAP_OFFSET[hal]
    return out

@lru_cache(maxsize=64)
def oxidation_energy(formula_sn2: str) -> float:
    """ΔEₒₓ per Sn for      CsSnX₃ + ½ O₂ → ½(Cs₂SnX₆ + SnO₂).
    Positive ΔEₒₓ ⇒ oxidation uphill (good).  Returns 0 for Sn-free."""
    if "Sn" not in formula_sn2:
        return 0.0
    hal = next((h for h in ("I", "Br", "Cl") if h in formula_sn2), None)
    if hal is None:
        return 0.0

    def ΔH(formula: str):
        doc = fetch_mp_data(formula, ["formation_energy_per_atom"])
        if not doc or doc["formation_energy_per_atom"] is None:
            raise ValueError(f"Missing ΔHf for {formula}")
        comp = Composition(formula)
        return doc["formation_energy_per_atom"] * comp.num_atoms

    try:
        H_reac  = ΔH(formula_sn2)
        H_prod1 = ΔH(f"Cs2Sn{hal}6")
        H_prod2 = ΔH("SnO2")
        return 0.5 * (H_prod1 + H_prod2) - H_reac
    except Exception:        # fallback: neutral (no penalty/bonus)
        return 0.0

score_band_gap = lambda Eg, lo, hi: 1.0 if lo <= Eg <= hi else 0.0

# ───────────────────────── binary screen ─────────────────────────

def screen_binary(A, B, rh, temp, bg, bow, dx, *, z: float = 0.0):
    """Wrapper for Streamlit caching."""
    return mix_abx3(A, B, rh, temp, bg, bow, dx, z=z)

def mix_abx3(
    A: str,
    B: str,
    rh: float,
    temp: float,
    bg_window: tuple[float, float],
    bowing: float = 0.0,
    dx: float = 0.05,
    z: float = 0.0,           # Ge fraction on B-site
    alpha: float = 1.0,       # Ehull softness
    beta: float = 1.0,        # tolerance-factor weight
) -> pd.DataFrame:

    lo, hi = bg_window
    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    # Ge surrogates (fall back to Sn when missing)
    dGA, dGB = dA, dB
    if z > 0:
        hal_A = next(h for h in ("I", "Br", "Cl") if h in A)
        hal_B = next(h for h in ("I", "Br", "Cl") if h in B)
        dGA = fetch_mp_data(f"CsGe{hal_A}3", ["band_gap", "energy_above_hull"]) or dA
        dGB = fetch_mp_data(f"CsGe{hal_B}3", ["band_gap", "energy_above_hull"]) or dB
    else:
        z = 0.0

    hal = next(h for h in ("I", "Br", "Cl") if h in A)
    rA, rB, rX = (IONIC_RADII[k] for k in ("Cs", "Sn", hal))
    oxA, oxB = oxidation_energy(A), oxidation_energy(B)

    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        # --- band gap ---
        Eg_sn = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bowing * x * (1 - x)
        Eg_ge = (1 - x) * dGA["band_gap"] + x * dGB["band_gap"] - bowing * x * (1 - x)
        Eg    = (1 - z) * Eg_sn + z * Eg_ge
        # --- Ehull ---
        Eh_sn = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        Eh_ge = (1 - x) * dGA["energy_above_hull"] + x * dGB["energy_above_hull"]
        Eh    = (1 - z) * Eh_sn + z * Eh_ge
        # --- oxidation ---
        dEox  = (1 - x) * oxA + x * oxB      # Ge oxidation ~ 0
        score = (score_band_gap(Eg, lo, hi)
                 * math.exp(-Eh / (alpha*0.0259))
                 * math.exp(-beta*abs(((rA+rX)/(math.sqrt(2)*(rB+rX))) - 0.95))
                 * math.exp(dEox / K_T_EFF))

        rows.append({
            "x": round(x,3), "z": round(z,2),
            "Eg": round(Eg,3), "Ehull": round(Eh,4), "Eox": round(dEox,3),
            "score": round(score,3),
            "formula": f"{A}-{B} x={x:.2f} z={z:.2f}",
        })

    # ── NEW: convert raw → 0-to-1 score ─────────────────────────────
    raw_max = max(r["raw"] for r in rows) or 1.0          # avoid /0
    for r in rows:
        r["score"] = round(r["raw"] / raw_max, 3)
        del r["raw"]

    return (pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True))

# ───────────────────────── ternary screen (Sn only) ─────────────────────────

def screen_ternary(
    A: str, B: str, C: str,
    rh: float, temp: float,
    bg: tuple[float, float],
    bows: dict[str, float],
    dx: float = 0.10, dy: float = 0.10,
) -> pd.DataFrame:

    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()
    oxA, oxB, oxC = oxidation_energy(A), oxidation_energy(B), oxidation_energy(C)
    lo, hi = bg

    rows: list[dict] = []
    for x in np.arange(0.0, 1.0 + 1e-9, dx):
        for y in np.arange(0.0, 1.0 - x + 1e-9, dy):
            z = 1 - x - y
            Eg = (z*dA["band_gap"] + x*dB["band_gap"] + y*dC["band_gap"]
                  - bows["AB"]*x*z - bows["AC"]*y*z - bows["BC"]*x*y)
            Eh = z*dA["energy_above_hull"] + x*dB["energy_above_hull"] + y*dC["energy_above_hull"]
            dEox = z*oxA + x*oxB + y*oxC
            score = (score_band_gap(Eg, lo, hi)
                     * math.exp(-Eh/0.0518)
                     * math.exp(dEox / K_T_EFF))
            rows.append({"x":round(x,3),"y":round(y,3),
                         "Eg":round(Eg,3),"Ehull":round(Eh,4),"Eox":round(dEox,3),
                         "score":round(score,3),
                         "formula":f"{A}-{B}-{C} x={x:.2f} y={y:.2f}"})
    # ── normalise ──────────────────────────────────────────────────
    raw_max = max(r["raw"] for r in rows) or 1.0
    for r in rows:
        r["score"] = round(r["raw"] / raw_max, 3)
        del r["raw"]

    return (pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True))

# keep alias for auto-report
_summary = fetch_mp_data
