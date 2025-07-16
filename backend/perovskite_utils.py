# ─────────────────────────────────────────────────────────────────────────────
#  perovskite_utils_refactor.py – EnerMat backend **v10.1**  (2025‑07‑16)
# -----------------------------------------------------------------------------
#  CHANGE LOG (v10.1 vs v10.0)
#  • FIXED: Ge fraction *z* now genuinely affects binary A–B screens.
#           – If z > 0 we fetch the Ge analogues (CsGeX3) of both end‑members
#             and linearly interpolate band‑gap and Ehull:
#                 Eg  = (1‑z)·Eg_Sn  + z·Eg_Ge
#                 Eh  = (1‑z)·Eh_Sn  + z·Eh_Ge
#           – Oxidation term stays Sn‑only (Ge oxidation negligible).
#  • Cosmetic: doc‑strings, typing tweaks, __all__ updated.
# -----------------------------------------------------------------------------
from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from joblib import Memory
from mp_api.client import MPRester
from pymatgen.core import Composition

__all__ = [
    "init_mprester",
    "screen_binary",
    "screen_ternary",
    "DesignSpace",
    "END_MEMBERS",
]

# ─────────── configuration / constants ───────────
_CACHE_DIR = os.getenv("ENERMAT_CACHE", os.path.expanduser("~/.enermat_cache"))
memory = Memory(_CACHE_DIR, verbose=0)

END_MEMBERS = [
    "CsSnI3", "CsSnBr3", "CsSnCl3",
    "CsGeBr3", "CsGeCl3",
]

CALIBRATED_GAPS: Dict[str, float] = {
    "CsSnBr3": 1.79,
    "CsSnCl3": 2.83,
    "CsSnI3":  1.00,
    "CsGeBr3": 2.20,
    "CsGeCl3": 3.30,
}

GAP_OFFSET = {"I": +0.52, "Br": +0.88, "Cl": +1.10}
IONIC_RADII = {"Cs": 1.88, "Sn": 1.18, "Ge": 0.73,
               "I": 2.20, "Br": 1.96, "Cl": 1.81}

# softening factors
kT_eff   = 0.020  # eV – controls Ehull penalty (≈kT*alpha)
K_OX     = 0.20   # eV – oxidation softening (higher → harsher)

# ─────────── Materials Project access (lazy) ───────────
_mpr: MPRester | None = None

def init_mprester(api_key: str | None = None) -> None:
    """Initialise (or re‑initialise) the global MPRester."""
    global _mpr
    key = api_key or os.getenv("MP_API_KEY")
    if not key or len(key) != 32:
        raise RuntimeError("MP_API_KEY (32‑char) is missing or invalid")
    _mpr = MPRester(key)

def _get_mpr() -> MPRester:
    if _mpr is None:
        init_mprester()
    return _mpr  # type: ignore [return-value]

# ─────────── MP helpers with on‑disk cache ───────────
@memory.cache(ignore=["fields"])
def fetch_mp_data(formula: str, fields: Tuple[str, ...]) -> dict | None:
    """Cached fetch – stores JSON on disk (≈1 kB per doc)."""
    docs = _get_mpr().summary.search(formula=formula, fields=fields)
    if not docs:
        return None
    ent = docs[0]
    out = {f: getattr(ent, f, None) for f in fields}

    # empirical gap correction
    if "band_gap" in fields:
        if formula in CALIBRATED_GAPS:
            out["band_gap"] = CALIBRATED_GAPS[formula]
        else:
            hal = next(h for h in ("I", "Br", "Cl") if h in formula)
            out["band_gap"] = (out["band_gap"] or 0.0) + GAP_OFFSET[hal]
    return out

# ─────────── oxidation penalty ───────────
@lru_cache(maxsize=128)
def oxidation_energy(formula_sn2: str) -> float:
    if "Sn" not in formula_sn2:
        return 0.0
    hal = next((h for h in ("I", "Br", "Cl") if h in formula_sn2), None)
    if hal is None:
        return 0.0

    def _H(formula: str) -> float:
        d = fetch_mp_data(formula, ("formation_energy_per_atom",))
        if not d or d["formation_energy_per_atom"] is None:
            raise ValueError(f"Missing formation energy for {formula}")
        comp = Composition(formula)
        return d["formation_energy_per_atom"] * comp.num_atoms

    H_reac  = _H(formula_sn2)
    H_prod1 = _H(f"Cs2Sn{hal}6")
    H_prod2 = _H("SnO2")
    return 0.5 * (H_prod1 + H_prod2) - H_reac

# ─────────── scoring helpers ───────────
def _soft_gap(Eg: np.ndarray, lo: float, hi: float, sigma: float = 0.10) -> np.ndarray:
    mid = 0.5 * (lo + hi)
    return np.exp(-0.5 * ((Eg - mid) / sigma) ** 2)

# ─────────── binary grid (Sn ⇄ Ge ready) ───────────
@memory.cache(ignore=["dx", "bg", "bow", "beta_Ox"])
def screen_binary(
    A: str,
    B: str,
    rh: float,
    temp: float,
    bg: Tuple[float, float],
    bow: float,
    dx: float,
    *,
    z: float = 0.0,          # Ge fraction on B‑site
    beta_Ox: float = 1.0,
) -> pd.DataFrame:
    """Score CsSn₁₋zGe_zX₃–CsSn₁₋zGe_zY₃ binary alloy.
        If *z* > 0 we linearly blend Sn and Ge branches for Eg and Eh.
    """
    # --- Sn branch data ----------------------------------------------------
    dA = fetch_mp_data(A, ("band_gap", "energy_above_hull"))
    dB = fetch_mp_data(B, ("band_gap", "energy_above_hull"))
    if not (dA and dB):
        return pd.DataFrame()

    # --- optional Ge branch ------------------------------------------------
    if z > 0:
        A_Ge = A.replace("Sn", "Ge")
        B_Ge = B.replace("Sn", "Ge")
        dA_Ge = fetch_mp_data(A_Ge, ("band_gap", "energy_above_hull")) or dA
        dB_Ge = fetch_mp_data(B_Ge, ("band_gap", "energy_above_hull")) or dB
    else:
        dA_Ge = dA
        dB_Ge = dB

    # --- oxidation (Sn only) ----------------------------------------------
    oxA, oxB = oxidation_energy(A), oxidation_energy(B)

    xs = np.arange(0.0, 1.0 + 1e-12, dx)

    # ----- band gap --------------------------------------------------------
    Eg_Sn = (1 - xs) * dA["band_gap"] + xs * dB["band_gap"] - bow * xs * (1 - xs)
    Eg_Ge = (1 - xs) * dA_Ge["band_gap"] + xs * dB_Ge["band_gap"] - bow * xs * (1 - xs)
    Eg    = (1.0 - z) * Eg_Sn + z * Eg_Ge

    # ----- hull energy -----------------------------------------------------
    Eh_Sn = (1 - xs) * dA["energy_above_hull"] + xs * dB["energy_above_hull"]
    Eh_Ge = (1 - xs) * dA_Ge["energy_above_hull"] + xs * dB_Ge["energy_above_hull"]
    Eh    = (1.0 - z) * Eh_Sn + z * Eh_Ge

    # ----- oxidation -------------------------------------------------------
    dEox = (1 - xs) * oxA + xs * oxB

    # ----- score -----------------------------------------------------------
    Sgap = _soft_gap(Eg, *bg)
    S = Sgap * np.exp(-Eh / kT_eff) * np.exp(beta_Ox * dEox / K_OX)

    df = pd.DataFrame({
        "x": xs.round(3),
        "z": round(z, 2),
        "Eg": Eg.round(3),
        "Ehull": Eh.round(4),
        "Eox": dEox.round(3),
        "score": S,
        "formula": [f"{A}-{B} x={x:.2f} z={z:.2f}" for x in xs],
    })
    df["score"] /= df["score"].max()
    df["score"] = df["score"].round(3)
    df.sort_values("score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ─────────── ternary grid (unchanged from v10.0) ───────────
@memory.cache(ignore=["dx", "dy", "bg", "bows", "beta_Ox"])
def screen_ternary(
