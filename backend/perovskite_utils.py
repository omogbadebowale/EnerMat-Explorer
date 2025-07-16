# ─────────────────────────────────────────────────────────────────────────────
#  perovskite_utils_refactor.py – EnerMat backend **v10.2.0**  (2025‑07‑16)
# -----------------------------------------------------------------------------
#  • RESTORES full ternary implementation (previous hot‑fix left a placeholder).
#  • Keeps safe handling of missing formation energies but still computes Eox
#    when data are available.
# -----------------------------------------------------------------------------
from __future__ import annotations

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
    "END_MEMBERS",
]

# ─────────── configuration ───────────
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

kT_eff = 0.020  # Ehull softening (eV)
K_OX   = 0.20  # Oxidation softening (eV)

# ─────────── lazy MP access ───────────
_mpr: MPRester | None = None

def init_mprester(api_key: str | None = None) -> None:
    global _mpr
    key = api_key or os.getenv("MP_API_KEY")
    if not key or len(key) != 32:
        raise RuntimeError("MP_API_KEY (32‑char) missing or invalid")
    _mpr = MPRester(key)

def _mpr_client() -> MPRester:
    if _mpr is None:
        init_mprester()
    return _mpr  # type: ignore

# ─────────── cached fetch helper ───────────
@memory.cache(ignore=["fields"])
def fetch_mp_data(formula: str, fields: Tuple[str, ...]) -> dict | None:
    docs = _mpr_client().summary.search(formula=formula, fields=fields)
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
    for f in fields:
        out.setdefault(f, None)
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
        fe = d and d["formation_energy_per_atom"]
        if fe is None:
            return float("nan")
        return fe * Composition(formula).num_atoms

    H_reac  = _H(formula_sn2)
    H_prod1 = _H(f"Cs2Sn{hal}6")
    H_prod2 = _H("SnO2")
    if np.isnan([H_reac, H_prod1, H_prod2]).any():
        return 0.0
    return 0.5 * (H_prod1 + H_prod2) - H_reac

# helper
_soft_gap = lambda Eg, lo, hi, s=0.10: np.exp(-0.5 * ((Eg - (lo + hi) / 2) / s) ** 2)

# ─────────── binary screen ───────────
@memory.cache(ignore=["dx", "bg", "bow", "beta_Ox"])
def screen_binary(A: str, B: str, rh: float, temp: float, bg: Tuple[float, float], bow: float, dx: float, *, z: float = 0.0, beta_Ox: float = 1.0) -> pd.DataFrame:
    dA = fetch_mp_data(A, ("band_gap", "energy_above_hull"))
    dB = fetch_mp_data(B, ("band_gap", "energy_above_hull"))
    if not (dA and dB):
        return pd.DataFrame()
    dA_Ge = fetch_mp_data(A.replace("Sn", "Ge"), ("band_gap", "energy_above_hull")) if z > 0 else dA
    dB_Ge = fetch_mp_data(B.replace("Sn", "Ge"), ("band_gap", "energy_above_hull")) if z > 0 else dB
    oxA, oxB = oxidation_energy(A), oxidation_energy(B)

    xs = np.arange(0.0, 1.0 + 1e-12, dx)
    Eg = (1 - z) * ((1 - xs) * dA["band_gap"] + xs * dB["band_gap"] - bow * xs * (1 - xs)) + \
         z * ((1 - xs) * dA_Ge["band_gap"] + xs * dB_Ge["band_gap"] - bow * xs * (1 - xs))
    Eh = (1 - z) * ((1 - xs) * dA["energy_above_hull"] + xs * dB["energy_above_hull"]) + \
         z * ((1 - xs) * dA_Ge["energy_above_hull"] + xs * dB_Ge["energy_above_hull"])
    dEox = (1 - xs) * oxA + xs * oxB

    S = _soft_gap(Eg, *bg) * np.exp(-Eh / kT_eff) * np.exp(beta_Ox * dEox / K_OX)
    S /= S.max()

    df = pd.DataFrame({
        "x": xs.round(3),
        "z": round(z, 2),
        "Eg": Eg.round(3),
        "Ehull": Eh.round(4),
        "Eox": dEox.round(3),
        "score": S.round(3),
        "formula": [f"{A}-{B} x={x:.2f} z={z:.2f}" for x in xs],
    })
    return df.sort_values("score", ascending=False).reset_index(drop=True)

# ─────────── ternary screen ───────────
@memory.cache(ignore=["dx", "dy", "bg", "bows", "beta_Ox"])
def screen_ternary(A: str, B: str, C: str, rh: float, temp: float, bg: Tuple[float, float], bows: Dict[str, float], *, dx: float = 0.05, dy: float = 0.05, z: float = 0.0, beta_Ox: float = 1.0) -> pd.DataFrame:
    dA = fetch_mp_data(A, ("band_gap", "energy_above_hull"))
    dB = fetch_mp_data(B, ("band_gap", "energy_above_hull"))
    dC = fetch_mp_data(C, ("band_gap", "energy_above_hull"))
    if not (dA and dB and dC):
        return pd.DataFrame()

    dA_Ge = fetch_mp_data(A.replace("Sn", "Ge"), ("band_gap", "energy_above_hull")) if z > 0 else dA
    dB_Ge = fetch_mp_data(B.replace("Sn", "Ge"), ("band_gap", "energy_above_hull")) if z > 0 else dB
    dC_Ge = fetch_mp_data(C.replace("Sn", "Ge"), ("band_gap", "energy_above_hull")) if z > 0 else dC

    oxA, oxB, oxC = (oxidation_energy(f) for f in (A, B, C))

    xs = np.arange(0.0, 1.0 + 1e-12, dx)
    ys = np.arange(0.0, 1.0 + 1e-
