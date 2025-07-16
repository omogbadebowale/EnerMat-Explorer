# ─────────────────────────────────────────────────────────────────────────────
#  perovskite_utils_refactor.py – EnerMat backend v10.0  (2025‑07‑16)
#  Major refactor:
#   • decouple Streamlit & secrets (init_mprester)
#   • filesystem cache via joblib.Memory (speeds up MP queries, works offline)
#   • full NumPy vectorisation for binary + ternary grids  (Δx,Δy≤0.01 in ms)
#   • soft Gaussian band‑gap weight instead of hard 0/1 window
#   • exposable oxidation weight  (beta_Ox)  & hull softening (kT_eff)
#   • results returned as DataFrame _and_ optional DesignSpace wrapper
# -----------------------------------------------------------------------------
from __future__ import annotations

import math, os
from functools import lru_cache
from typing import Tuple

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

CALIBRATED_GAPS = {
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
    return _mpr  # type: ignore [return‑value]

# ─────────── MP helpers with on‑disk cache ───────────
@memory.cache(ignore=["fields"])
def fetch_mp_data(formula: str, fields: Tuple[str, ...]) -> dict | None:
    """Cached fetch – stores JSON on disk (≈1 kB per doc)."""
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

# ─────────── scoring functions ───────────
def _soft_gap(Eg: np.ndarray, lo: float, hi: float, sigma: float = 0.10) -> np.ndarray:
    mid = 0.5 * (lo + hi)
    return np.exp(-0.5 * ((Eg - mid) / sigma) ** 2)

# ─────────── binary grid ───────────
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
    z: float = 0.0,
    beta_Ox: float = 1.0,
) -> pd.DataFrame:
    dA = fetch_mp_data(A, ("band_gap", "energy_above_hull"))
    dB = fetch_mp_data(B, ("band_gap", "energy_above_hull"))
    if not (dA and dB):
        return pd.DataFrame()

    hal = next(h for h in ("I", "Br", "Cl") if h in A)
    rA, rB, rX = (IONIC_RADII[k] for k in ("Cs", "Sn", hal))
    oxA, oxB = oxidation_energy(A), oxidation_energy(B)

    xs = np.arange(0.0, 1.0 + 1e‑12, dx)
    Eg = (1 - xs) * dA["band_gap"] + xs * dB["band_gap"] - bow * xs * (1 - xs)
    Eh = (1 - xs) * dA["energy_above_hull"] + xs * dB["energy_above_hull"]
    dEox = (1 - xs) * oxA + xs * oxB

    Sgap = _soft_gap(Eg, *bg)
    S = (
        Sgap
        * np.exp(-Eh / kT_eff)
        * np.exp(beta_Ox * dEox / K_OX)
        * np.exp(-abs((rA + rX) / (math.sqrt(2) * (rB + rX)) - 0.95))
    )

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

# ─────────── ternary grid ───────────
@memory.cache(ignore=["dx", "dy", "bg", "bows", "beta_Ox"])
def screen_ternary(
    A: str,
    B: str,
    C: str,
    rh: float,
    temp: float,
    bg: Tuple[float, float],
    bows: dict[str, float],
    *,
    dx: float = 0.05,
    dy: float = 0.05,
    z: float = 0.0,
    beta_Ox: float = 1.0,
) -> pd.DataFrame:
    dA = fetch_mp_data(A, ("band_gap", "energy_above_hull"))
    dB = fetch_mp_data(B, ("band_gap", "energy_above_hull"))
    dC = fetch_mp_data(C, ("band_gap", "energy_above_hull"))
    if not (dA and dB and dC):
        return pd.DataFrame()

    # optional Ge branch
    if z > 0:
        A_Ge = A.replace("Sn", "Ge"); dA_Ge = fetch_mp_data(A_Ge, ("band_gap", "energy_above_hull")) or dA
        B_Ge = B.replace("Sn", "Ge"); dB_Ge = fetch_mp_data(B_Ge, ("band_gap", "energy_above_hull")) or dB
        C_Ge = C.replace("Sn", "Ge"); dC_Ge = fetch_mp_data(C_Ge, ("band_gap", "energy_above_hull")) or dC
    else:
        dA_Ge = dA_Ge = dA
        dB_Ge = dB_Ge = dB
        dC_Ge = dC_Ge = dC

    oxA, oxB, oxC = (oxidation_energy(f) for f in (A, B, C))

    xs = np.arange(0.0, 1.0 + 1e‑12, dx)
    ys = np.arange(0.0, 1.0 + 1e‑12, dy)
    xx, yy = np.meshgrid(xs, ys)
    mask = xx + yy <= 1.0 + 1e‑12
    x = xx[mask]; y = yy[mask]; w = 1.0 - x - y

    # band gap (Sn)
    bowAB = bows["AB"]; bowAC = bows["AC"]; bowBC = bows["BC"]
    Eg_Sn = (
        w * dA["band_gap"]
      + x * dB["band_gap"]
      + y * dC["band_gap"]
      - bowAB * x * w
      - bowAC * y * w
      - bowBC * x * y
    )
    Eg_Ge = (
        w * dA_Ge["band_gap"]
      + x * dB_Ge["band_gap"]
      + y * dC_Ge["band_gap"]
      - bowAB * x * w
      - bowAC * y * w
      - bowBC * x * y
    )
    Eg = (1.0 - z) * Eg_Sn + z * Eg_Ge

    # Ehull
    Eh_Sn = (
        w * dA["energy_above_hull"]
      + x * dB["energy_above_hull"]
      + y * dC["energy_above_hull"]
    )
    Eh_Ge = (
        w * dA_Ge["energy_above_hull"]
      + x * dB_Ge["energy_above_hull"]
      + y * dC_Ge["energy_above_hull"]
    )
    Eh = (1.0 - z) * Eh_Sn + z * Eh_Ge

    # oxidation (Sn only)
    dEox = w * oxA + x * oxB + y * oxC

    Sgap = _soft_gap(Eg, *bg)
    S = Sgap * np.exp(-Eh / kT_eff) * np.exp(beta_Ox * dEox / K_OX)

    df = pd.DataFrame({
        "x": x.round(3),
        "y": y.round(3),
        "z": round(z, 2),
        "Eg": Eg.round(3),
        "Ehull": Eh.round(4),
        "Eox": dEox.round(3),
        "score": S,
        "formula": [f"{A}-{B}-{C} x={xi:.2f} y={yi:.2f} z={z:.2f}" for xi, yi in zip(x, y)],
    })
    df["score"] /= df["score"].max()
    df["score"] = df["score"].round(3)
    df.sort_values("score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ─────────── optional wrapper ───────────
class DesignSpace(pd.DataFrame):
    """Thin wrapper that keeps metadata and helper plots (to be used in Streamlit)."""
    _metadata = ["meta"]

    def __init__(self, *args, meta: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta = meta or {}

    @property
    def _constructor(self):
        return DesignSpace

    # example helper
    def plot_gap_vs_hull(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        sc = ax.scatter(self["Ehull"], self["Eg"], c=self["score"], s=30)
        ax.set_xlabel("E_hull [eV/atom]")
        ax.set_ylabel("Band gap [eV]")
        plt.colorbar(sc, ax=ax, label="score")
        return ax
