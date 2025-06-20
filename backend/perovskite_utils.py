import os
import math
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Composition

# ── Load Materials Project API key ─────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 Please set MP_API_KEY to your 32-char Materials Project API key")
mpr = MPRester(API_KEY)

# ── Supported end-members ──────────────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

# ── Ionic radii for formability ───────────────────────────────────────
IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81
}

# backward alias (assigned after fetch_mp_data)
_summary = None

def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    """Return first entry's requested fields as a dict, or None if missing."""
    docs = mpr.summary.search(material_ids_or_formula=formula)
    for entry in docs:
        out: dict = {}
        for f in fields:
            if hasattr(entry, f):
                out[f] = getattr(entry, f)
        return out or None
    return None

# set alias
_summary = fetch_mp_data


def score_band_gap(bg: float, lo: float, hi: float) -> float:
    """Score for how close bg is to the [lo, hi] window."""
    if bg < lo:
        return max(0.0, 1 - (lo - bg) / (hi - lo))
    if bg > hi:
        return max(0.0, 1 - (bg - hi) / (hi - lo))
    return 1.0


def mix_abx3(
    formula_A: str,
    formula_B: str,
    rh: float,
    temp: float,
    bg_window: tuple[float, float],
    bowing: float = 0.0,
    dx: float = 0.05,
    alpha: float = 1.0,
    beta: float = 1.0
) -> pd.DataFrame:
    lo, hi = bg_window

    dA = fetch_mp_data(formula_A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(formula_B, ["band_gap", "energy_above_hull"])
    if not (dA and dB):
        return pd.DataFrame()

    comp = Composition(formula_A)
    A_site = next(e.symbol for e in comp.elements if e.symbol in IONIC_RADII)
    B_site = next(e.symbol for e in comp.elements if e.symbol in {"Pb", "Sn"})
    X_site = next(e.symbol for e in comp.elements if e.symbol in {"I", "Br", "Cl"})
    rA, rB, rX = IONIC_RADII[A_site], IONIC_RADII[B_site], IONIC_RADII[X_site]

    rows = []
    for x in np.arange(0, 1 + 1e-6, dx):
        Eg = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bowing * x * (1 - x)
        t = (rA + rX) / (math.sqrt(2) * (rB + rX))
        mu = rB / rX
        form_score = float(0.75 <= t <= 1.0 and 0.41 <= mu <= 0.9)

        hull = dA.get("energy_above_hull", 0.0)
        stability = math.exp(-max(hull, 0) / 0.10)

        bg_score = score_band_gap(Eg, lo, hi)
        env_pen = math.exp((rh / 85) + (temp / 100))
        comp_score = form_score * stability * bg_score / env_pen

        rows.append({
            "x": round(x, 4),
            "band_gap": round(Eg, 4),
            "stability": round(stability, 4),
            "score": round(comp_score, 4),
            "formula": f"{formula_A}-{formula_B} x={x:.2f}"
        })

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def screen_ternary(
    A: str, B: str, C: str,
    rh: float, temp: float,
    bg: tuple[float, float],
    bows: dict[str, float],
    dx: float = 0.1, dy: float = 0.1,
    n_mc: int = 200
) -> pd.DataFrame:
    lo, hi = bg
    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    rows = []
    for x in np.arange(0, 1 + 1e-6, dx):
        for y in np.arange(0, 1 - x + 1e-6, dy):
            z = 1 - x - y
            Eg = (
                (1 - x - y) * dA["band_gap"] + x * dB["band_gap"] + y * dC["band_gap"]
                - bows['AB'] * x * (1 - x - y)
                - bows['AC'] * y * (1 - x - y)
                - bows['BC'] * x * y
            )
            Eh = (
                (1 - x - y) * dA["energy_above_hull"] + x * dB["energy_above_hull"] + y * dC["energy_above_hull"]
                + bows['AB'] * x * (1 - x - y)
                + bows['AC'] * y * (1 - x - y)
                + bows['BC'] * x * y
            )
            compA, compB, compX = Composition(A), Composition(B), Composition(A)
            rA = IONIC_RADII[next(e.symbol for e in compA.elements if e.symbol in IONIC_RADII)]
            rB = IONIC_RADII[next(e.symbol for e in compB.elements if e.symbol in {"Pb","Sn"})]
            rX = IONIC_RADII[next(e.symbol for e in compX.elements if e.symbol in {"I","Br","Cl"})]
            t = (rA + rX) / (math.sqrt(2) * (rB + rX))
            mu = rB / rX
            form_score = math.exp(-0.5 * ((t - 0.90) / 0.07) ** 2) * math.exp(-0.5 * ((mu - 0.50) / 0.07) ** 2)
            stability = math.exp(-max(Eh, 0) / 0.10)
            gap_score = score_band_gap(Eg, lo, hi)
            env_pen = 1 + rh / 100 + temp / 100
            comp_score = form_score * stability * gap_score / env_pen
            rows.append({
                "x": round(x, 4),
                "y": round(y, 4),
                "Eg": round(Eg, 4),
                "Eh": round(Eh, 4),
                "score": round(comp_score, 4)
            })
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
