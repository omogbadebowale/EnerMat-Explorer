
import os
from dotenv import load_dotenv
load_dotenv()

# for secrets fallback on Streamlit Cloud
import streamlit as st

import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Composition

# ── Load Materials Project API key ─────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError(
        "🛑 Please set MP_API_KEY to your 32-character Materials Project API key"
    )
mpr = MPRester(API_KEY)

# ── Supported end-members ──────────────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

# ── Ionic radii (Å) for Goldschmidt tolerance ──────────────────────────
IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81,
}


def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    """Return a dict of the first matching entry's requested fields, or None."""
    docs = mpr.summary.search(formula=formula)
    if not docs:
        return None
    entry = docs[0]
    out: dict = {}
    for f in fields:
        if hasattr(entry, f):
            out[f] = getattr(entry, f)
    return out if out else None


def score_band_gap(bg: float, lo: float, hi: float) -> float:
    """How close bg is to the [lo, hi] window."""
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
    beta: float = 1.0,
) -> pd.DataFrame:
    """Binary screening A–B across x from 0→1."""
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
        hull = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        stability = max(0.0, 1 - hull)
        gap_score = score_band_gap(Eg, lo, hi)
        t = (rA + rX) / (np.sqrt(2) * (rB + rX))
        mu = rB / rX
        form_score = np.exp(-0.5 * ((t - 0.90) / 0.07) ** 2) * np.exp(-0.5 * ((mu - 0.50) / 0.07) ** 2)
        env_pen = 1 + alpha * (rh / 100) + beta * (temp / 100)
        score = form_score * stability * gap_score / env_pen
        rows.append({
            "x": round(x, 3),
            "Eg": round(Eg, 3),
            "stability": round(stability, 3),
            "score": round(score, 3),
            "formula": f"{formula_A}-{formula_B} x={x:.2f}",
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def screen_ternary(
    A: str,
    B: str,
    C: str,
    rh: float,
    temp: float,
    bg: tuple[float, float],
    bows: dict[str, float],
    dx: float = 0.1,
    dy: float = 0.1,
    n_mc: int = 200,
) -> pd.DataFrame:
    """Ternary screening A–B–C over x,y fractions."""
    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    lo, hi = bg
    rows = []
    for x in np.arange(0, 1 + 1e-6, dx):
        for y in np.arange(0, 1 - x + 1e-6, dy):
            z = 1 - x - y
            Eg = (
                (1 - x - y) * dA["band_gap"] + x * dB["band_gap"] + y * dC["band_gap"]
                - bows["AB"] * x * (1 - x - y)
                - bows["AC"] * y * (1 - x - y)
                - bows["BC"] * x * y
            )
            Eh_val = (
                (1 - x - y) * dA["energy_above_hull"] + x * dB["energy_above_hull"] + y * dC["energy_above_hull"]
                + bows["AB"] * x * (1 - x - y)
                + bows["AC"] * y * (1 - x - y)
                + bows["BC"] * x * y
            )
            stability = np.exp(-max(Eh_val, 0) / 0.1)
            gap_score = score_band_gap(Eg, lo, hi)
            score = stability * gap_score
            rows.append({"x": round(x,3), "y": round(y,3), "Eg": round(Eg,3), "score": round(score,3)})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

# alias for backwards compatibility
_summary = fetch_mp_data
import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def load_experimental_bandgaps(csv_file="benchmark_eg.csv"):
    """Load experimental band gap data from CSV."""
    df = pd.read_csv(csv_file)
    return df

@st.cache_data
def fit_vegard_bowing_params(df):
    """
    Fit a quadratic (Vegard's law with bowing) to the band gap data for each binary alloy system.
    Returns a dictionary mapping each system to its Eg_A, Eg_B, and bowing parameter b.
    """
    params = {}
    # Determine grouping key (either 'system' column if present, or formula columns)
    if 'system' in df.columns:
        grouped = df.groupby('system')
    elif 'formula_A' in df.columns and 'formula_B' in df.columns:
        grouped = df.groupby(['formula_A', 'formula_B'])
    else:
        # If no explicit system grouping, treat entire dataset as one group
        grouped = [(None, df)]
    for key, subdf in grouped:
        # Determine system label and end-member formulas
        if isinstance(key, tuple):
            formula_A, formula_B = key
            system_label = f"{formula_A}-{formula_B}"
        else:
            system_label = key if key is not None else "Alloy"
            # Fallback names if formula info not provided
            formula_A = f"{system_label} (A)"
            formula_B = f"{system_label} (B)"
        # Perform quadratic fit: Eg(x) ≈ a*x^2 + b*x + c
        x_vals = subdf['x'].values
        Eg_vals = subdf['Eg_exp'].values
        # Use degree-2 fit if possible, otherwise fall back to lower degree
        deg = 2 if len(x_vals) >= 3 else len(x_vals) - 1
        coeffs = np.polyfit(x_vals, Eg_vals, deg)
        # If we got a linear fit (deg=1), pad it to quadratic form (a=0)
        if deg < 2:
            a, b_coeff, c = 0.0, coeffs[0], coeffs[1]
        else:
            a, b_coeff, c = coeffs  # a = quadratic term, b_coeff = linear term, c = intercept
        # Extract Vegard-bowing parameters:
        Eg_A = c                                # Eg at x=0 (intercept)
        Eg_B = a + b_coeff + c                 # Eg at x=1 (sum of coeffs)
        bowing = a                             # bowing parameter (quadratic term)
        params[system_label] = {
            "Eg_A": Eg_A,
            "Eg_B": Eg_B,
            "b": bowing,
            "formula_A": formula_A,
            "formula_B": formula_B
        }
    return params

# Load data and fit parameters (cached to avoid re-running on each reload)
df_experiment = load_experimental_bandgaps("benchmark_eg.csv")
bandgap_params = fit_vegard_bowing_params(df_experiment)

# Sidebar: Select the alloy system to analyze
system_list = sorted(bandgap_params.keys())
selected_system = st.sidebar.selectbox("Alloy System", system_list)
# Retrieve calibrated parameters for the selected system
param = bandgap_params[selected_system]
EgA_default = param["Eg_A"]
EgB_default = param["Eg_B"]
b_default   = param["b"]
formulaA    = param["formula_A"]
formulaB    = param["formula_B"]

# Sidebar inputs: display and allow override of Eg_A, Eg_B, and b
st.sidebar.markdown("**Band Gap Model Parameters**")
EgA_input = st.sidebar.number_input(f"Eg of {formulaA} (Eg_A)", 
                                    min_value=0.0, max_value=5.0, 
                                    value=float(EgA_default), step=0.01)
EgB_input = st.sidebar.number_input(f"Eg of {formulaB} (Eg_B)", 
                                    min_value=0.0, max_value=5.0, 
                                    value=float(EgB_default), step=0.01)
b_input   = st.sidebar.number_input("Bowing parameter (b)", 
                                    min_value=-1.0, max_value=5.0, 
                                    value=float(b_default), step=0.01)

# Main panel: composition selection and band gap calculation
st.markdown("### Band Gap Interpolation (Vegard + Bowing Model)")
st.write(f"**Selected system:** {selected_system}")
# Choose composition fraction x (fraction of formula_B in the alloy)
x = st.slider(f"Fraction of {formulaB} in the alloy (x)", 
              min_value=0.0, max_value=1.0, value=0.50, step=0.01)
# Compute interpolated band gap using Vegard's law with bowing
Eg_pred = EgA_input * (1 - x) + EgB_input * x - b_input * x * (1 - x)
st.write(f"**Predicted Band Gap (Eg)** at x = {x:.2f}: **{Eg_pred:.3f} eV**")
st.caption("Computed as Eg = (1−x)·Eg_A + x·Eg_B − b·x·(1−x):contentReference[oaicite:3]{index=3}")

# Composite score S calculation (updated to use the calibrated Eg_pred)
st.markdown("### Composite Score S")
# (Assuming other criteria like stability are available in the app context)
# Example: incorporate band gap and stability into a composite score
if 'stability_value' in locals() or 'stability_value' in globals():
    stability_val = locals().get('stability_value', globals().get('stability_value'))
else:
    stability_val = None
if stability_val is not None:
    # **Example** scoring: favor band gap ~1.34 eV (optimal for solar) and high stability
    ideal_Eg = 1.34  # target band gap (e.g. for single-junction solar cell)
    bandgap_score = 1 - abs(Eg_pred - ideal_Eg) / ideal_Eg  # closer to 1.34 eV -> higher score
    bandgap_score = max(0.0, min(1.0, bandgap_score))       # clamp between 0 and 1
    stability_score = max(0.0, min(1.0, stability_val))     # assume stability_val already 0 to 1
    composite_S = 0.5 * bandgap_score + 0.5 * stability_score
    st.write(f"Composite Score **S**: **{composite_S:.3f}**")
else:
    st.write("Composite Score **S** will be calculated once stability data is available.")
