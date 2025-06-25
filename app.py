# ============================================================
# EnerMat‑Explorer  v9.8  (2025‑06‑25)
# Two‑file code drop‑in:  backend/perovskite_utils.py  +  app.py
# Copy each block into its *separate* file path in the repo tree.
# ============================================================

# ─────────────────────────────────────────────────────────────
# File: backend/perovskite_utils.py
# ─────────────────────────────────────────────────────────────
"""Utility layer for EnerMat‑Explorer – Materials‑Project access,
score models & binary / ternary screens.
Updated 2025‑06‑25 to address reviewer points:
  • stability  =  exp(−E_hull / 0.06 eV)  (≈ metastability window)
  • fetch_mp_data now chooses the *lowest‑energy* polymorph
  • gap_score & env_pen surfaced in the returned DataFrames
  • screen_ternary gains RH / T penalty for parity with binary mode
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition

# ── API key ───────────────────────────────────────────────────
load_dotenv()
import streamlit as st    # secrets fallback on Streamlit Cloud
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑  Set a valid 32‑char MP_API_KEY (environment or secrets).")

mpr = MPRester(API_KEY)

# ── Reference lists ───────────────────────────────────────────
END_MEMBERS = [
    "CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3",
]

IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81,
}

# ── Helpers ───────────────────────────────────────────────────

def fetch_mp_data(formula: str, fields: list[str]) -> dict | None:
    """Return requested fields for the *ground‑state* MP entry."""
    docs = mpr.summary.search(formula=formula)
    if not docs:
        return None
    best = min(docs, key=lambda d: d.energy_above_hull or 9e9)
    return {f: getattr(best, f) for f in fields if hasattr(best, f)}


def score_band_gap(bg: float, lo: float, hi: float) -> float:
    """Triangular window – unity inside [lo,hi], taper to zero at ±(hi‑lo)."""
    if bg < lo:
        return max(0.0, 1 - (lo - bg) / (hi - lo))
    if bg > hi:
        return max(0.0, 1 - (bg - hi) / (hi - lo))
    return 1.0


# ── Binary screen ─────────────────────────────────────────────

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
    """Screen A‑B mix across x ∈ [0,1] with step dx."""
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

    rows: list[dict] = []
    for x in np.arange(0, 1 + 1e-6, dx):
        Eg   = (1 - x) * dA["band_gap"] + x * dB["band_gap"] - bowing * x * (1 - x)
        Eh   = (1 - x) * dA["energy_above_hull"] + x * dB["energy_above_hull"]
        stability = np.exp(-max(Eh, 0) / 0.06)     # ≈ kT @ 700 K
        gap_score = score_band_gap(Eg, lo, hi)
        t  = (rA + rX) / (np.sqrt(2) * (rB + rX))
        mu = rB / rX
        form_score = np.exp(-0.5 * ((t  - 0.90) / 0.07) ** 2) * \
                     np.exp(-0.5 * ((mu - 0.50) / 0.07) ** 2)
        env_pen = 1 + alpha * (rh / 100) + beta * (temp / 100)
        S = form_score * stability * gap_score / env_pen
        rows.append({
            "x": round(x, 3),
            "Eg": round(Eg, 3),
            "stability": round(stability, 3),
            "gap_score": round(gap_score, 3),
            "score": round(S, 3),
            "formula": f"{formula_A}-{formula_B} x={x:.2f}",
        })

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


# ── Ternary screen ────────────────────────────────────────────

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
    alpha: float = 1.0,
    beta: float = 1.0,
) -> pd.DataFrame:
    """Grid search over ternary fractions (x,y; z=1‑x‑y)."""
    dA = fetch_mp_data(A, ["band_gap", "energy_above_hull"])
    dB = fetch_mp_data(B, ["band_gap", "energy_above_hull"])
    dC = fetch_mp_data(C, ["band_gap", "energy_above_hull"])
    if not (dA and dB and dC):
        return pd.DataFrame()

    lo, hi = bg
    rows: list[dict] = []
    for x in np.arange(0, 1 + 1e-6, dx):
        for y in np.arange(0, 1 - x + 1e-6, dy):
            z = 1 - x - y
            Eg = (
                z * dA["band_gap"] + x * dB["band_gap"] + y * dC["band_gap"]
                - bows["AB"] * x * z - bows["AC"] * y * z - bows["BC"] * x * y
            )
            Eh = (
                z * dA["energy_above_hull"] + x * dB["energy_above_hull"] + y * dC["energy_above_hull"]
                + bows["AB"] * x * z + bows["AC"] * y * z + bows["BC"] * x * y
            )
            stability = np.exp(-max(Eh, 0) / 0.06)
            gap_score = score_band_gap(Eg, lo, hi)
            env_pen   = 1 + alpha * (rh / 100) + beta * (temp / 100)
            S = stability * gap_score / env_pen
            rows.append({
                "x": round(x, 3),
                "y": round(y, 3),
                "Eg": round(Eg, 3),
                "stability": round(stability, 3),
                "gap_score": round(gap_score, 3),
                "score": round(S, 3),
            })

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


# ── Convenience alias for app.py legacy import ────────────────
_summary = fetch_mp_data


# ============================================================
# File: app.py   (Streamlit front‑end)
# ============================================================

"""EnerMat‑Explorer  Streamlit UI – v9.8  (2025‑06‑25)
Major fixes:
  • removed stray st.header inside cached function (NameError)
  • plot guards for empty DataFrames
  • adds optional humidity‑curve button (Fig 3 generator)
"""

import io, os, datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from docx import Document
from backend.perovskite_utils import (
    mix_abx3 as screen,
    screen_ternary,
    END_MEMBERS,
    _summary,
)

# ── Config & API key check ───────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    st.error("🛑  Set a 32‑char MP_API_KEY in Secrets ⟶ Deploy →  Secrets tab.")
    st.stop()

st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.8")

# ── Session state ────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ─────────────────────────────────────────────────-
with st.sidebar:
    mode = st.radio("Choose screening type", ["Binary A–B", "Ternary A–B–C"])
    st.markdown("### End‑members")
    preset_A = st.selectbox("Preset / custom A", END_MEMBERS, index=0)
    preset_B = st.selectbox("Preset / custom B", END_MEMBERS, index=1)
    custom_A = st.text_input("", key="custA")
    custom_B = st.text_input("", key="custB")
    A = custom_A.strip() or preset_A
    B = custom_B.strip() or preset_B
    if mode == "Ternary A–B–C":
        preset_C = st.selectbox("Preset / custom C", END_MEMBERS, index=2)
        custom_C = st.text_input("", key="custC")
        C = custom_C.strip() or preset_C

    st.markdown("### Environment")
    rh   = st.slider("Humidity [%]", 0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)

    st.markdown("### Physics window")
    bg_lo, bg_hi = st.slider("Target gap [eV]", 0.5, 3.0, (1.0, 1.4), 0.01)

    st.markdown("### Model settings")
    bow = st.number_input("Bowing [eV]", 0.0, 1.0, 0.30, 0.05)
    dx  = st.number_input("x‑step", 0.01, 0.5, 0.05, 0.01)
    if mode == "Ternary A–B–C":
        dy  = st.number_input("y‑step", 0.01, 0.5, 0.05, 0.01)

    if st.button("🗑 Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()

# ── Cached runner (no UI calls!) ─────────────────────────────
@st.cache_data(show_spinner="⏳ Running screening…", max_entries=20)
def run_screen(formula_A, formula_B, rh, temp, bg, bow, dx):
    return screen(
        formula_A=formula_A,
        formula_B=formula_B,
        rh=rh,
        temp=temp,
        bg_window=bg,
        bowing=bow,
        dx=dx,
    )

# ── Run / back buttons ───────────────────────────────────────
col_run, col_back = st.columns([3, 1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=not st.session_state.history)

if do_back:
    prev = st.session_state.history.pop()
    mode, df = prev["mode"], prev["df"]
    A, B, rh, temp, bow, dx, (bg_lo, bg_hi) = (
        prev[k] for k in ("A", "B", "rh", "temp", "bow", "dx", "bg")
    )
    if mode == "Ternary A–B–C":
        C, dy = prev["C"], prev["dy"]
    st.success("Previous result loaded.")

elif do_run:
    # Basic validation (avoids empty DF crash)
    if not (_summary(A, ["band_gap"]) and _summary(B, ["band_gap"])):
        st.error("One or more formulas not recognised by Materials Project.")
        st.stop()
    if mode == "Ternary A–B–C" and not _summary(C, ["band_gap"]):
        st.error("Third formula invalid.")
        st.stop()

    if mode == "Binary A–B":
        df = run_screen(A, B, rh, temp, (bg_lo, bg_hi), bow, dx)
    else:
        df = screen_ternary(
            A, B, C,
            rh=rh, temp=temp,
            bg=(bg_lo, bg_hi),
            bows={"AB": bow, "AC": bow, "BC": bow},
            dx=dx, dy=dy,
        )
    if df.empty:
        st.error("No data returned – check formulas or API quota.")
        st.stop()

    st.session_state.history.append({
        "mode": mode, "A": A, "B": B, "rh": rh, "temp": temp,
        "bow": bow, "dx": dx, "bg": (bg_lo, bg_hi), "df": df,
        **({"C": C, "dy": dy} if mode == "Ternary A–B–C" else {}),
    })

elif st.session_state.history:
    df = st.session_state.history[-1]["df"]
else:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────
TA, TP, TD = st.tabs(["📊 Table", "📈 Plot", "📥 Download"])

with TA:
    st.subheader("Candidate Results")
    st.dataframe(df, use_container_width=True, height=420)

with TP:
    if mode == "Binary A–B":
        if df.empty:
            st.warning("No data to plot.")
        else:
            fig = px.scatter(
                df, x="stability", y="Eg", color="score",
                color_continuous_scale="Turbo",
                hover_data=["formula", "x", "Eg", "stability", "score"],
                width=1200, height=700,
            )
            fig.update_traces(marker=dict(size=10, line=dict(width=1,color="black")))
            st.plotly_chart(fig, use_container_width=True)
            if st.button("📈 Show S vs RH curve"):
                from functools import lru_cache
                @lru_cache(maxsize=16)
                def humidity_curve(a,b,xfix,temp,bow):
                    rec=[]
                    for RH in range(0,101,10):
                        dd = screen(a,b,rh=RH,temp=temp,bg_window=(bg_lo,bg_hi),bowing=bow,dx=0.005)
                        row = dd.iloc[(dd["x"]-xfix).abs().argmin()]
                        rec.append({"RH %":RH,"S":row.score})
                    return pd.DataFrame(rec)
                curve_df = humidity_curve(A,B,0.30,temp,bow)
                fig2 = px.line(curve_df,x="RH %",y="S",markers=True,title="Humidity curve x=0.30")
                st.plotly_chart(fig2,use_container_width=True)
    else:
        if df.empty:
            st.warning("No data to plot.")
        else:
            fig3d = px.scatter_3d(
                df, x="x", y="y", z="score", color="score",
                color_continuous_scale="Turbo",
                hover_data=["x","y","Eg","score"],
                width=1200,height=800,
            )
            st.plotly_chart(fig3d, use_container_width=True)

with TD:
    csv = df.to_csv(index=False).encode()
    st.download_button("📥 Download CSV", csv, "EnerMat_results.csv", "text/csv")
    top = df.iloc[0]
    if mode == "Binary A–B":
        label = top.formula
    else:
        label = f"{A}-{B}-{C} x={top.x:.2f} y={top.y:.2f}"
    txt = f"EnerMat report ({datetime.date.today()})\nTop candidate : {label}\nBand-gap      : {top.Eg}\nStability     : {top.stability}\nGap factor    : {top.gap_score}\nComposite S   : {top.score}\n"
    st.download_button("📄 Download TXT", txt, "EnerMat_report.txt", "text/plain")
    doc = Document(); doc.add_heading("EnerMat Report",0)
    rows=[("Band-gap",top.Eg),("Stability",top.stability),("Gap factor",top.gap_score),("Composite S",top.score)]
    table=doc.add_table(rows=1,cols=2); table.rows[0].cells[0].text="Property"; table.rows[0].cells[1].text="Value"
    for k,v in rows:
        r=table.add_row(); r.cells[0].text=k; r.cells[1].text=str(v)
    buf=io.BytesIO(); doc.save(buf); buf.seek(0)
    st.download_button("📝 Download DOCX",buf,"EnerMat_report.docx","application/vnd.openxmlformats-officedocument.wordprocessingml.document")
