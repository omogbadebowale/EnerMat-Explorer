# app.py – EnerMat Perovskite Explorer v9.6  (2025‑07‑14, Ge‑ready)
# -----------------------------------------------------------------------------
"""Streamlit front‑end for fast alloy screening + auto‑report generation.

* Supports binary (A–B) and ternary (A–B–C) halide perovskites.
* Optional Sn↔Ge B‑site mixing via slider * z * (0 – 0.30).
* Uses a single YAML file (data/end_members.yaml) as ground‑truth for all
  end‑member properties – reproducible & reviewer‑friendly.

Assumes back‑end helpers live in `src/` (see README).
"""

from __future__ import annotations
import io, os, datetime, functools, pathlib

import streamlit as st
import pandas as pd
import plotly.express as px
from docx import Document

# ──────────────────────────────────────────────────────────────────────────────
# 1  Materials‑Project API key (required by the back‑end)
# ──────────────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY", "")
if len(API_KEY) != 32:
    st.error("🛑  A valid 32‑character MP_API_KEY must be placed in *Secrets*.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# 2  Local helpers
# ──────────────────────────────────────────────────────────────────────────────
from backend.perovskite_utils import mix_abx3 as screen_binary, screen_ternary
from src.materials import load_end_members  # single source of truth

END_MEMBERS = list(load_end_members())  # → ["CsSnBr3", "CsSnCl3", …]

# ──────────────────────────────────────────────────────────────────────────────
# 3  Streamlit page configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="EnerMat Explorer", layout="wide")
st.title("🔬  EnerMat **Perovskite** Explorer v9.6")

# ──────────────────────────────────────────────────────────────────────────────
# 4  Session state (simple history stack)
# ──────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

# ──────────────────────────────────────────────────────────────────────────────
# 5  Sidebar – I/O controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose screening type", ["Binary A–B", "Ternary A–B–C"])

    st.header("End‑members")
    preset_A = st.selectbox("Preset A", END_MEMBERS, index=0)
    preset_B = st.selectbox("Preset B", END_MEMBERS, index=1)
    custom_A = st.text_input("Custom A (optional)").strip()
    custom_B = st.text_input("Custom B (optional)").strip()

    A = custom_A or preset_A
    B = custom_B or preset_B

    if mode.startswith("Ternary"):
        preset_C = st.selectbox("Preset C", END_MEMBERS, index=2)
        custom_C = st.text_input("Custom C (optional)").strip()
        C = custom_C or preset_C

    st.header("Environment")
    rh   = st.slider("Humidity [%]",      0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)

    st.header("Target band‑gap [eV]")
    bg_lo, bg_hi = st.slider("Gap window", 0.50, 3.00, (1.00, 1.40), 0.01)

    st.header("Model settings")
    bow = st.number_input("Bowing (eV, negative ⇒ gap↑)", -1.0, 1.0, -0.15, 0.05)
    dx  = st.number_input("x‑step", 0.01, 0.50, 0.05, 0.01)
    if mode.startswith("Ternary"):
        dy = st.number_input("y‑step", 0.01, 0.50, 0.05, 0.01)

    # Optional Ge slider (only shown when b‑site mixing enabled in config)
    if st.session_state.get("b_site_mixing", True):
        z = st.slider("Ge fraction z", 0.00, 0.30, 0.10, 0.05,
                      help="B‑site Ge²⁺ fraction in CsSn₁₋zGe_z(Br,Cl)₃")
    else:
        z = 0.0

    if st.button("🗑  Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()

    st.caption(
        f"⚙️ Build SHA : {st.secrets.get('GIT_SHA', 'dev')}   •  "
        f"🕒 {datetime.datetime.now():%Y‑%m‑%d %H:%M}"
    )

# ──────────────────────────────────────────────────────────────────────────────
# 6  Cached wrappers (keep things snappy)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Screening …", max_entries=20)
def _run_binary(*args, **kws):
    return screen_binary(*args, **kws)

@st.cache_data(show_spinner="⏳ Screening …", max_entries=10)
def _run_ternary(*args, **kws):
    return screen_ternary(*args, **kws)

# ──────────────────────────────────────────────────────────────────────────────
# 7  Action buttons
# ──────────────────────────────────────────────────────────────────────────────
col_run, col_prev = st.columns([3, 1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_prev = col_prev.button("⏪ Previous", disabled=not st.session_state.history)

# ──────────────────────────────────────────────────────────────────────────────
# 8  Handle *Previous*
# ──────────────────────────────────────────────────────────────────────────────
if do_prev:
    st.session_state.history.pop()         # discard current state
    if st.session_state.history:
        st.success("Showing previous result")
    else:
        st.experimental_rerun()

# ──────────────────────────────────────────────────────────────────────────────
# 9  Handle *Run screening*
# ──────────────────────────────────────────────────────────────────────────────
if do_run:
    # ---- simple formula sanity check -------------------------------------
    formulas = [A, B] if mode.startswith("Binary") else [A, B, C]
    for fml in formulas:
        if fml not in END_MEMBERS:
            st.error(f"❌ Unknown end‑member formula: {fml}")
            st.stop()

    # ---- run the actual screen ------------------------------------------
    if mode.startswith("Binary"):
        df = _run_binary(A, B, rh, temp, (bg_lo, bg_hi), bow, dx, z=z)
    else:
        df = _run_ternary(A, B, C, rh, temp, (bg_lo, bg_hi), bow, dx, dy, z)

    st.session_state.history.append({"mode": mode, "df": df})

# ──────────────────────────────────────────────────────────────────────────────
# 10  Nothing to show yet → prompt user
# ──────────────────────────────────────────────────────────────────────────────
if not st.session_state.history:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# 11  Active DataFrame / mode
# ──────────────────────────────────────────────────────────────────────────────
active   = st.session_state.history[-1]
mode     = active["mode"]
df: pd.DataFrame = active["df"].copy()

# ──────────────────────────────────────────────────────────────────────────────
# 12  Tabs: table • plot • download
# ──────────────────────────────────────────────────────────────────────────────
_tab_tbl, _tab_plot, _tab_dl = st.tabs(["📊 Table", "📈 Plot", "📥 Download"])

with _tab_tbl:
    st.dataframe(df, use_container_width=True, height=420)

with _tab_plot:
    if mode.startswith("Binary") and {"Ehull", "Eg"}.issubset(df.columns):
        fig = px.scatter(
            df, x="Ehull", y="Eg", color="score", color_continuous_scale="Turbo",
            hover_data=df.columns, height=800,
        )
        fig.update_traces(marker=dict(size=9, line=dict(width=1, color="black")))
        st.plotly_chart(fig, use_container_width=True)
    elif mode.startswith("Ternary") and {"x", "y", "score"}.issubset(df.columns):
        fig = px.scatter_3d(
            df, x="x", y="y", z="score", color="score",
            color_continuous_scale="Turbo", hover_data=df.columns, height=820,
        )
        fig.update_traces(marker=dict(size=4))
        st.plotly_chart(fig, use_container_width=True)

with _tab_dl:
    st.download_button(
        "📥 Download CSV", df.to_csv(index=False).encode(),
        "EnerMat_results.csv", mime="text/csv",
    )

# ──────────────────────────────────────────────────────────────────────────────
# 13  Auto‑report (TXT & DOCX)
# ──────────────────────────────────────────────────────────────────────────────
# Top row == best candidate (df already sorted by score in back‑end)
_top = df.iloc[0]
formula = str(_top["formula"])
coords = [
    f"{c}={_top[c]:.2f}"
    for c in ("x", "y", "z", "ge_frac") if c in _top and pd.notna(_top[c])
]
coord_txt = ", ".join(coords)
label = formula if len(df) == 1 else f"{formula} ({coord_txt})"

_txt = (
    "EnerMat auto‑report  "
    f"{datetime.date.today()}\n"
    f"Top candidate   : {label}\n"
    f"Band‑gap [eV]   : {_top['Eg']}\n"
    f"Ehull [eV/at.]  : {_top['Ehull']}\n"
    f"Eox_e [eV/e⁻]   : {_top.get('Eox_e', 'N/A')}\n"
    f"Score           : {_top['score']}\n"
)

st.download_button("📄 Download TXT", _txt, "EnerMat_report.txt", mime="text/plain")

_doc = Document()
_doc.add_heading("EnerMat Report", level=0)
_doc.add_paragraph(f"Date: {datetime.date.today()}")
_doc.add_paragraph(f"Top candidate: {label}")
_t = _doc.add_table(rows=1, cols=2)
_t.style = "LightShading-Accent1"
_hdr = _t.rows[0].cells
_hdr[0].text, _hdr[1].text = "Property", "Value"
for k in ("Eg", "Ehull", "Eox_e", "score"):
    if k in _top:
        row = _t.add_row().cells
        row[0].text, row[1].text = k, str(_top[k])
_buf = io.BytesIO(); _doc.save(_buf); _buf.seek(0)
st.download_button(
    "📝 Download DOCX", _buf, "EnerMat_report.docx",
    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
)
