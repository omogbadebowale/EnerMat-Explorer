# app.py  –  EnerMat Perovskite Explorer  v9.6  (2025-07-12)
# Streamlit front-end only.  All heavy lifting lives in backend/perovskite_utils.py

import datetime
import io
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from docx import Document

# ── backend helpers ──────────────────────────────────────────────
from backend.perovskite_utils import (
    mix_abx3        as screen_binary,
    screen_ternary,
    END_MEMBERS,
    fetch_mp_data   as _summary,      # << same spelling as in the backend
)

# ── API key sanity check ─────────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    st.error("🛑 Please set a valid 32-character MP_API_KEY in Streamlit Secrets.")
    st.stop()

# ── Streamlit page ───────────────────────────────────────────────
st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.6")

# ―― session history – enables ⏪ Previous
if "history" not in st.session_state:
    st.session_state.history = []

# ── sidebar controls ─────────────────────────────────────────────
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose screening type", ["Binary A–B", "Ternary A–B–C"])

    st.header("End-members")
    preset_A = st.selectbox("Preset A", END_MEMBERS, index=0)
    preset_B = st.selectbox("Preset B", END_MEMBERS, index=1)
    custom_A = st.text_input("Custom A (optional)").strip()
    custom_B = st.text_input("Custom B (optional)").strip()
    A = custom_A or preset_A
    B = custom_B or preset_B

    if mode == "Ternary A–B–C":
        preset_C = st.selectbox("Preset C", END_MEMBERS, index=2)
        custom_C = st.text_input("Custom C (optional)").strip()
        C = custom_C or preset_C

    st.header("Environment")
    rh   = st.slider("Humidity [%]", 0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)

    st.header("Target band-gap [eV]")
    bg_lo, bg_hi = st.slider("Gap window", 0.5, 3.0, (1.0, 1.4), 0.01)

    st.header("Model settings")
    bow = st.number_input("Bowing (eV, neg ↑ gap)", -1.0, 1.0, -0.15, 0.05)
    dx  = st.number_input("x-step", 0.01, 0.50, 0.05, 0.01)
    if mode == "Ternary A–B–C":
        dy = st.number_input("y-step", 0.01, 0.50, 0.05, 0.01)

    if st.button("🗑 Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()

# ── cached wrappers for speed ────────────────────────────────────
@st.cache_data(show_spinner="⏳ Running binary screen…", max_entries=20)
def _run_binary(*args, **kws):
    return screen_binary(*args, **kws)

@st.cache_data(show_spinner="⏳ Running ternary screen…", max_entries=10)
def _run_ternary(*args, **kws):
    return screen_ternary(*args, **kws)

# ── main buttons ─────────────────────────────────────────────────
col_run, col_back = st.columns([3, 1])
run_clicked   = col_run.button("▶ Run screening", type="primary")
back_clicked  = col_back.button("⏪ Previous", disabled=not st.session_state.history)

# -----------------------------------------------------------------
if back_clicked:
    st.session_state.history.pop()
    df = st.session_state.history[-1]["df"]
    mode = st.session_state.history[-1]["mode"]
    st.success("Showing previous result")

elif run_clicked:
    if mode == "Binary A–B":
        df = _run_binary(
            A, B, rh, temp, (bg_lo, bg_hi), bow, dx
        )
    else:
        df = _run_ternary(
            A, B, C, rh, temp, (bg_lo, bg_hi),
            {"AB": bow, "AC": bow, "BC": bow},
            dx, dy
        )

    st.session_state.history.append({"df": df, "mode": mode})

elif st.session_state.history:
    df  = st.session_state.history[-1]["df"]
    mode = st.session_state.history[-1]["mode"]
else:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

# ── UI: table / plot / download ──────────────────────────────────
tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table", "📈 Plot", "📥 Download"])

with tab_tbl:
    st.dataframe(df, use_container_width=True, height=400)

with tab_plot:
    if mode == "Binary A–B":
        if {"Eg", "score", "Ehull"} <= set(df.columns):
            fig = px.scatter(df, x="Ehull", y="Eg", color="score",
                             color_continuous_scale="Turbo",
                             hover_data=["formula", "x", "Eox"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Missing columns for plotting.")
    else:
        if {"x", "y", "score"} <= set(df.columns):
            fig3 = px.scatter_3d(df, x="x", y="y", z="score",
                                 color="score",
                                 hover_data=["formula", "Eg", "Ehull", "Eox"])
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.error("Missing columns for ternary plot.")

with tab_dl:
    st.download_button("📥 Download CSV",
                       df.to_csv(index=False).encode(),
                       "EnerMat_results.csv", "text/csv")

    top = df.iloc[0]
    label = top.formula
    txt = (f"EnerMat report ({datetime.date.today()})\n"
           f"Top candidate: {label}\n"
           f"Eg: {top.Eg}\nEhull: {top.Ehull}\nEox: {top.Eox}\n"
           f"Score: {top.score}\n")
    st.download_button("📄 Download TXT", txt, "EnerMat_report.txt", "text/plain")

    doc = Document()
    doc.add_heading("EnerMat report", 0)
    doc.add_paragraph(txt)
    buf = io.BytesIO(); doc.save(buf); buf.seek(0)
    st.download_button("📝 Download DOCX", buf,
                       "EnerMat_report.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
