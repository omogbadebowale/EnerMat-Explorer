# ─────────────────────────────────────────────────────────────────────────────
#  app.py  · EnerMat Perovskite Explorer  (stable 2025-06-25)
# ─────────────────────────────────────────────────────────────────────────────
#  • Works with the patched backend/perovskite_utils.py
#  • Ternary mode shows stability + formula, hover tool-tips include Eg
#  • No undefined symbols, no blank-cell issues
# ─────────────────────────────────────────────────────────────────────────────

import io
import os
import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from docx import Document

# ─── API-key check (for Streamlit Cloud as well) ─────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    st.error("🛑  Please set a valid 32-character MP_API_KEY in Streamlit secrets.")
    st.stop()

# ─── Backend utilities (patched version) ─────────────────────────────────────
from backend.perovskite_utils import (
    mix_abx3 as screen_binary,
    screen_ternary,
    END_MEMBERS,
    fetch_mp_data as _summary,
)

# ─── General UI config ───────────────────────────────────────────────────────
st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer – Stable build 2025-06-25")

# Persistent history (simple in-memory stack)
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Sidebar controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose screening type", ["Binary A–B", "Ternary A–B–C"])

    st.header("End-members")
    preset_A = st.selectbox("Preset A", END_MEMBERS, index=0)
    preset_B = st.selectbox("Preset B", END_MEMBERS, index=1)
    custom_A = st.text_input("Custom A (optional)", "").strip()
    custom_B = st.text_input("Custom B (optional)", "").strip()
    A = custom_A or preset_A
    B = custom_B or preset_B

    if mode == "Ternary A–B–C":
        preset_C = st.selectbox("Preset C", END_MEMBERS, index=2)
        custom_C = st.text_input("Custom C (optional)", "").strip()
        C = custom_C or preset_C

    st.header("Environment")
    rh = st.slider("Humidity [%]", 0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)

    st.header("Target band-gap window [eV]")
    bg_lo, bg_hi = st.slider("Gap range", 0.5, 4.0, (1.0, 1.4), 0.01)

    st.header("Model settings")
    bow = st.number_input("Bowing [eV]", 0.0, 1.0, 0.30, 0.05)
    dx = st.number_input("x-step", 0.01, 0.50, 0.05, 0.01)
    if mode == "Ternary A–B–C":
        dy = st.number_input("y-step", 0.01, 0.50, 0.05, 0.01)

    if st.button("🗑 Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()

    GIT_SHA = st.secrets.get("GIT_SHA", "dev")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    st.caption(f"⚙️ Version `{GIT_SHA}` • ⏱ {ts}")
    st.caption("© 2025 Dr Gbadebo Taofeek Yusuf")

# ─── Cached runners ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Running binary screen …", max_entries=20)
def run_binary(**kw):
    return screen_binary(**kw)

@st.cache_data(show_spinner="⏳ Running ternary screen …", max_entries=20)
def run_ternary(**kw):
    return screen_ternary(**kw)

# ─── Run / back buttons ──────────────────────────────────────────────────────
col_run, col_back = st.columns([3, 1])
do_run = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=not st.session_state.history)

# ─── Restore previous or execute new run ─────────────────────────────────────
if do_back:
    st.session_state.history.pop()
    prev = st.session_state.history[-1]
    mode, A, B, rh, temp = prev["mode"], prev["A"], prev["B"], prev["rh"], prev["temp"]
    bg_lo, bg_hi, bow, dx = prev["bg"][0], prev["bg"][1], prev["bow"], prev["dx"]
    if mode == "Ternary A–B–C":
        C, dy = prev["C"], prev["dy"]
    df = prev["df"]
    st.success("Showing previous result")

elif do_run:
    # quick API sanity: make sure end-members exist
    for f in (A, B, *(C,) if mode == "Ternary A–B–C" else ()):
        _summary(f, ["band_gap", "energy_above_hull"])

    if mode == "Binary A–B":
        df = run_binary(
            formula_A=A,
            formula_B=B,
            rh=rh,
            temp=temp,
            bg_window=(bg_lo, bg_hi),
            bowing=bow,
            dx=dx,
        )
    else:
        df = run_ternary(
            A=A,
            B=B,
            C=C,
            rh=rh,
            temp=temp,
            bg=(bg_lo, bg_hi),
            bows={"AB": bow, "AC": bow, "BC": bow},
            dx=dx,
            dy=dy,
        )

    # store history
    st.session_state.history.append(
        dict(
            mode=mode,
            A=A,
            B=B,
            C=C if mode == "Ternary A–B–C" else None,
            rh=rh,
            temp=temp,
            bg=(bg_lo, bg_hi),
            bow=bow,
            dx=dx,
            dy=dy if mode == "Ternary A–B–C" else None,
            df=df,
        )
    )

elif st.session_state.history:  # first load after rerun
    df = st.session_state.history[-1]["df"]

else:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table", "📈 Plot", "📥 Download"])

# ─── Table tab ───────────────────────────────────────────────────────────────
with tab_tbl:
    st.dataframe(df, use_container_width=True, height=420)

# ─── Plot tab ────────────────────────────────────────────────────────────────
with tab_plot:
    if mode == "Binary A–B":
        required = ["stability", "Eg", "score"]
        if not all(c in df.columns for c in required):
            st.warning("Binary results missing required columns.")
            st.stop()

        fig = px.scatter(
            df,
            x="stability",
            y="Eg",
            color="score",
            color_continuous_scale="Turbo",
            hover_data=["formula", "x", "Eg", "stability", "score"],
            width=1200,
            height=800,
        )
        fig.update_traces(marker=dict(size=12, opacity=0.9, line=dict(width=1, color="black")))
        st.plotly_chart(fig, use_container_width=True)

    else:  # ternary
        required = ["x", "y", "Eg", "stability", "score"]
        if not all(c in df.columns for c in required):
            st.warning("Ternary results missing required columns.")
            st.stop()

        fig3d = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="score",
            color="score",
            color_continuous_scale="Turbo",
            hover_data={k: True for k in required},
            width=1200,
            height=900,
        )
        fig3d.update_traces(marker=dict(size=5, opacity=0.9, line=dict(width=1, color="black")))
        st.plotly_chart(fig3d, use_container_width=True)

# ─── Download tab ────────────────────────────────────────────────────────────
with tab_dl:
    st.download_button("📥 CSV", df.to_csv(index=False).encode(), "EnerMat_results.csv", "text/csv")
