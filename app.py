import io 
import os
import datetime
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from docx import Document
from mp_api.client import MPRester

from backend.perovskite_utils import (
    mix_abx3 as screen,
    screen_ternary,
    END_MEMBERS,
    fetch_mp_data as _summary
)

# ───────────────────────────────────── App Config ─────────────────────────────
st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.6")

# ─────────────────────────────────── Session History ──────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ───────────────────────────────────── Sidebar ────────────────────────────────
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose screening type", ["Binary A–B", "Ternary A–B–C"], key="mode")

    st.header("End-members")
    preset_A = st.selectbox("Preset A", END_MEMBERS, index=0, key="preset_A")
    preset_B = st.selectbox("Preset B", END_MEMBERS, index=1, key="preset_B")
    custom_A = st.text_input("Custom A (optional)", "", key="custom_A").strip()
    custom_B = st.text_input("Custom B (optional)", "", key="custom_B").strip()
    A = custom_A or preset_A
    B = custom_B or preset_B

    if mode == "Ternary A–B–C":
        preset_C = st.selectbox("Preset C", END_MEMBERS, index=2, key="preset_C")
        custom_C = st.text_input("Custom C (optional)", "", key="custom_C").strip()
        C = custom_C or preset_C
    else:
        C = None

    st.header("Environment")
    rh = st.slider("Humidity [%]", 0, 100, 50, key="rh")
    temp = st.slider("Temperature [°C]", -20, 100, 25, key="temp")

    st.header("Target gap [eV]")
    bg_lo, bg_hi = st.slider("Gap window [eV]", 0.5, 3.0, (1.0, 1.4), 0.01, key="bg")

    st.header("Model knobs")
    bow = st.number_input("Bowing [eV]", 0.0, 1.0, 0.30, 0.05, key="bow")
    dx = st.number_input("x-step", 0.01, 0.50, 0.05, 0.01, key="dx")
    dy = None
    if mode == "Ternary A–B–C":
        dy = st.number_input("y-step", 0.01, 0.50, 0.05, 0.01, key="dy")

    if st.button("🗑 Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()

    st.caption("© 2025 Dr Gbadebo Taofeek Yusuf")
    GIT_SHA = st.secrets.get("GIT_SHA", "dev")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    st.caption(f"⚙️ Version: `{GIT_SHA}` • ⏱ {ts}")

# ────────────────────────────────── Backend Call ──────────────────────────────
@st.cache_data(show_spinner="⏳ Screening …")
def run_screen(**kw):
    return screen(**kw)

# ───────────────────────────────── Run / Back Logic ───────────────────────────
col_run, col_back = st.columns([3, 1])
do_run = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=len(st.session_state.history) < 1)

if do_back and st.session_state.history:
    params = st.session_state.history.pop()
    mode = params["mode"]
    A, B, C = params["A"], params["B"], params.get("C")
    rh, temp = params["rh"], params["temp"]
    bg_lo, bg_hi = params["bg_lo"], params["bg_hi"]
    bow, dx, dy = params["bow"], params["dx"], params.get("dy")
    df = params["df"]
    st.success("Showing previous result")

elif do_run:
    # validate formulas
    try:
        docA = _summary(A, ["band_gap", "energy_above_hull"])
        docB = _summary(B, ["band_gap", "energy_above_hull"])
    except Exception as e:
        st.error(f"❌ Error querying Materials Project: {e}")
        st.stop()
    if not docA or not docB:
        st.error("❌ Invalid formula(s)")
        st.stop()

    # run screening
    if mode == "Binary A–B":
        df = run_screen(A=A, B=B, rh=rh, temp=temp, bg=(bg_lo, bg_hi), bow=bow, dx=dx)
    else:
        try:
            df = screen_ternary(
                A=A, B=B, C=C,
                rh=rh, temp=temp,
                bg=(bg_lo, bg_hi),
                bows={"AB": bow, "AC": bow, "BC": bow},
                dx=dx, dy=dy, n_mc=200
            )
        except Exception as e:
            st.error(f"❌ Ternary error: {e}")
            st.stop()

    if df.empty:
        st.error("No candidates found – try widening your gap or composition window.")
        st.stop()

    # rename Eg → band_gap for consistency
    df = df.rename(columns={"Eg": "band_gap"})
    # push to history (store all params + df)
    st.session_state.history.append({
        "mode": mode, "A": A, "B": B, "C": C,
        "rh": rh, "temp": temp,
        "bg_lo": bg_lo, "bg_hi": bg_hi,
        "bow": bow, "dx": dx, "dy": dy,
        "df": df
    })

elif st.session_state.history:
    # first load (no buttons pressed)
    last = st.session_state.history[-1]
    df = last["df"]
else:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

# ─────────────────────────────────── Tabs ─────────────────────────────────────
tab_tbl, tab_plot, tab_dl, tab_bench, tab_results = st.tabs([
    "📊 Table", "📈 Plot", "📥 Download", "⚖ Benchmark", "📑 Results Summary"
])

# Table Tab
with tab_tbl:
    params = pd.DataFrame({
        "Parameter": ["Humidity [%]", "Temperature [°C]", "Gap window [eV]", "Bowing [eV]", "x-step"] + (["y-step"] if mode == "Ternary A–B–C" else []),
        "Value": [rh, temp, f"{bg_lo:.2f}–{bg_hi:.2f}", bow, dx] + ([dy] if mode == "Ternary A–B–C" else [])
    })
    st.markdown("**Run parameters**")
    st.table(params)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**A-endmember: {A}**")
        st.write(f"MP band gap: {docA['band_gap']:.2f} eV")
        st.write(f"MP E_above_hull: {docA['energy_above_hull']:.3f} eV/atom")
    with c2:
        st.markdown(f"**B-endmember: {B}**")
        st.write(f"MP band gap: {docB['band_gap']:.2f} eV")
        st.write(f"MP E_above_hull: {docB['energy_above_hull']:.3f} eV/atom")

    st.dataframe(df, height=400, use_container_width=True)

# Plot Tab
with tab_plot:
    if mode == "Binary A–B":
        st.caption("ℹ️ Hover circles; scroll to zoom; drag to pan")
        top_cut = df.score.quantile(0.80)
        df['is_top'] = df.score >= top_cut
        fig = px.scatter(
            df, x='stability', y='band_gap',
            color='score', color_continuous_scale='plasma',
            hover_data=['formula','x','band_gap','stability','score'], height=450
        )
        fig.update_traces(marker=dict(size=18, line_width=1), opacity=0.9)
        fig.add_trace(
            go.Scatter(
                x=df.loc[df.is_top, 'stability'],
                y=df.loc[df.is_top, 'band_gap'],
                mode='markers',
                marker=dict(size=22, color='rgba(0,0,0,0)', line=dict(width=2, color='black')),
                hoverinfo='skip', showlegend=False
            )
        )
        fig.update_xaxes(title='<b>Stability</b>', range=[0.75,1.0], dtick=0.05)
        fig.update_yaxes(title='<b>Band-gap (eV)</b>', range=[0,3.5], dtick=0.5)
        fig.update_coloraxes(colorbar_title="<b>Score</b>")
        fig.update_layout(template='simple_white', margin=dict(l=70,r=40,t=25,b=65))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.caption("ℹ️ Hover points; scroll to zoom; drag to rotate")
        fig3d = px.scatter_3d(
            df, x="x", y="y", z="score",
            color="score",
            hover_data=["formula","band_gap","stability"],
            height=600
        )
        fig3d.update_layout(
            template="simple_white",
            scene=dict(
                xaxis_title="B fraction (x)",
                yaxis_title="C fraction (y)",
                zaxis_title="Score"
            ),
            coloraxis_colorbar=dict(title="<b>Score</b>")
        )
        st.plotly_chart(fig3d, use_container_width=True)

# ... the rest of your tabs (Download, Benchmark, Results Summary) remain unchanged,
# since they already reference `band_gap` and no longer `Eg`.

