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
# from docx import Document  # Commented out due to missing module error
from mp_api.client import MPRester

from backend.perovskite_utils import screen, END_MEMBERS, _summary

#───────────────────────────────────── App Config ─────────────────────────────
st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.6")

#────────────────────────── Session History ───────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

#─────────────────────────────────── Sidebar ───────────────────────────────────
with st.sidebar:
    st.header("Environment")
    rh = st.slider("Humidity [%]", 0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)
    bg_lo, bg_hi = st.slider("Target gap [eV]", 0.5, 3.0, (1.0, 1.4), 0.01)

    st.header("Parent formulas")
    A_pick = st.selectbox("Preset A", END_MEMBERS, index=0)
    B_pick = st.selectbox("Preset B", END_MEMBERS, index=1)
    A = st.text_input("Custom A (optional)", "").strip() or A_pick
    B = st.text_input("Custom B (optional)", "").strip() or B_pick

    st.header("Model knobs")
    bow = st.number_input("Bowing [eV]", 0.0, 1.0, 0.30, 0.05)
    dx = st.number_input("x-step", 0.01, 0.50, 0.05, 0.01)

    if st.button("🗑 Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()

    st.caption("© 2025 Dr Gbadebo Taofeek Yusuf")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    st.caption(f"⏱ Timestamp: {ts}")

#───────────────────────────────── Backend Call ───────────────────────────────
@st.cache_data(show_spinner="Monte-Carlo sampling …")
def run_screen(**kw):
    return screen(**kw)

#──────────────────────────── Run / Back Logic ────────────────────────────────
col_run, col_back = st.columns([3,1])
do_run = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=len(st.session_state.history)<1)

if do_back and st.session_state.history:
    st.session_state.history.pop()
    A, B, rh, temp, df = st.session_state.history[-1]
    st.success("Showing previous result")
elif do_run:
    dA, dB = _summary(A), _summary(B)
    if not dA or not dB:
        st.error("Failed to fetch Materials Project data for endmembers.")
        st.stop()
    df = run_screen(A=A, B=B, rh=rh, temp=temp, bg=(bg_lo, bg_hi), bow=bow, dx=dx)
    if df.empty:
        st.error("No candidates found – try widening your window.")
        st.stop()
    st.session_state.history.append((A, B, rh, temp, df))
elif st.session_state.history:
    A, B, rh, temp, df = st.session_state.history[-1]
else:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

#─────────────────────────────────── Table ─────────────────────────────────────
st.subheader("📊 Screening Table")
params = pd.DataFrame({
    "Parameter": ["Humidity [%]", "Temperature [°C]", "Gap window [eV]", "Bowing [eV]", "x-step"],
    "Value": [rh, temp, f"{bg_lo:.2f}–{bg_hi:.2f}", bow, dx]
})
st.table(params)

st.dataframe(df, height=400, use_container_width=True)

#─────────────────────────────────── Plot ──────────────────────────────────────
st.subheader("📈 Stability vs. Band-Gap")
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
fig.update_layout(template='simple_white', margin=dict(l=70, r=40, t=25, b=65))
st.plotly_chart(fig, use_container_width=True)
