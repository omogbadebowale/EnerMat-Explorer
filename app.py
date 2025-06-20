```python
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

# load env
load_dotenv()
# fallback for Streamlit Cloud
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    st.error("🛑 Please set MP_API_KEY to your 32‑character Materials Project API key")
    st.stop()

# backend imports
from backend.perovskite_utils import (
    mix_abx3 as screen_binary,
    screen_ternary as screen_ternary,
    END_MEMBERS,
    fetch_mp_data as _summary
)

# app config
st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.6")

# session history
def _init_history():
    if "history" not in st.session_state:
        st.session_state.history = []
_init_history()

# sidebar UI
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose screening type", ["Binary A–B", "Ternary A–B–C"])

    st.markdown("---")
    st.header("End‑members")
    A = st.selectbox("Preset A", END_MEMBERS, index=0)
    B = st.selectbox("Preset B", END_MEMBERS, index=1)
    customA = st.text_input("Custom A (optional)", key="customA")
    customB = st.text_input("Custom B (optional)", key="customB")
    if customA:
        A = customA.strip()
    if customB:
        B = customB.strip()
    if mode == "Ternary A–B–C":
        C = st.selectbox("Preset C", END_MEMBERS, index=2)
        customC = st.text_input("Custom C (optional)", key="customC")
        if customC:
            C = customC.strip()

    st.markdown("---")
    st.header("Environment")
    rh = st.slider("Humidity [%]", 0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)

    st.markdown("---")
    st.header("Model knobs")
    bg_lo, bg_hi = st.slider("Target gap [eV]", 0.5, 3.0, (1.0, 1.4), 0.01)
    bow = st.number_input("Bowing [eV]", 0.0, 1.0, 0.30, 0.05)
    dx = st.number_input("x-step", 0.01, 0.5, 0.05, 0.01)
    dy = None
    bows = None
    if mode == "Ternary A–B–C":
        dy = st.number_input("y-step", 0.01, 0.5, 0.05, 0.01)
        bows = {
            'AB': st.number_input("Bow AB [eV]", 0.0, 1.0, 0.30, 0.05),
            'AC': st.number_input("Bow AC [eV]", 0.0, 1.0, 0.30, 0.05),
            'BC': st.number_input("Bow BC [eV]", 0.0, 1.0, 0.30, 0.05),
        }
    if st.button("🗑 Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()
    st.caption("© 2025 Dr Gbadebo Taofeek Yusuf")

# run/prev buttons
col_run, col_prev = st.columns([3,1])
do_run = col_run.button("▶ Run screening", type="primary")
do_prev = col_prev.button("⏪ Previous", disabled=len(st.session_state.history)<1)

if do_prev and st.session_state.history:
    st.session_state.history.pop()
    df = st.session_state.history[-1]
elif do_run:
    try:
        _ = _summary(A, ["band_gap","energy_above_hull"]);
        _ = _summary(B, ["band_gap","energy_above_hull"])
        if mode == "Ternary A–B–C":
            _ = _summary(C, ["band_gap","energy_above_hull"])
    except Exception as e:
        st.error(f"❌ Error querying Materials Project: {e}")
        st.stop()

    if mode == "Binary A–B":
        df = screen_binary(A=A, B=B, rh=rh, temp=temp, bg=(bg_lo,bg_hi), bowing=bow, dx=dx)
    else:
        df = screen_ternary(A=A, B=B, C=C, rh=rh, temp=temp, bg=(bg_lo,bg_hi), bows=bows, dx=dx, dy=dy)

    if df.empty:
        st.error("No candidates found – try expanding your ranges.")
        st.stop()
    st.session_state.history.append(df)
else:
    if st.session_state.history:
        df = st.session_state.history[-1]
    else:
        st.info("Press ▶ Run screening to begin.")
        st.stop()

# display table
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Table","📈 Plot","📥 Download","⚖ Benchmark","📑 Results Summary"])
with tab1:
    st.markdown("**Run parameters**")
    st.table(pd.DataFrame({
        "Parameter":["Humidity [%]","Temperature [°C]","Gap [eV]","Bowing [eV]","x-step"] + (["y-step"] if dy else []),
        "Value":[rh,temp,f"{bg_lo:.2f}–{bg_hi:.2f}",bow,dx] + ([dy] if dy else [])
    }))
    st.dataframe(df, height=350)
```
