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
    st.error("🛑 Please set MP_API_KEY to your 32-character Materials Project API key")
    st.stop()
mpr = MPRester(API_KEY)

# ── Supported end-members ──────────────────────────────────────────────
END_MEMBERS = ["CsPbBr3", "CsSnBr3", "CsSnCl3", "CsPbI3"]

# ── Ionic radii (Å) for Goldschmidt tolerance ──────────────────────────
IONIC_RADII = {
    "Cs": 1.88, "Rb": 1.72, "MA": 2.17, "FA": 2.53,
    "Pb": 1.19, "Sn": 1.18, "I": 2.20, "Br": 1.96, "Cl": 1.81
}

# ── Backend functions (reuse existing) ─────────────────────────────────
from backend.perovskite_utils import (
    mix_abx3 as screen_binary,
    screen_ternary as screen_tern,
    END_MEMBERS,
    fetch_mp_data as _summary
)

# ───────────────────────────────────── App Config ─────────────────────────────
st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.6")

# Sidebar: mode and inputs
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose screening type", ["Binary A–B", "Ternary A–B–C"])
    st.markdown("---")
    st.header("End-members")
    if mode == "Binary A–B":
        A = st.selectbox("Preset A", END_MEMBERS, index=0)
        B = st.selectbox("Preset B", END_MEMBERS, index=1)
        st.text_input("Custom A (optional)", key="customA")
        st.text_input("Custom B (optional)", key="customB")
        if st.session_state.customA:
            A = st.session_state.customA.strip()
        if st.session_state.customB:
            B = st.session_state.customB.strip()
    else:
        A = st.selectbox("Preset A", END_MEMBERS, index=0)
        B = st.selectbox("Preset B", END_MEMBERS, index=1)
        C = st.selectbox("Preset C", END_MEMBERS, index=2)
        st.text_input("Custom A (optional)", key="customA")
        st.text_input("Custom B (optional)", key="customB")
        st.text_input("Custom C (optional)", key="customC")
        if st.session_state.customA: A = st.session_state.customA.strip()
        if st.session_state.customB: B = st.session_state.customB.strip()
        if st.session_state.customC: C = st.session_state.customC.strip()

    st.markdown("---")
    st.header("Environment")
    rh = st.slider("Humidity [%]", 0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)

    st.markdown("---")
    st.header("Model knobs")
    bg_lo, bg_hi = st.slider("Target gap [eV]", 0.5, 3.0, (1.0, 1.4), 0.01)
    bow = st.number_input("Bowing [eV]", 0.0, 1.0, 0.30, 0.05)
    dx = st.number_input("x-step", 0.01, 0.50, 0.05, 0.01)
    if mode == "Ternary A–B–C":
        dy = st.number_input("y-step", 0.01, 0.50, 0.05, 0.01)
        bows = {
            'AB': st.number_input("Bow AB [eV]", 0.0, 1.0, 0.30, 0.05),
            'AC': st.number_input("Bow AC [eV]", 0.0, 1.0, 0.30, 0.05),
            'BC': st.number_input("Bow BC [eV]", 0.0, 1.0, 0.30, 0.05)
        }
    else:
        dy = None
        bows = None

    st.markdown("---")
    if st.button("🗑 Clear history"):
        st.session_state.clear()
    st.caption("© 2025 Dr Gbadebo Taofeek Yusuf")

# run logic
if 'history' not in st.session_state:
    st.session_state.history = []

def run_screen(**kw):
    if mode == "Binary A–B":
        return screen_binary(**kw)
    else:
        return screen_tern(**kw)

col1, col2 = st.columns([3,1])
do_run = col1.button("▶ Run screening")
if do_run:
    try:
        docA = _summary(A, ['band_gap','energy_above_hull'])
        docB = _summary(B, ['band_gap','energy_above_hull'])
        if mode == "Ternary A–B–C": docC = _summary(C, ['band_gap','energy_above_hull'])
    except Exception as e:
        st.error(f"❌ Error querying MP: {e}")
        st.stop()
    df = run_screen(A=A, B=B, C=C if mode!='Binary A–B' else None, rh=rh, temp=temp,
                    bg=(bg_lo,bg_hi), bow=bow, dx=dx, dy=dy, bows=bows)
    st.session_state.history.append(df)
elif st.session_state.history:
    df = st.session_state.history[-1]
else:
    st.info("Press ▶ to run screening.")
    st.stop()

# display results
st.header("📊 Table")
st.dataframe(df)
