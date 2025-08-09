import datetime
import io
from pathlib import Path
import base64

import streamlit as st
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from docx import Document
from backend.perovskite_utils import (
    screen_binary,
    screen_ternary,
    END_MEMBERS,
)
# ─────────── STREAMLIT PAGE CONFIG ───────────
st.set_page_config("EnerMat Explorer – Lead-Free Perovskite PV Discovery Tool", layout="wide")
st.title("☀️ EnerMat Explorer | Lead-Free Perovskite PV Discovery Tool")
st.markdown(
    """
    <style>
      /* Target the sidebar wrapper and give it a colored left border */
      .css-1d391kg {
        border-right: 3px solid #0D47A1 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
# ─────────── SESSION STATE ───────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────── SIDEBAR ───────────
with st.sidebar:
    st.header("Mode")
    mode = st.radio(
        "Choose screening type",
        ["Binary A–B", "Ternary A–B–C"]
    )

    st.header("End-members")
    preset_A = st.selectbox("Preset A", END_MEMBERS, 0)
    preset_B = st.selectbox("Preset B", END_MEMBERS, 1)
    custom_A = st.text_input("Custom A (optional)").strip()
    custom_B = st.text_input("Custom B (optional)").strip()
    A = custom_A or preset_A
    B = custom_B or preset_B
    if mode.startswith("Ternary"):
        preset_C = st.selectbox("Preset C", END_MEMBERS, 2)
        custom_C = st.text_input("Custom C (optional)").strip()
        C = custom_C or preset_C

    st.header("Application")
    application = st.selectbox(
        "Select application",
        ["single", "tandem", "indoor", "detector"]
    )

    # ── Environment sliders removed ──

    st.header("Target band-gap [eV]")
    bg_lo, bg_hi = st.slider(
        "Gap window", 0.50, 3.00, (1.00, 1.40), 0.01
    )

    st.header("Model settings")
    bow = st.number_input(
        "Bowing (eV, negative ⇒ gap↑)",
        -1.0, 1.0, -0.15, 0.05
    )
    dx = st.number_input("x-step", 0.01, 0.50, 0.05, 0.01)
    if mode.startswith("Ternary"):
        dy = st.number_input("y-step", 0.01, 0.50, 0.05, 0.01)

    z = st.slider(
        "Ge fraction z", 0.00, 0.80, 0.10, 0.05,
        help="B-site Ge²⁺ in CsSn₁₋zGeₓX₃"
    )

   # ── Clear history button ──
    if st.button("🗑 Clear history"):
        # Safely clear
        if "history" in st.session_state:
            st.session_state.history = []
        # Re-run with clean state
        st.rerun()

    # ── Developer credit in sidebar footer ──
    st.markdown(
    """
    <div style="font-size:0.85rem; color:#555; margin-top:0.5rem;">
      <strong>Developer:</strong> Dr Gbadebo Taofeek Yusuf (Academic World)  
      📞 +44 7776 727237  ✉️ das@academicworld.co.uk
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────── CACHE WRAPPERS ───────────
@st.cache_data(show_spinner="⏳ Screening …", max_entries=20)
def _run_binary(*args, **kwargs):
    return screen_binary(*args, **kwargs)

@st.cache_data(show_spinner="⏳ Screening …", max_entries=10)
def _run_ternary(*args, **kwargs):
    return screen_ternary(*args, **kwargs)

# ─────────── OVERVIEW & RESEARCH OPPORTUNITIES ───────────
st.markdown(
    """
    <style>
      .overview-box {
        background-color: #ffffff;       /* white for max contrast */
        border: 1px solid #dddddd;       /* light grey border */
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 32px;
        color: #333333;
        font-family: Arial, sans-serif;
      }
      .overview-box h2 {
        margin-top: 0;
        color: #005FAD;                  /* deep brand-blue */
        font-size: 1.8rem;
      }
      .overview-box p {
        font-size: 1rem;
        line-height: 1.5;
        margin-bottom: 16px;
      }
      .overview-box ul {
        margin: 0 0 16px 1.2em;
        font-size: 0.95rem;
        line-height: 1.4;
      }
      .overview-box ul li {
        margin-bottom: 8px;
      }
    </style>

    <div class="overview-box">
      <h2>Context &amp; Scientific Justification</h2>
      <p>
        Lead–halide perovskites deliver record solar efficiencies but suffer from environmental toxicity and rapid degradation under heat, moisture, or oxygen.
        Tin-based, lead-free analogues offer a safer path, yet optimising their key metrics remains a major hurdle:
      </p>
      <ul>
        <li><strong>Eg</strong> (band gap): ideal ≈ 1.3 eV for single-junction PV absorption.</li>
        <li><strong>E<sub>hull</sub></strong> (phase stability): &lt; 0.05 eV / atom ⇒ likely synthesizable.</li>
        <li><strong>ΔE<sub>ox</sub></strong> (oxidation resistance): positive values resist Sn²⁺ → Sn⁴⁺.</li>
        <li><strong>PCE<sub>max</sub></strong> (Shockley–Queisser limit): theoretical upper bound on efficiency.</li>
      </ul>
      <p>
        <em>EnerMa
