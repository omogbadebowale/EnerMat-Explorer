"""
EnerMat Perovskite Explorer – Streamlit front‑end
Clean build • 2025‑07‑13 🟢   (includes Sn‑oxidation column + plotting)

Usage:
$ streamlit run frontend/app.py
"""

from __future__ import annotations
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# ---- local backend ----
try:
    from perovskite_utils import mix_abx3, screen_ternary
except ImportError as exc:
    st.error("❌ Could not import backend module 'perovskite_utils'. "
             "Make sure it is on PYTHONPATH!\n" + str(exc))
    st.stop()

# ---- env / API key guard ----
load_dotenv()
if not os.getenv("MAPI_KEY"):
    st.warning("⚠️  Materials Project API key (MAPI_KEY) not found in environment. "
               "Backend calls will fail. Add it to a .env file or shell.")

st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🧪 EnerMat Perovskite Explorer")

st.markdown(
    "Use this tool to **screen lead‑free perovskite compositions** for band‑gap, "
    "thermodynamic stability (E<sub>hull</sub>) *and* Sn oxidation resistance "
    "(ΔE<sub>ox</sub>). Adjust the sliders, hit *Run*, and inspect the ranked list.",
    unsafe_allow_html=True,
)

# ---- user inputs ----
with st.sidebar:
    st.header("🔧 Screening parameters")

    form = st.form("params")
    col1, col2 = form.columns(2)
    formula_A = col1.text_input("Compound A (Sn²⁺)", "CsSnI3")
    formula_B = col2.text_input("Compound B (Sn²⁺)", "CsSnBr3")

    x_step = form.number_input("x‑increment (composition grid)", 0.0, 1.0, 0.25, 0.05)
    bg_min, bg_max = form.slider("Target band‑gap window / eV", 0.8, 2.5, (1.1, 1.6), 0.05)

    rh = form.number_input("Relative humidity %", 0, 100, 30)
    temp = form.number_input("Temperature K", 250, 400, 300)
    ternary = form.checkbox("Run ternary sweep (slow)", False)

    submitted = form.form_submit_button("▶ Run screen")

# ---- run backend ----
@st.cache_data(show_spinner=True)
def _run_screen(fA: str, fB: str, step: float, bg: tuple[float, float], rh: int, temp: int, ternary: bool):
    if ternary:
        df = screen_ternary(fA, fB, bg_window=bg, rh=rh, temp=temp, dx=step)
    else:
        df = mix_abx3(fA, fB, bg_window=bg, rh=rh, temp=temp, dx=step)
    return df

if submitted:
    try:
        df = _run_screen(formula_A.strip(), formula_B.strip(), x_step, (bg_min, bg_max), rh, temp, ternary)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    if df.empty:
        st.info("No compositions met the criteria. Try relaxing the band‑gap or stability window.")
        st.stop()

    # ---- show table ----
    show_cols = ["formula", "x", "Eg", "Ehull", "Eox", "score"]
    st.subheader("📄 Results table (sorted by score)")
    st.dataframe(df[show_cols].sort_values("score", ascending=False), height=400)

    # ---- scatter plot ΔEox vs Eg ----
    st.subheader("📈 ΔEox vs Eg plot")
    fig, ax = plt.subplots()
    sc = ax.scatter(df["Eg"], df["Eox"], s=df["score"]*100, alpha=0.7)
    ax.set_xlabel("Band‑gap Eg / eV")
    ax.set_ylabel("ΔEox / eV   (Sn²⁺→Sn⁴⁺)")
    ax.axhline(0.0, linestyle="--")
    st.pyplot(fig)

    # ---- download ----
    csv = df.to_csv(index=False).encode()
    st.download_button("💾 Download CSV", csv, f"screen_{formula_A}_{formula_B}.csv", "text/csv")

else:
    st.info("Configure parameters and press **Run screen**.")
