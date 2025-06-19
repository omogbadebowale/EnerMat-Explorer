import io
import uuid
import datetime
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import nbformat as nbf
from docx import Document
from backend.perovskite_utils import screen, END_MEMBERS, _summary
import plotly.io as pio

# ───────────────────── App Config ─────────────────────
st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.6")

# ───────────────────── Sidebar ─────────────────────
with st.sidebar:
    st.header("Environment")
    RH = st.slider("Humidity [%]", 0, 100, 50)
    T = st.slider("Temperature [°C]", -20, 100, 25)
    Eg_min, Eg_max = st.slider("Target gap [eV]", 0.5, 3.0, (0.5, 2.59))

    st.header("Parent formulas")
    preset_A = st.selectbox("Preset A", END_MEMBERS, index=0)
    preset_B = st.selectbox("Preset B", END_MEMBERS, index=1)
    custom_A = st.text_input("Custom A (optional)", value="").strip()
    custom_B = st.text_input("Custom B (optional)", value="").strip()
    A = custom_A if custom_A else preset_A
    B = custom_B if custom_B else preset_B

    st.header("Model knobs")
    bow = st.number_input("Bowing [eV]", 0.0, 1.0, 0.35, 0.05)
    dx = st.number_input("x-step", 0.01, 0.5, 0.05, 0.01)

    if st.button("🗑 Clear history"):
        if "history" in st.session_state:
            del st.session_state["history"]
        st.rerun()

    st.caption("© 2025 Dr Gbadebo Taofeek Yusuf")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    st.caption(f"🧪 Session started: {ts}")

# ───────────────────── Run Button ─────────────────────
if st.button("▶️ Run screening"):
    try:
        dA = _summary(A)
        dB = _summary(B)
    except Exception as e:
        st.error(f"❌ Error loading material data: {e}")
        st.stop()

    results_df = screen(A, B, bow, dx, RH, T, Eg_min, Eg_max)
    st.session_state["results"] = results_df
    st.session_state["params"] = dict(A=A, B=B, bow=bow, dx=dx, RH=RH, T=T, Eg_min=Eg_min, Eg_max=Eg_max)
    st.success("✅ Screening completed successfully!")

# ───────────────────── Results Display ────────────────────
if "results" in st.session_state:
    st.subheader("📊 Results Summary")
    df = st.session_state["results"]
    st.dataframe(df)

    # Plot score vs. composition
    fig = px.scatter(df, x="x", y="score", color="formula", title="Composite Score vs. Composition")
    st.plotly_chart(fig, use_container_width=True)

    # OPTIONAL: Remove or comment this block if Kaleido is not supported
    # try:
    #     png = pio.to_image(fig, format='png', scale=2)
    #     st.download_button("📥 Download Plot as PNG", data=png, file_name="perovskite_plot.png", mime="image/png")
    # except Exception as e:
    #     st.warning("⚠️ PNG export unavailable – Kaleido or Chromium not found.")
    #     st.text(f"Details: {e}")
