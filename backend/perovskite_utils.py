"""
EnerMat Perovskite Explorer – Streamlit Front‑End
Clean build • 2025‑07‑13 🟢

Run locally with:
    streamlit run frontend/app.py

The UI lets you:
* pick binary or ternary mixes from a pre‑set list or free text,
* set relative humidity, temperature, band‑gap window,
* view a sortable table including ΔEox and overall score,
* plot Eg vs ΔEox (bubble size ∝ score),
* download the DataFrame as CSV.
"""
from __future__ import annotations

import io
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ─── backend helpers -----------------------------------------------------------
from backend.perovskite_utils import (
    mix_abx3 as screen_binary,
    screen_ternary,
    END_MEMBERS,
    fetch_mp_data,  # legacy import still available in backend
)

st.set_page_config(page_title="EnerMat Explorer", layout="wide")
st.title("⚛️ EnerMat Perovskite Explorer")

# ─── sidebar – global parameters ---------------------------------------------
st.sidebar.header("Global parameters")
bg_lo = st.sidebar.number_input("Min Eg (eV)", 0.8, 3.5, 1.1, 0.05)
bg_hi = st.sidebar.number_input("Max Eg (eV)", 0.8, 3.5, 1.6, 0.05)
rh    = st.sidebar.slider("Relative humidity (%)", 0, 100, 30)
Temp  = st.sidebar.slider("Ambient T (°C)", 0, 100, 25)

mode = st.sidebar.radio("Screening mode", ["Binary", "Ternary"], index=0)

# ─── input helpers ------------------------------------------------------------
def _pick_formula(label: str) -> str:
    preset = st.selectbox(label + " – presets", END_MEMBERS, key=label)
    custom = st.text_input(label + " – custom", key=label + "_custom")
    return custom.strip() or preset

# ─── run screening ------------------------------------------------------------

if mode == "Binary":
    st.header("Binary ABX₃ → (1−x)A + xB")

    col1, col2 = st.columns(2)
    with col1:
        fA = _pick_formula("Formula A")
    with col2:
        fB = _pick_formula("Formula B")

    run = st.button("Run binary screen →")
    if run:
        df = screen_binary(
            fA, fB,
            rh=rh, temp=Temp,
            bg_window=(bg_lo, bg_hi),
        )
        if df.empty:
            st.error("No data – check formulas or MP API quota.")
        else:
            st.success(f"{len(df)} compositions evaluated")
            st.dataframe(df)

            # plot Eg vs Eox
            fig, ax = plt.subplots()
            sc = ax.scatter(df["Eg"], df["Eox"], s=df["score"]*200,
                            alpha=0.6)
            ax.set_xlabel("Band gap Eg (eV)")
            ax.set_ylabel("ΔEox (eV)")
            ax.set_title("Optical gap vs Sn oxidation driving‑force")
            st.pyplot(fig)

            # download button
            csv = df.to_csv(index=False).encode()
            st.download_button("Download CSV", csv, file_name="binary_screen.csv")

else:  # ternary mode
    st.header("Ternary ABX₃ → zA + xB + yC  with z=1−x−y")

    colA, colB, colC = st.columns(3)
    with colA:
        fA = _pick_formula("Formula A")
    with colB:
        fB = _pick_formula("Formula B")
    with colC:
        fC = _pick_formula("Formula C")

    dx = st.sidebar.slider("Δx step", 0.05, 0.25, 0.10, 0.05)

    run = st.button("Run ternary screen →")
    if run:
        df = screen_ternary(
            fA, fB, fC,
            rh=rh, temp=Temp,
            bg=(bg_lo, bg_hi),
            dx=dx, dy=dx,
        )
        if df.empty:
            st.error("No data – check formulas or MP API quota.")
        else:
            st.success(f"{len(df)} compositions evaluated")
            st.dataframe(df)

            fig, ax = plt.subplots()
            sc = ax.scatter(df["Eg"], df["Eox"], s=df["score"]*120,
                            alpha=0.6)
            ax.set_xlabel("Band gap Eg (eV)")
            ax.set_ylabel("ΔEox (eV)")
            ax.set_title("Optical gap vs Sn oxidation driving‑force")
            st.pyplot(fig)

            csv = df.to_csv(index=False).encode()
            st.download_button("Download CSV", csv, file_name="ternary_screen.csv")
