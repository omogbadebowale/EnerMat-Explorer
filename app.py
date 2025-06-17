# app.py  –  EnerMat Perovskite Explorer v9.6 with Publication-Ready Plots & Local-Benchmark
# Author: Dr Gbadebo Taofeek Yusuf

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

from backend.perovskite_utils import screen, END_MEMBERS, _summary

# ───────────────────────────────────── App config ────────────────────────────
st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.6")

# ───────────────────────────────── Session History ──────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []

# ───────────────────────────────────── Sidebar ───────────────────────────────
with st.sidebar:
    st.header("Environment")
    rh   = st.slider("Humidity [%]", 0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)
    bg_lo, bg_hi = st.slider("Target gap [eV]", 0.5, 3.0, (1.0, 1.4), 0.01)

    st.header("Parent formulas")
    A_pick = st.selectbox("Preset A", END_MEMBERS, 0)
    B_pick = st.selectbox("Preset B", END_MEMBERS, 1)
    A = st.text_input("Custom A (optional)", "").strip() or A_pick
    B = st.text_input("Custom B (optional)", "").strip() or B_pick

    st.header("Model knobs")
    bow = st.number_input("Bowing [eV]", 0.0, 1.0, 0.30, 0.05)
    dx  = st.number_input("x-step",   0.01, 0.50, 0.05, 0.01)

    if st.button("🗑 Clear history"):
        st.session_state["history"].clear()
        st.experimental_rerun()

    st.caption("© 2025 Dr Gbadebo Taofeek Yusuf")
    GIT_SHA = st.secrets.get("GIT_SHA", "dev")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    st.caption(f"⚙️ Version: `{GIT_SHA}` • ⏱ {ts}")

    st.markdown("### How to explore")
    st.markdown(
        "1. ▶️ Run screening → open **Plot** tab  \n"
        "2. 🔍 Hover for formula & scores  \n"
        "3. 🖱️ Scroll/drag to zoom & pan  \n"
        "4. 📊 Sort **Table** by header click  \n"
        "5. ⬇ Download results"
    )
    with st.expander("🔎 About this tool", expanded=False):
        st.image("https://your-cdn.com/images/logo.png", width=100)
        st.markdown(
            "This app screens perovskite alloys for band-gap and stability "
            "using Materials Project data and Monte Carlo sampling."
        )

# ─────────────────────────────────── Backend call ────────────────────────────
@st.cache_data(show_spinner="Monte-Carlo sampling …")
def run_screen(**kw):
    return screen(**kw)

# ─────────────────────────────── Run / Back logic ──────────────────────────
col_run, col_back = st.columns([3, 1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=len(st.session_state["history"]) < 1)

if do_back and st.session_state["history"]:
    st.session_state["history"].pop()
    A, B, rh, temp, df = st.session_state["history"][-1]
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
    st.session_state["history"].append((A, B, rh, temp, df))
elif st.session_state["history"]:
    A, B, rh, temp, df = st.session_state["history"][-1]
else:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

# ───────────────────────────────────── Tabs ─────────────────────────────────
tab_tbl, tab_plot, tab_dl, tab_bench = st.tabs(
    ["📊 Table", "📈 Plot", "⬇ Download", "⚖ Benchmark"]
)

# (Table, Plot, Download tabs unchanged – see previous code)

# ─────────── Benchmark Tab (Local or Upload) ───────────
with tab_bench:
    st.markdown("## ⚖ Benchmark: DFT vs. Experimental Gaps")

    # check for local CSV
    local_path = Path(__file__).parent / "exp_bandgaps.csv"
    if local_path.exists():
        exp_df = pd.read_csv(local_path)
        st.success("Loaded experimental data from local file `exp_bandgaps.csv`")
    else:
        uploaded = st.file_uploader(
            "Upload experimental band-gap CSV (`formula`,`exp_gap`)",
            type="csv", help="Columns: formula, exp_gap"
        )
        if not uploaded:
            st.info("Please upload `exp_bandgaps.csv` to benchmark DFT vs. experiment.")
            st.stop()
        exp_df = pd.read_csv(uploaded)

    # validate
    if not {"formula", "exp_gap"}.issubset(exp_df.columns):
        st.error("Your CSV must contain columns `formula` and `exp_gap`.")
        st.stop()

    # fetch DFT
    load_dotenv()
    mpr = MPRester(os.getenv("MP_API_KEY", ""))
    bench = []
    for f in END_MEMBERS:
        entry = next(mpr.summary.search(formula=f, fields=["band_gap"]), None)
        if entry:
            bench.append({"formula": f, "dft_gap": entry.band_gap})
    dft_df = pd.DataFrame(bench)

    # merge & calc error
    merged = (
        dft_df
        .merge(exp_df.rename(columns={"formula":"formula","exp_gap":"exp_gap"}),
               on="formula", how="inner")
        .assign(error=lambda d: d["dft_gap"] - d["exp_gap"])
    )
    if merged.empty:
        st.error("No matching formulas between DFT and experimental data.")
        st.stop()

    # metrics
    mae  = merged["error"].abs().mean()
    rmse = np.sqrt((merged["error"]**2).mean())
    st.write(f"**MAE:** {mae:.3f} eV **RMSE:** {rmse:.3f} eV")

    # parity plot (pub-ready)
    fig1 = px.scatter(
        merged, x="exp_gap", y="dft_gap", text="formula",
        labels={"exp_gap":"Exp Eg (eV)","dft_gap":"DFT Eg (eV)"},
        title="Parity Plot: DFT vs. Experimental"
    )
    mn = merged[["exp_gap","dft_gap"]].min().min()
    mx = merged[["exp_gap","dft_gap"]].max().max()
    fig1.add_shape(
        type="line", x0=mn, y0=mn, x1=mx, y1=mx,
        line=dict(dash="dash", color="black", width=1.5)
    )
    fig1.update_layout(
        template="simple_white",
        font=dict(family="Times New Roman", size=16),
        xaxis=dict(
            title=dict(text="<b>Experimental Eg (eV)</b>", font=dict(size=18)),
            ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True
        ),
        yaxis=dict(
            title=dict(text="<b>DFT Eg (eV)</b>", font=dict(size=18)),
            ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True
        ),
        margin=dict(l=80, r=40, t=50, b=80)
    )
    st.plotly_chart(fig1, use_container_width=True)

    png1 = fig1.to_image(format="png", scale=3)
    st.download_button(
        "📥 Download Parity Plot (PNG)",
        png1, "parity_plot.png", "image/png"
    )

    # error histogram (pub-ready)
    fig2 = px.histogram(
        merged, x="error", nbins=12,
        labels={"error":"Δ Eg (eV)"},
        title="Error Distribution (DFT – Exp)"
    )
    fig2.update_layout(
        template="simple_white",
        font=dict(family="Times New Roman", size=16),
        xaxis=dict(
            title=dict(text="<b>Δ Eg (eV)</b>", font=dict(size=18)),
            ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True
        ),
        yaxis=dict(
            title=dict(text="<b>Count</b>", font=dict(size=18)),
            ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True
        ),
        margin=dict(l=80, r=40, t=50, b=80)
    )
    st.plotly_chart(fig2, use_container_width=True)

    png2 = fig2.to_image(format="png", scale=3)
    st.download_button(
        "📥 Download Error Histogram (PNG)",
        png2, "error_histogram.png", "image/png"
    )
