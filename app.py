# app.py  –  EnerMat Perovskite Explorer v9.6  
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
    dx  = st.number_input("x-step", 0.01, 0.50, 0.05, 0.01)

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

# ─────────── Table Tab ───────────
with tab_tbl:
    params = pd.DataFrame({
        "Parameter": ["Humidity [%]", "Temperature [°C]", "Gap window [eV]", "Bowing [eV]", "x-step"],
        "Value":     [rh, temp, f"{bg_lo:.2f}–{bg_hi:.2f}", bow, dx]
    })
    st.markdown("**Run parameters**")
    st.table(params)

    docA, docB = _summary(A), _summary(B)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**A-endmember: {A}**")
        st.write(f"MP band gap: {docA.band_gap:.2f} eV")
        st.write(f"MP E_above_hull: {docA.energy_above_hull:.3f} eV/atom")
    with c2:
        st.markdown(f"**B-endmember: {B}**")
        st.write(f"MP band gap: {docB.band_gap:.2f} eV")
        st.write(f"MP E_above_hull: {docB.energy_above_hull:.3f} eV/atom")

    st.dataframe(df, height=400, use_container_width=True)

# ─────────── Plot Tab ───────────
with tab_plot:
    st.caption("ℹ️ **Tip**: Hover circles; scroll to zoom; drag to pan")
    top_cut = df["score"].quantile(0.80)
    df["is_top"] = df["score"] >= top_cut

    fig = px.scatter(
        df, x="stability", y="band_gap",
        color="score", color_continuous_scale="plasma",
        hover_data=["formula","x","band_gap","stability","score"]
    )
    fig.update_traces(
        marker=dict(size=20, line=dict(width=1.5, color="black")),
        selector=dict(mode="markers")
    )
    outline = go.Scatter(
        x=df.loc[df.is_top, "stability"],
        y=df.loc[df.is_top, "band_gap"],
        mode="markers", hoverinfo="skip",
        marker=dict(size=24, color="rgba(0,0,0,0)", line=dict(width=2, color="black")),
        showlegend=False
    )
    fig.add_trace(outline)

    fig.update_layout(
        template="simple_white",
        font=dict(family="Times New Roman", size=16, color="black"),
        xaxis=dict(
            title=dict(text="<b>Thermodynamic Stability</b>", font=dict(size=18)),
            ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True,
            range=[0.75,1.00], dtick=0.05
        ),
        yaxis=dict(
            title=dict(text="<b>Band-gap (eV)</b>", font=dict(size=18)),
            ticks="outside", showline=True, linecolor="black", linewidth=2, mirror=True,
            range=[0,3.5], dtick=0.5
        ),
        coloraxis_colorbar=dict(
            title=dict(text="<b>Composite Score</b>", font=dict(size=16)),
            tickfont=dict(size=14), thickness=15, lenmode="fraction", len=0.5
        ),
        margin=dict(l=80, r=40, t=50, b=80)
    )
    st.plotly_chart(fig, use_container_width=True)

    png = fig.to_image(format="png", scale=3)
    st.download_button(
        "📥 Download stability vs gap plot (PNG)",
        png, "stability_vs_gap.png", "image/png",
        use_container_width=True
    )

# ─────────── Download Tab ───────────
with tab_bench:
    st.markdown("## ⚖ Benchmark: DFT vs. Experimental Gaps")

    # 1) Allow user to upload experimental CSV if they like
    uploaded = st.file_uploader(
        "Upload experimental band-gap CSV (`formula`, `exp_gap`)", type="csv"
    )

    # 2) Load experimental data
    if uploaded:
        exp_df = pd.read_csv(uploaded)
        st.success("Loaded experimental data from uploaded file")
    else:
        exp_df = pd.read_csv("exp_bandgaps.csv")
        st.success("Loaded experimental data from bundled CSV")

    # 3) Load DFT/reference data from your bundled file
    dft_df = pd.read_csv("pbe_bandgaps.csv")
    st.info(f"Loaded {len(dft_df)} DFT band gaps from bundled CSV")

    # 4) Sanity‐check columns
    if not {"formula","exp_gap"}.issubset(exp_df.columns):
        st.error("Experimental CSV needs columns: formula, exp_gap")
        st.stop()
    if not {"formula","pbe_gap"}.issubset(dft_df.columns):
        st.error("DFT CSV needs columns: formula, pbe_gap")
        st.stop()

    # 5) Prepare & merge
    exp_df = exp_df.rename(columns={"formula":"Formula","exp_gap":"Exp Eg (eV)"})
    dft_df = dft_df.rename(columns={"formula":"Formula","pbe_gap":"DFT Eg (eV)"})
    dfm = pd.merge(dft_df, exp_df, on="Formula")

    # 6) Compute error metrics
    dfm["Δ Eg (eV)"] = dfm["DFT Eg (eV)"] - dfm["Exp Eg (eV)"]
    mae  = dfm["Δ Eg (eV)"].abs().mean()
    rmse = (dfm["Δ Eg (eV)"]**2).mean()**0.5
    st.write(f"**MAE:** {mae:.3f} eV  **RMSE:** {rmse:.3f} eV")

    # 7) Parity plot
    fig1 = px.scatter(
        dfm, x="Exp Eg (eV)", y="DFT Eg (eV)",
        text="Formula", title="Parity: DFT vs. Experimental",
        labels={"Exp Eg (eV)":"Experimental Eg (eV)",
                "DFT Eg (eV)":"DFT Eg (eV)"}
    )
    mn = dfm[["Exp Eg (eV)","DFT Eg (eV)"]].min().min()
    mx = dfm[["Exp Eg (eV)","DFT Eg (eV)"]].max().max()
    fig1.add_shape("line", x0=mn,y0=mn,x1=mx,y1=mx,
                   line=dict(dash="dash",color="gray"))
    st.plotly_chart(fig1, use_container_width=True)
    png1 = fig1.to_image(format="png", scale=2)
    st.download_button("📥 Download Parity Plot (PNG)",
                       png1, "parity_plot.png", "image/png",
                       use_container_width=True)

    # 8) Error histogram
    fig2 = px.histogram(
        dfm, x="Δ Eg (eV)", nbins=10,
        title="Error Distribution (DFT – Experimental)",
        labels={"Δ Eg (eV)":"ΔEg (eV)"}
    )
    st.plotly_chart(fig2, use_container_width=True)
    png2 = fig2.to_image(format="png", scale=2)
    st.download_button("📥 Download Error Histogram (PNG)",
                       png2, "error_histogram.png", "image/png",
                       use_container_width=True)
