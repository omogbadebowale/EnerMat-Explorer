# app.py  –  EnerMat Perovskite Explorer v9.6
# Author: Dr Gbadebo Taofeek Yusuf

import io, os, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from docx import Document
from backend.perovskite_utils import screen, END_MEMBERS, _summary
from mp_api.client import MPRester

# ─── App config ────────────────────────────────────────────────
st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.6")

if "history" not in st.session_state:
    st.session_state.history = []

# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Environment")
    rh   = st.slider("Humidity [%]", 0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)
    bg_lo, bg_hi = st.slider("Target gap [eV]", 0.5, 3.0, (1.0, 1.4), 0.01)

    st.header("Parent formulas")
    pick = END_MEMBERS
    A_pick = st.selectbox("Preset A", pick, 0)
    B_pick = st.selectbox("Preset B", pick, 1)
    A = st.text_input("Custom A (optional)", "").strip() or A_pick
    B = st.text_input("Custom B (optional)", "").strip() or B_pick

    st.header("Model knobs")
    bow = st.number_input("Bowing [eV]", 0.0, 1.0, 0.30, 0.05)
    dx  = st.number_input("x-step",    0.01, 0.50, 0.05, 0.01)

    if st.button("🗑 Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()

    st.caption("© 2025 Dr Gbadebo Taofeek Yusuf")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    st.caption(f"⏱ {ts}")

# ─── Run screening ─────────────────────────────────────────────
@st.cache_data(show_spinner="Monte-Carlo sampling …")
def run_screen(**kw):
    return screen(**kw)

col_run, col_back = st.columns([3,1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=not st.session_state.history)

if do_back:
    st.session_state.history.pop()
elif do_run:
    dA, dB = _summary(A), _summary(B)
    if not dA or not dB:
        st.error("Failed to fetch Materials Project data for endmembers.")
        st.stop()
    df = run_screen(A=A, B=B, rh=rh, temp=temp, bg=(bg_lo,bg_hi), bow=bow, dx=dx)
    if df.empty:
        st.error("No candidates – try widening your window.")
        st.stop()
    st.session_state.history.append((A,B,rh,temp,df))

if st.session_state.history:
    _, _, _, _, df = st.session_state.history[-1]
else:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

# ─── Tabs ───────────────────────────────────────────────────────
tab_tbl, tab_plot, tab_dl, tab_bench = st.tabs(
    ["📊 Table","📈 Plot","⬇ Download","⚖ Benchmark"]
)

# ─── Table ──────────────────────────────────────────────────────
with tab_tbl:
    params = pd.DataFrame({
        "Parameter":["Humidity [%]","Temperature [°C]","Gap window [eV]","Bowing [eV]","x-step"],
        "Value":[rh,temp,f"{bg_lo:.2f}–{bg_hi:.2f}",bow,dx]
    })
    st.markdown("**Run parameters**")
    st.table(params)

    docA, docB = _summary(A), _summary(B)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"**A-endmember: {A}**")
        st.write(f"MP band gap: {docA.band_gap:.2f} eV")
        st.write(f"MP E_hull: {docA.energy_above_hull:.3f} eV/atom")
    with c2:
        st.markdown(f"**B-endmember: {B}**")
        st.write(f"MP band gap: {docB.band_gap:.2f} eV")
        st.write(f"MP E_hull: {docB.energy_above_hull:.3f} eV/atom")

    st.dataframe(df, use_container_width=True, height=400)

# ─── Plot ────────────────────────────────────────────────────────
with tab_plot:
    st.caption("ℹ️ Hover & zoom")
    top_cut = df.score.quantile(0.8)
    df["is_top"] = df.score>=top_cut

    fig = px.scatter(
        df, x="stability", y="band_gap",
        color="score", color_continuous_scale="plasma",
        hover_data=["formula","x","band_gap","stability","score"],
        height=450
    )
    fig.update_traces(marker=dict(size=18,line_width=1), opacity=0.9)
    outline = fig.data[0].copy()
    outline.marker.color="rgba(0,0,0,0)"
    outline.marker.line=dict(width=2,color="black")
    outline.x = df.loc[df.is_top,"stability"]
    outline.y = df.loc[df.is_top,"band_gap"]
    fig.add_trace(outline)

    fig.update_xaxes(title="<b>Stability</b>",range=[0.75,1.0],dtick=0.05)
    fig.update_yaxes(title="<b>Band-gap (eV)</b>",range=[0,3.5],dtick=0.5)
    fig.update_layout(template="simple_white",margin=dict(l=70,r=40,t=25,b=65),
                      coloraxis_colorbar=dict(title="<b>Score</b>"))
    st.plotly_chart(fig, use_container_width=True)

    png = fig.to_image(format="png",scale=2)
    st.download_button("📥 Download plot as PNG", png, "stability_vs_gap.png","image/png",
                       use_container_width=True)

# ─── Download ─────────────────────────────────────────────────────
with tab_dl:
    csv = df.to_csv(index=False).encode()
    st.download_button("CSV", csv, "EnerMat_results.csv","text/csv")
    top = df.iloc[0]
    txt = (
        f"EnerMat report ({datetime.date.today()})\n"
        f"Top candidate: {top.formula}\n"
        f"Band-gap     : {top.band_gap}\n"
        f"Stability    : {top.stability}\n"
        f"Score        : {top.score}\n"
    )
    st.download_button("TXT report", txt, "EnerMat_report.txt","text/plain")

    doc = Document()
    doc.add_heading("EnerMat Report",0)
    doc.add_paragraph(f"Date: {datetime.date.today()}")
    doc.add_paragraph(f"Top candidate: {top.formula}")
    tbl = doc.add_table(rows=1,cols=2)
    for k,v in [("Band-gap",top.band_gap),("Stability",top.stability),("Score",top.score)]:
        row = tbl.add_row()
        row.cells[0].text, row.cells[1].text = k, str(v)
    buf = io.BytesIO(); doc.save(buf); buf.seek(0)
    st.download_button("📥 DOCX report", buf, "EnerMat_report.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# ─── Benchmark ────────────────────────────────────────────────────
with tab_bench:
    st.markdown("## ⚖ Benchmark: DFT vs. Experimental Gaps")

    # (1) Experimental uploader or fallback
    uploaded = st.file_uploader("Upload experimental CSV (`formula`,`exp_gap`)",type="csv")
    if uploaded:
        exp_df = pd.read_csv(uploaded)
        st.success("Loaded experimental data from local file")
    else:
        st.info("Fetching bundled experimental CSV…")
        exp_df = pd.read_csv("exp_bandgaps.csv")
        st.success("Loaded experimental data from bundled CSV")

    if not {"formula","exp_gap"}.issubset(exp_df.columns):
        st.error("CSV must have `formula` and `exp_gap` columns."); st.stop()

    # (2) DFT loader
    dfb = pd.read_csv("pbe_bandgaps.csv")  # expects columns formula,pbe_gap
    dfb = dfb.rename(columns={"formula":"Formula","pbe_gap":"DFT Eg (eV)"})

    exp_df = exp_df.rename(columns={"formula":"Formula","exp_gap":"Exp Eg (eV)"})
    dfm = pd.merge(dfb, exp_df, on="Formula", how="inner")
    dfm["Δ Eg (eV)"] = dfm["DFT Eg (eV)"] - dfm["Exp Eg (eV)"]

    # (3) Stats
    mae  = dfm["Δ Eg (eV)"].abs().mean()
    rmse = np.sqrt((dfm["Δ Eg (eV)"]**2).mean())
    st.write(f"**MAE:** {mae:.3f} eV **RMSE:** {rmse:.3f} eV")

    # (4) Label selector
    formulas = sorted(dfm["Formula"].unique())
    to_label = st.multiselect(
        "Formulas to draw labels for:",
        formulas,
        default=formulas[:5],
        help="Pick a few to avoid overlap"
    )

    # (5) Scatter with jitter & text
    jitter = np.random.normal(scale=0.005, size=len(dfm))
    dfm["Exp_jit"] = dfm["Exp Eg (eV)"] + jitter
    labels = [f if f in to_label else "" for f in dfm["Formula"]]

    fig1 = px.scatter(
        dfm,
        x="Exp_jit", y="DFT Eg (eV)",
        hover_name="Formula",
        title="Parity Plot: DFT vs. Experimental"
    )
    fig1.update_traces(
        marker=dict(size=8, opacity=0.8),
        text=labels,
        textposition="top center",
        textfont=dict(size=10)
    )

    mn, mx = dfm[["Exp Eg (eV)","DFT Eg (eV)"]].min().min(), dfm[["Exp Eg (eV)","DFT Eg (eV)"]].max().max()
    # 1:1 line
    fig1.add_shape("line", x0=mn, y0=mn, x1=mx, y1=mx,
                   line=dict(dash="dash", color="gray"))

    # linear trend
    m,b = np.polyfit(dfm["Exp Eg (eV)"], dfm["DFT Eg (eV)"], 1)
    fig1.add_shape("line",
        x0=mn, y0=m*mn+b, x1=mx, y1=m*mx+b,
        line=dict(dash="dot", color="black")
    )

    st.plotly_chart(fig1, use_container_width=True)
    png1 = fig1.to_image(format="png",scale=2)
    st.download_button("📥 Download parity plot (PNG)", png1, "parity_plot.png", "image/png")

    # (6) Error histogram
    fig2 = px.histogram(
        dfm, x="Δ Eg (eV)",
        nbins=20,
        title="Error Distribution (DFT – Experimental)"
    )
    st.plotly_chart(fig2, use_container_width=True)
    png2 = fig2.to_image(format="png",scale=2)
    st.download_button("📥 Download error histogram (PNG)", png2, "error_histogram.png","image/png")
