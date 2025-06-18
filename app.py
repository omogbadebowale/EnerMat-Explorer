# app.py  –  EnerMat Perovskite Explorer v9.6 with Benchmark & Download Plot
# Author: Dr Gbadebo Taofeek Yusuf

import io
import os
import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from docx import Document
from dotenv import load_dotenv

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

# ─────────────────────────────────── Backend call ────────────────────────────
@st.cache_data(show_spinner="Monte-Carlo sampling …")
def run_screen(**kw):
    return screen(**kw)

# ──────────────────────────────── Run / Back logic ──────────────────────────
col_run, col_back = st.columns([3,1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=len(st.session_state["history"])<1)

if do_back and st.session_state["history"]:
    st.session_state["history"].pop()
    A, B, rh, temp, df = st.session_state["history"][-1]
    st.success("Showing previous result")

elif do_run:
    dA, dB = _summary(A), _summary(B)
    if not dA or not dB:
        st.error("Failed to fetch Materials Project data for endmembers.")
        st.stop()
    df = run_screen(A=A, B=B, rh=rh, temp=temp, bg=(bg_lo,bg_hi), bow=bow, dx=dx)
    if df.empty:
        st.error("No candidates found – try widening your window.")
        st.stop()
    st.session_state["history"].append((A,B,rh,temp,df))

elif st.session_state["history"]:
    A,B,rh,temp,df = st.session_state["history"][-1]
else:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

# ───────────────────────────────────── Tabs ─────────────────────────────────
tab_tbl, tab_plot, tab_dl, tab_bench = st.tabs(
    ["📊 Table","📈 Plot","⬇ Download","⚖ Benchmark"]
)

# ─────────── Table Tab ───────────
with tab_tbl:
    params = pd.DataFrame({
        "Parameter": ["Humidity [%]","Temperature [°C]","Gap window [eV]","Bowing [eV]","x-step"],
        "Value":     [rh,temp,f"{bg_lo:.2f}–{bg_hi:.2f}",bow,dx]
    })
    st.markdown("**Run parameters**")
    st.table(params.astype(str))

    docA, docB = _summary(A), _summary(B)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"**A-endmember: {A}**")
        st.write(f"MP band gap: {docA.band_gap:.2f} eV")
        st.write(f"MP E_above_hull: {docA.energy_above_hull:.3f} eV/atom")
    with c2:
        st.markdown(f"**B-endmember: {B}**")
        st.write(f"MP band gap: {docB.band_gap:.2f} eV")
        st.write(f"MP E_above_hull: {docB.energy_above_hull:.3f} eV/atom")

    st.dataframe(df.astype(str), height=400, use_container_width=True)

# ─────────── Plot Tab ───────────
with tab_plot:
    st.caption("ℹ️ **Tip**: Hover circles; scroll to zoom; drag to pan")
    top_cut = df["score"].quantile(0.80)
    df["is_top"] = df["score"] >= top_cut

    fig = px.scatter(
        df, x="stability", y="band_gap",
        color="score", color_continuous_scale="plasma",
        hover_data=["formula","x","band_gap","stability","score"],
        height=450
    )
    fig.update_traces(marker=dict(size=18, line_width=1), opacity=0.9)
    outline = go.Scatter(
        x=df.loc[df.is_top,"stability"],
        y=df.loc[df.is_top,"band_gap"],
        mode="markers", hoverinfo="skip",
        marker=dict(size=22,color="rgba(0,0,0,0)",line=dict(width=2,color="black")),
        showlegend=False
    )
    fig.add_trace(outline)
    fig.update_xaxes(title="<b>Stability</b>", range=[0.75,1.00], dtick=0.05)
    fig.update_yaxes(title="<b>Band-gap (eV)</b>", range=[0,3.5], dtick=0.5)
    fig.update_layout(
        template="simple_white",
        margin=dict(l=70,r=40,t=25,b=65),
        coloraxis_colorbar=dict(title="<b>Score</b>")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Download PNG
    png = fig.to_image(format="png", scale=2)
    st.download_button(
        "📥 Download plot as PNG",
        png, "stability_vs_gap.png","image/png",
        use_container_width=True
    )

# ─────────── Download Tab ───────────
with tab_dl:
    st.download_button("CSV", df.to_csv(index=False).encode(), "EnerMat_results.csv","text/csv")
    top = df.iloc[0]
    report_txt = (
        f"EnerMat report ({datetime.date.today()})\n"
        f"Top candidate : {top['formula']}\n"
        f"Band-gap     : {top['band_gap']}\n"
        f"Stability    : {top['stability']}\n"
        f"Score        : {top['score']}\n"
    )
    st.download_button("TXT report", report_txt, "EnerMat_report.txt","text/plain")

    # DOCX report
    doc = Document()
    doc.add_heading("EnerMat Report",0)
    doc.add_paragraph(f"Date: {datetime.date.today()}")
    doc.add_paragraph(f"Top candidate: {top['formula']}")
    table = doc.add_table(rows=1, cols=2)
    for k,v in [("Band-gap",top['band_gap']),("Stability",top['stability']),("Score",top['score'])]:
        row = table.add_row()
        row.cells[0].text = k
        row.cells[1].text = str(v)
    buf = io.BytesIO(); doc.save(buf); buf.seek(0)
    st.download_button(
        "📥 DOCX report", buf,
        "EnerMat_report.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

# ─────────── Benchmark Tab ───────────
with tab_bench:
    st.markdown("## ⚖ Benchmark: DFT vs. Experimental Gaps")

    # 1) Experimental upload
    uploaded = st.file_uploader("Upload experimental CSV (`formula`, `exp_gap`)", type="csv")
    if uploaded:
        exp_df = pd.read_csv(uploaded)
        st.success("Loaded experimental data from local file")
    else:
        # fallback to bundled
        bundled = Path(__file__).parent / "exp_bandgaps.csv"
        exp_df = pd.read_csv(bundled)
        st.info("No upload — using bundled experimental CSV")

    # validate
    if not {"formula","exp_gap"}.issubset(exp_df.columns):
        st.error("CSV must contain `formula` and `exp_gap` columns.")
        st.stop()
    exp_df = exp_df[["formula","exp_gap"]].rename(columns={"formula":"Formula","exp_gap":"Exp Eg (eV)"})

    # 2) Load DFT gaps
    pbe = Path(__file__).parent / "pbe_bandgaps.csv"
    dft_df = pd.read_csv(pbe)
    if not {"formula","pbe_gap"}.issubset(dft_df.columns):
        st.error("DFT CSV needs `formula` and `pbe_gap` columns.")
        st.stop()
    dft_df = dft_df.rename(columns={"formula":"Formula","pbe_gap":"DFT Eg (eV)"})

    # 3) merge & stats
    dfm = pd.merge(dft_df, exp_df, on="Formula", how="inner")
    dfm["Δ Eg (eV)"] = dfm["DFT Eg (eV)"] - dfm["Exp Eg (eV)"]
    mae  = dfm["Δ Eg (eV)"].abs().mean()
    rmse = np.sqrt((dfm["Δ Eg (eV)"]**2).mean())
    st.write(f"**MAE:** {mae:.3f} eV **RMSE:** {rmse:.3f} eV")

    # 4) Parity plot + linear fit
    x,y = dfm["Exp Eg (eV)"], dfm["DFT Eg (eV)"]
    m,b = np.polyfit(x,y,1)
    mn,mx = x.min(), x.max()
    fig1 = px.scatter(dfm, x="Exp Eg (eV)", y="DFT Eg (eV)", text="Formula")
    fig1.add_trace(go.Scatter(
        x=[mn,mx], y=[m*mn+b, m*mx+b],
        mode="lines", line=dict(dash="dash", color="gray"), showlegend=False
    ))
    fig1.update_layout(template="simple_white", margin=dict(l=60,r=40,t=30,b=60))
    st.plotly_chart(fig1, use_container_width=True)
    png1 = fig1.to_image(format="png", scale=2)
    st.download_button("📥 Download Parity Plot (PNG)", png1, "parity.png","image/png")

    # 5) Error histogram
    fig2 = px.histogram(dfm, x="Δ Eg (eV)", nbins=10, title="Error Distribution (DFT – Experimental)")
    fig2.update_layout(template="simple_white", margin=dict(l=60,r=40,t=30,b=60))
    st.plotly_chart(fig2, use_container_width=True)
    png2 = fig2.to_image(format="png", scale=2)
    st.download_button("📥 Download Error Histogram (PNG)", png2, "error_hist.png","image/png")
