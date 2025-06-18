# app.py — EnerMat Perovskite Explorer v9.6 (with Publication-Ready Benchmark)
# Author: Dr Gbadebo Taofeek Yusuf

import io, os, datetime
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from docx import Document
from mp_api.client import MPRester

from backend.perovskite_utils import screen, END_MEMBERS, _summary

# ─── Page config ─────────────────────────────────────────────────────
st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.6")

# ─── Sidebar ──────────────────────────────────────────────────────────
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
    dx  = st.number_input("x-step",    0.01, 0.50, 0.05, 0.01)

    if st.button("🗑 Clear history"):
        st.session_state.clear()
        st.experimental_rerun()

    st.caption("© 2025 Dr Gbadebo Taofeek Yusuf")
    GIT_SHA = st.secrets.get("GIT_SHA", "dev")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    st.caption(f"⚙️ Version: `{GIT_SHA}` • ⏱ {ts}")

# ─── Monte-Carlo sampling (cached) ────────────────────────────────────
@st.cache_data(show_spinner="Monte-Carlo sampling …")
def run_screen(**kwargs):
    return screen(**kwargs)

# ─── Run logic ───────────────────────────────────────────────────────
col_run, col_back = st.columns([3,1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=("history" not in st.session_state))

if do_back:
    st.session_state.history.pop()
elif do_run:
    dA, dB = _summary(A), _summary(B)
    if not dA or not dB:
        st.error("Failed to fetch Materials Project data.")
        st.stop()
    df = run_screen(A=A, B=B, rh=rh, temp=temp, bg=(bg_lo,bg_hi), bow=bow, dx=dx)
    if df.empty:
        st.error("No candidates found – widen your window.")
        st.stop()
    st.session_state.history = st.session_state.get("history", []) + [(A,B,rh,temp,df)]
elif "history" in st.session_state:
    A,B,rh,temp,df = st.session_state.history[-1]
else:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

# ─── Tabs ───────────────────────────────────────────────────────────
tab_tbl, tab_plot, tab_dl, tab_bench = st.tabs(
    ["📊 Table","📈 Plot","⬇ Download","⚖ Benchmark"]
)

# ─── Table Tab ──────────────────────────────────────────────────────
with tab_tbl:
    params = pd.DataFrame({
        "Parameter": ["Humidity [%]","Temperature [°C]","Gap window [eV]","Bowing [eV]","x-step"],
        "Value":     [rh,temp,f"{bg_lo:.2f}–{bg_hi:.2f}",bow,dx]
    })
    st.markdown("**Run parameters**")
    st.table(params)

    docA, docB = _summary(A), _summary(B)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"**A-endmember: {A}**")
        st.write(f"Band gap: {docA.band_gap:.2f} eV")
        st.write(f"E_above_hull: {docA.energy_above_hull:.3f} eV/atom")
    with c2:
        st.markdown(f"**B-endmember: {B}**")
        st.write(f"Band gap: {docB.band_gap:.2f} eV")
        st.write(f"E_above_hull: {docB.energy_above_hull:.3f} eV/atom")

    st.dataframe(df, height=400, use_container_width=True)

# ─── Plot Tab ────────────────────────────────────────────────────────
with tab_plot:
    st.caption("ℹ️ Hover to see formula, scroll/drag to zoom")
    top_cut = df.score.quantile(0.80)
    df["is_top"] = df.score >= top_cut

    fig = px.scatter(
        df, x="stability", y="band_gap", color="score",
        color_continuous_scale="plasma",
        hover_data=["formula","x","band_gap","stability","score"],
        title="Stability vs. Band-Gap Score",
        labels={"stability":"Stability","band_gap":"Band-Gap (eV)"},
        height=450
    )
    fig.update_traces(marker=dict(size=16, line_width=1), opacity=0.9)
    fig.add_trace(px.scatter(df[df.is_top], x="stability", y="band_gap")
                  .update_traces(marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(width=2,color="black")))
                  .data[0])
    fig.update_layout(template="simple_white",
                      margin=dict(l=60,r=20,t=40,b=60),
                      coloraxis_colorbar=dict(title="Score"))
    st.plotly_chart(fig, use_container_width=True)

    # Download
    png = fig.to_image(format="png", scale=2)
    st.download_button("📥 Download Plot (PNG)",
                       png, "stability_vs_gap.png", "image/png",
                       use_container_width=True)

# ─── Download Tab ────────────────────────────────────────────────────
with tab_dl:
    st.download_button("⬇ CSV", df.to_csv(index=False).encode(),
                       "EnerMat_results.csv","text/csv")
    top = df.iloc[0]
    txt = (
        f"EnerMat report ({datetime.date.today()})\n"
        f"Top candidate: {top.formula}\n"
        f"Band-gap: {top.band_gap}\n"
        f"Stability: {top.stability}\n"
        f"Score: {top.score}\n"
    )
    st.download_button("📄 TXT report", txt, "EnerMat_report.txt","text/plain")

    doc = Document()
    doc.add_heading("EnerMat Report",0)
    doc.add_paragraph(f"Date: {datetime.date.today()}")
    doc.add_paragraph(f"Top candidate: {top.formula}")
    tbl = doc.add_table(rows=1, cols=2)
    for k,v in [("Band-gap",top.band_gap),("Stability",top.stability),("Score",top.score)]:
        r = tbl.add_row(); r.cells[0].text=k; r.cells[1].text=str(v)
    buf = io.BytesIO(); doc.save(buf); buf.seek(0)
    st.download_button("📥 DOCX report", buf,
                       "EnerMat_report.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# ─── Benchmark Tab ────────────────────────────────────────────────────
with tab_bench:
    st.markdown("## ⚖ Benchmark: DFT vs. Experimental Gaps")

    # 1) Upload or load bundled
    uploaded = st.file_uploader("Upload experimental CSV (`formula`,`exp_gap`)", type="csv")
    if uploaded:
        exp_df = pd.read_csv(uploaded)
        st.success("Loaded experimental data from uploaded file")
    else:
        exp_csv = Path(__file__).parent / "exp_bandgaps.csv"
        exp_df = pd.read_csv(exp_csv)
        st.info("Loaded experimental data from bundled CSV")

    # Validate cols
    if not {"formula","exp_gap"}.issubset(exp_df.columns):
        st.error("CSV must contain `formula` and `exp_gap` columns.")
        st.stop()

    # 2) Load DFT bundle
    pbe_csv = Path(__file__).parent / "pbe_bandgaps.csv"
    pbe_df = pd.read_csv(pbe_csv)
    if not {"formula","pbe_gap"}.issubset(pbe_df.columns):
        st.error("Bundled DFT CSV needs `formula`,`pbe_gap`.")
        st.stop()
    st.info(f"Loaded {len(pbe_df)} DFT band gaps from bundled CSV")

    # 3) Merge & stats
    exp_df = exp_df.rename(columns={"formula":"formula","exp_gap":"Exp Eg"})
    pbe_df = pbe_df.rename(columns={"formula":"formula","pbe_gap":"DFT Eg"})
    dfm = pd.merge(pbe_df, exp_df, on="formula", how="inner")
    dfm["Δ Eg"] = dfm["DFT Eg"] - dfm["Exp Eg"]

    mae  = dfm["Δ Eg"].abs().mean()
    rmse = np.sqrt((dfm["Δ Eg"]**2).mean())
    st.write(f"**MAE:** {mae:.3f} eV    **RMSE:** {rmse:.3f} eV")

    # 4) Parity plot (publication style)
    fig1 = px.scatter(
        dfm, x="Exp Eg", y="DFT Eg",
        hover_name="formula",
        title="Parity Plot: DFT vs. Experimental",
        labels={"Exp Eg":"Experimental Eg (eV)","DFT Eg":"DFT Eg (eV)"},
        trendline="ols", trendline_color_override="gray"
    )
    fig1.update_traces(marker=dict(size=8))
    fig1.update_layout(template="simple_white",
                       font=dict(size=12),
                       margin=dict(l=60,r=20,t=50,b=50))
    st.plotly_chart(fig1, use_container_width=True)
    png1 = fig1.to_image(format="png", scale=2)
    st.download_button("📥 Download Parity Plot (PNG)", png1,
                       "parity_plot.png","image/png", use_container_width=True)

    # 5) Error histogram
    fig2 = px.histogram(dfm, x="Δ Eg", nbins=20,
                        title="Error Distribution (DFT − Experimental)",
                        labels={"Δ Eg":"Δ Eg (eV)","count":"Count"})
    fig2.update_layout(template="simple_white",
                       font=dict(size=12),
                       margin=dict(l=60,r=20,t=50,b=50))
    st.plotly_chart(fig2, use_container_width=True)
    png2 = fig2.to_image(format="png", scale=2)
    st.download_button("📥 Download Error Histogram (PNG)", png2,
                       "error_histogram.png","image/png", use_container_width=True)
