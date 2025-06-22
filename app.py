import io
import os
import sys
import datetime
from pathlib import Path

# Ensure project root is on path for backend import
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from docx import Document

# ─── Load API Key ─────────────────────────────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    st.error("🛑 Please set a valid 32-character MP_API_KEY in Streamlit Secrets.")
    st.stop()

# ─── Backend Imports ──────────────────────────────────────────────────────────
from backend.perovskite_utils import (
    mix_abx3 as screen,
    screen_ternary,
    END_MEMBERS,
    fetch_mp_data as _summary,
)

# ─── Streamlit Config ─────────────────────────────────────────────────────────
st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.6")

# ─── Session State Init ───────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Sidebar Configuration ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose screening type", ["Binary A–B", "Ternary A–B–C"])

    st.header("End-members")
    preset_A = st.selectbox("Preset A", END_MEMBERS, index=0)
    preset_B = st.selectbox("Preset B", END_MEMBERS, index=1)
    custom_A = st.text_input("Custom A (optional)", "").strip()
    custom_B = st.text_input("Custom B (optional)", "").strip()
    A = custom_A or preset_A
    B = custom_B or preset_B
    if mode == "Ternary A–B–C":
        preset_C = st.selectbox("Preset C", END_MEMBERS, index=2)
        custom_C = st.text_input("Custom C (optional)", "").strip()
        C = custom_C or preset_C

    st.header("Environment")
    rh = st.slider("Humidity [%]", 0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)

    st.header("Penalty Coefficients")
    alpha = st.slider(
        "Humidity penalty (α)",
        min_value=-0.05,
        max_value=0.0,
        value=-0.012395,
        step=0.0005,
    )
    beta = st.slider(
        "Temperature penalty (β)",
        min_value=0.0,
        max_value=0.05,
        value=0.027888,
        step=0.0005,
    )

    st.header("Target Band Gap [eV]")
    bg_lo, bg_hi = st.slider("Gap window [eV]", 0.5, 3.0, (1.0, 1.4), 0.01)

    st.header("Model Settings")
    bow = st.number_input("Bowing [eV]", 0.0, 1.0, 0.30, 0.05)
    dx = st.number_input("x-step", 0.01, 0.50, 0.05, 0.01)
    if mode == "Ternary A–B–C":
        dy = st.number_input("y-step", 0.01, 0.50, 0.05, 0.01)

    if st.button("🗑 Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()

    GIT_SHA = st.secrets.get("GIT_SHA", "dev")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    st.caption(f"⚙️ Version: `{GIT_SHA}` • ⏱ {ts}")
    st.caption("© 2025 Dr Gbadebo Taofeek Yusuf")

# ─── Cached Screen Runner ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Running screening…", max_entries=50)
def run_screen(formula_A, formula_B, rh, temp, bg_window, bowing, dx, alpha, beta):
    return screen(
        formula_A=formula_A,
        formula_B=formula_B,
        rh=rh,
        temp=temp,
        bg_window=bg_window,
        bowing=bow,
        dx=dx,
        alpha=alpha,
        beta=beta,
    )

# ─── Execution Control ────────────────────────────────────────────────────────
col_run, col_back = st.columns([3, 1])
do_run = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=not st.session_state.history)

if do_back:
    prev = st.session_state.history.pop()
    mode = prev["mode"]
    A, B, rh, temp = prev["A"], prev["B"], prev["rh"], prev["temp"]
    bg_lo, bg_hi = prev["bg"]
    bow, dx = prev["bow"], prev["dx"]
    alpha, beta = prev["alpha"], prev["beta"]
    if mode == "Ternary A–B–C":
        C, dy = prev["C"], prev["dy"]
    df = prev["df"]
    st.success("Showing previous result")

elif do_run:
    try:
        docA = _summary(A, ["band_gap", "energy_above_hull"])
        docB = _summary(B, ["band_gap", "energy_above_hull"])
        if mode == "Ternary A–B–C":
            docC = _summary(C, ["band_gap", "energy_above_hull"])
    except Exception as e:
        st.error(f"❌ Error querying Materials Project: {e}")
        st.stop()

    if not docA or not docB or (mode == "Ternary A–B–C" and not docC):
        st.error("❌ Invalid formula(s) — check your entries.")
        st.stop()

    if mode == "Binary A–B":
        df = run_screen(
            formula_A=A,
            formula_B=B,
            rh=rh,
            temp=temp,
            bg_window=(bg_lo, bg_hi),
            bowing=bow,
            dx=dx,
            alpha=alpha,
            beta=beta,
        )
    else:
        df = screen_ternary(
            A=A, B=B, C=C,
            rh=rh, temp=temp,
            bg=(bg_lo, bg_hi),
            bows={"AB": bow, "AC": bow, "BC": bow},
            dx=dx, dy=dy, n_mc=200
        )

    df = df.rename(columns={"energy_above_hull": "stability", "band_gap": "Eg"})
    entry = dict(mode=mode, A=A, B=B, rh=rh, temp=temp,
                 bg=(bg_lo, bg_hi), bow=bow, dx=dx,
                 alpha=alpha, beta=beta, df=df)
    if mode == "Ternary A–B–C":
        entry.update(C=C, dy=dy)
    st.session_state.history.append(entry)

elif st.session_state.history:
    prev = st.session_state.history[-1]
    mode = prev["mode"]
    A, B, rh, temp = prev["A"], prev["B"], prev["rh"], prev["temp"]
    bg_lo, bg_hi = prev["bg"]
    bow, dx = prev["bow"], prev["dx"]
    alpha, beta = prev["alpha"], prev["beta"]
    if mode == "Ternary A–B–C":
        C, dy = prev["C"], prev["dy"]
    df = prev["df"]
else:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

# ─── Tabs & Remaining App Logic ───────────────────────────────────────────────
with st.tabs(["📊 Table", "📈 Plot", "📥 Download"])[0]:
    st.markdown("**Run parameters**")
    params = {"Parameter": ["Humidity [%]","Temperature [°C]","Gap window [eV]","Bowing [eV]","x-step","Humidity penalty (α)","Temperature penalty (β)"],
              "Value": [rh,temp,f"{bg_lo:.2f}–{bg_hi:.2f}",bow,dx,alpha,beta]}
    if mode == "Ternary A–B–C": params["Parameter"].append("y-step"); params["Value"].append(dy)
    st.table(pd.DataFrame(params))
    st.subheader("Candidate Results")
    st.dataframe(df, use_container_width=True, height=400)
with st.tabs(["📊 Table", "📈 Plot", "📥 Download"])[1]:
    plot_df = df.dropna(subset=["stability","Eg","score"]) if mode=="Binary A–B" else df.dropna(subset=["x","y","score"])
    if mode=="Binary A–B":
        fig = px.scatter(plot_df, x="stability", y="Eg", color="score", color_continuous_scale="plasma",
                          hover_data=["formula","x","Eg","stability","score"])
        top_cut = plot_df["score"].quantile(0.8)
        fig.add_trace(go.Scatter(x=plot_df.loc[plot_df.score>=top_cut,"stability"],
                                 y=plot_df.loc[plot_df.score>=top_cut,"Eg"], mode="markers",
                                 marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(width=2,color="black")),
                                 hoverinfo="skip"))
        fig.update_layout(template="simple_white", margin=dict(l=60,r=30,t=30,b=60),
                          xaxis_title="Stability", yaxis_title="Band Gap (eV)")
        st.plotly_chart(fig,use_container_width=True)
    else:
        fig3d = px.scatter_3d(plot_df, x="x",y="y",z="score",color="score", hover_data={k:True for k in ["x","y","Eg","score"]})
        fig3d.update_layout(template="simple_white")
        st.plotly_chart(fig3d,use_container_width=True)
with st.tabs(["📊 Table", "📈 Plot", "📥 Download"])[2]:
    csv = df.to_csv(index=False).encode()
    st.download_button("📥 Download CSV", csv, "EnerMat_results.csv","text/csv")
    top = df.iloc[0]
    top_label = top.formula if mode=="Binary A–B" else f"{A}-{B}-{C} x={top.x:.2f} y={top.y:.2f}"
    txt = f"""EnerMat report ({datetime.date.today()})
Top candidate : {top_label}
Band-gap     : {top.Eg}
Stability    : {getattr(top,'stability','N/A')}
Score        : {top.score}
"""
    st.download_button("📄 Download TXT", txt, "EnerMat_report.txt","text/plain")
    doc = Document()
    doc.add_heading("EnerMat Report",0)
    doc.add_paragraph(f"Date: {datetime.date.today()}")
    doc.add_paragraph(f"Top candidate: {top_label}")
    tbl = doc.add_table(rows=1,cols=2); hdr=tbl.rows[0].cells; hdr[0].text="Property"; hdr[1].text="Value"
    for k,v in [("Band-gap",top.Eg), ("Stability",getattr(top,'stability','N/A')), ("Score",top.score)]: row=tbl.add_row(); row.cells[0].text=k; row.cells[1].text=str(v)
    buf=io.BytesIO(); doc.save(buf); buf.seek(0)
    st.download_button("📝 Download DOCX",buf,"EnerMat_report.docx","application/vnd.openxmlformats-officedocument.wordprocessingml.document")
