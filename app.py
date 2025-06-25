# app.py – EnerMat Perovskite Explorer v9.8  (Streamlit front-end)
# ─────────────────────────────────────────────────────────────────────────────
import io, os, datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from docx import Document

# ─── API key check (fails fast if missing) ───────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    st.error("🛑  Please set a valid 32-character MP_API_KEY in *Secrets*.")
    st.stop()

# ─── backend helpers ─────────────────────────────────────────────────────────
from backend.perovskite_utils import (  # type: ignore
    mix_abx3 as screen,
    screen_ternary,
    END_MEMBERS,
    fetch_mp_data as _summary,
)

# ─── page layout ─────────────────────────────────────────────────────────────
st.set_page_config("EnerMat Perovskite Explorer", layout="wide")
st.title("🔬  EnerMat **Perovskite** Explorer v9.8")

if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Choose screening type")
    mode = st.radio("", ["Binary A–B", "Ternary A–B–C"])

    st.subheader("End-members")
    preset_A = st.selectbox("Preset / custom A", END_MEMBERS, 0)
    preset_B = st.selectbox("Preset / custom B", END_MEMBERS, 1)
    A = st.text_input("", key="A_override").strip() or preset_A
    B = st.text_input("", key="B_override").strip() or preset_B

    if mode == "Ternary A–B–C":
        preset_C = st.selectbox("Preset / custom C", END_MEMBERS, 2)
        C = st.text_input("", key="C_override").strip() or preset_C

    st.subheader("Environment")
    rh   = st.slider("Humidity [%]",      0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)

    st.subheader("Physics window")
    bg_lo, bg_hi = st.slider("Target gap [eV]", 0.5, 3.0, (1.0, 1.4), 0.01)

    st.subheader("Model settings")
    bow = st.number_input("Bowing [eV]", 0.0, 1.0, 0.30, 0.05)
    dx  = st.number_input("x-step",       0.01, 0.5, 0.05, 0.01)
    if mode == "Ternary A–B–C":
        dy = st.number_input("y-step",   0.01, 0.5, 0.05, 0.01)

    if st.button("🗑 Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()

# ─── cached runner (no Streamlit calls inside!) ──────────────────────────────
@st.cache_data(show_spinner="⏳ Running screening …", max_entries=30)
def _run_screen_binary(a, b, rh, temp, window, bow, dx):
    return screen(a, b, rh, temp, window, bow, dx)

@st.cache_data(show_spinner="⏳ Running screening …", max_entries=30)
def _run_screen_ternary(a, b, c, rh, temp, window, bow, dx, dy):
    return screen_ternary(a, b, c, rh, temp, window,
                          {"AB": bow, "AC": bow, "BC": bow},
                          dx, dy)

# ─── Control buttons ─────────────────────────────────────────────────────────
col_run, col_back = st.columns([4, 1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=not st.session_state.history)

# ─── Main execution block ────────────────────────────────────────────────────
if do_back:
    st.session_state.history.pop()

if do_run or st.session_state.history:
    if not do_back and do_run:
        # fetch MP docs once (to fail early if formula typo)
        okA = _summary(A, ["band_gap"])
        okB = _summary(B, ["band_gap"])
        okC = _summary(C, ["band_gap"]) if mode.startswith("Ternary") else True
        if not (okA and okB and okC):
            st.error("❌ Invalid formula or Materials-Project entry missing.")
            st.stop()

        if mode == "Binary A–B":
            df = _run_screen_binary(A, B, rh, temp, (bg_lo, bg_hi), bow, dx)
        else:
            df = _run_screen_ternary(A, B, C, rh, temp, (bg_lo, bg_hi), bow, dx, dy)

        st.session_state.history.append(
            dict(mode=mode, A=A, B=B, C=(C if mode.startswith("Ternary") else ""),
                 rh=rh, temp=temp, bg=(bg_lo, bg_hi), bow=bow, dx=dx,
                 dy=(dy if mode.startswith("Ternary") else None), df=df)
        )

    ctx = st.session_state.history[-1]
    df  = ctx["df"]
else:
    st.info("Press ▶ **Run screening** to begin.")
    st.stop()

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table", "📈 Plot", "📥 Download"])

# ─── Table view ──────────────────────────────────────────────────────────────
with tab_tbl:
    st.subheader("Candidate Results")
    st.dataframe(df, use_container_width=True, height=420)

# ─── Plot view ───────────────────────────────────────────────────────────────
with tab_plot:
    if ctx["mode"] == "Binary A–B":
        plot_df = df.copy()
        fig = px.scatter(
            plot_df, x="stability", y="Eg", color="score",
            color_continuous_scale="Turbo",
            hover_data=["formula", "x", "Eg", "stability", "gap_score", "score"],
            width=1100, height=750,
        )
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color="black")))
        top_cut = plot_df["score"].quantile(0.80)
        mask    = plot_df["score"] >= top_cut
        fig.add_trace(
            go.Scatter(
                x=plot_df.loc[mask,"stability"],
                y=plot_df.loc[mask,"Eg"],
                mode="markers",
                marker=dict(size=18, symbol="circle-open",
                            line=dict(width=2, color="black")),
                hoverinfo="skip", showlegend=False,
            )
        )
        fig.update_layout(template="plotly_white",
                          xaxis_title="Stability (exp-weighted)",
                          yaxis_title="Band gap [eV]")
        st.plotly_chart(fig, use_container_width=True)

    else:
        tdf = df.copy()
        fig3d = px.scatter_3d(
            tdf, x="x", y="y", z="score", color="score",
            color_continuous_scale="Turbo",
            hover_data={"x":True,"y":True,"Eg":True,"score":True},
            width=1100, height=800,
        )
        fig3d.update_traces(marker=dict(size=4, line=dict(width=1,color="black")))
        fig3d.update_layout(template="plotly_white",
                            scene=dict(
                                xaxis_title="A fraction",
                                yaxis_title="B fraction",
                                zaxis_title="Composite score S"))
        st.plotly_chart(fig3d, use_container_width=True)

# ─── Download view ───────────────────────────────────────────────────────────
with tab_dl:
    st.download_button("💾 CSV", df.to_csv(index=False).encode(),
                       "EnerMat_results.csv", "text/csv")

    top = df.iloc[0]
    label = top.formula if ctx["mode"].startswith("Binary") \
            else f"{ctx['A']}-{ctx['B']}-{ctx['C']}  x={top.x:.2f} y={top.y:.2f}"

    txt_report = f"""EnerMat report ({datetime.date.today()})
Top candidate : {label}
Band-gap      : {top.Eg}
Stability     : {getattr(top,'stability','N/A')}
Gap factor    : {getattr(top,'gap_score','N/A')}
Composite S   : {top.score}
"""
    st.download_button("📄 TXT", txt_report, "EnerMat_report.txt", "text/plain")

    doc = Document();  doc.add_heading("EnerMat Report", 0)
    doc.add_paragraph(f"Date: {datetime.date.today()}")
    doc.add_paragraph(f"Top candidate: {label}")
    t=doc.add_table(rows=0,cols=2)
    for k,v in [("Band-gap [eV]",top.Eg),
                ("Stability",top.stability),
                ("Gap factor",top.gap_score),
                ("Composite S",top.score)]:
        r=t.add_row().cells; r[0].text=k; r[1].text=str(v)
    buf=io.BytesIO(); doc.save(buf); buf.seek(0)
    st.download_button("📝 DOCX", buf,  "EnerMat_report.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
