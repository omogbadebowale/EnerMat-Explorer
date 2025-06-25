import io, os, datetime
import streamlit as st
import numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
from docx import Document
from backend.perovskite_utils import (
    mix_abx3 as screen,            # binary
    screen_ternary,                # ternary
    END_MEMBERS, _summary          # quick MP peek
)

# ── API key guard ────────────────────────────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    st.error("🛑  Please set MP_API_KEY in the Secrets panel.")
    st.stop()

# ── Streamlit page basics ───────────────────────────────────────────────────
st.set_page_config(page_title="EnerMat Explorer", layout="wide")
st.title("🔬  EnerMat Perovskite Explorer v9.7")

if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar – inputs ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose screening type", ["Binary A–B", "Ternary A–B–C"])

    st.header("End-members")
    A = st.selectbox("Preset / custom A", END_MEMBERS, key="A")
    B = st.selectbox("Preset / custom B", END_MEMBERS, key="B")
    if mode == "Ternary A–B–C":
        C = st.selectbox("Preset / custom C", END_MEMBERS, key="C")

    st.header("Environment")
    rh   = st.slider("Humidity [%]",      0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)

    st.header("Physics window")
    bg_lo, bg_hi = st.slider("Target gap [eV]", 0.5, 3.0, (1.0, 1.4), 0.01)

    st.header("Calculation knobs")
    bow = st.number_input("Bowing [eV]", 0.0, 1.0, 0.30, 0.05)
    dx  = st.number_input("x-step", 0.01, 0.50, 0.05, 0.01)
    if mode == "Ternary A–B–C":
        dy = st.number_input("y-step", 0.01, 0.50, 0.05, 0.01)

    if st.button("🗑  Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()

# ── Run / back buttons ───────────────────────────────────────────────────────
col_run, col_back = st.columns([3,1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=not st.session_state.history)

@st.cache_data(show_spinner="⏳  querying MP …", max_entries=20)
def run_screen(**kw):
    return screen(**kw)

# ── Control flow ────────────────────────────────────────────────────────────
if do_back:
    st.session_state.history.pop()
elif do_run:
    if mode == "Binary A–B":
        df = run_screen(
            formula_A=A, formula_B=B,
            rh=rh, temp=temp,
            bg_window=(bg_lo, bg_hi),
            bowing=bow, dx=dx
        )
    else:
        df = screen_ternary(
            A=A, B=B, C=C,
            rh=rh, temp=temp,
            bg=(bg_lo, bg_hi),
            bows={"AB": bow, "AC": bow, "BC": bow},
            dx=dx, dy=dy
        )
    st.session_state.history.append({"df":df, "mode":mode, "A":A, "B":B,
                                     **({"C":C} if mode.startswith("Ternary") else {})})

if not st.session_state.history:
    st.info("Press ▶ Run screening")
    st.stop()

rec = st.session_state.history[-1]
df, mode = rec["df"], rec["mode"]

# ── Tabs ────────────────────────────────────────────────────────────────────
tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table", "📈 Plot", "📥 Download"])

# ===  TABLE  ================================================================
with tab_tbl:
    st.subheader("Candidate Results")
    st.dataframe(df, use_container_width=True, height=420)

# ===  PLOT  ================================================================
with tab_plot:
    if mode.startswith("Binary"):
        fig = px.scatter(
            df, x="stability", y="Eg", color="score", color_continuous_scale="Turbo",
            hover_data=df.columns.tolist(), width=1100, height=720
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig3d = px.scatter_3d(
            df, x="x", y="y", z="score", color="score",
            color_continuous_scale="Turbo", hover_data=df.columns.tolist(),
            width=1100, height=760
        )
        fig3d.update_layout(template="plotly_white")
        st.plotly_chart(fig3d, use_container_width=True)

# ===  DOWNLOADS  ============================================================
with tab_dl:
    st.download_button("📥  CSV", df.to_csv(index=False), "EnerMat_results.csv")
    top = df.iloc[0]
    top_label = (f"{A}-{B}" if mode=="Binary A–B"
                 else f"{A}-{B}-{rec['C']} x={top.x:.2f} y={top.y:.2f}")

    txt = (
        f"EnerMat report ({datetime.date.today()})\n"
        f"Top candidate : {top_label}\n"
        f"Band-gap      : {top.Eg}\n"
        f"Stability     : {getattr(top,'stability','N/A')}\n"
        f"Gap factor    : {getattr(top,'gap_score','N/A')}\n"
        f"Composite S   : {top.score}\n"
    )
    st.download_button("📄 TXT", txt, "EnerMat_report.txt")

    doc = Document(); doc.add_heading("EnerMat Report",0)
    tbl = doc.add_table(rows=1, cols=2); hdr = tbl.rows[0].cells
    hdr[0].text, hdr[1].text = "Property", "Value"
    for k,v in [("Band-gap",top.Eg),
                ("Stability", getattr(top,"stability","—")),
                ("Gap factor", getattr(top,"gap_score","—")),
                ("Composite S", top.score)]:
        row = tbl.add_row(); row.cells[0].text, row.cells[1].text = k, str(v)
    buf = io.BytesIO(); doc.save(buf); buf.seek(0)
    st.download_button("📝 DOCX", buf,
        "EnerMat_report.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
