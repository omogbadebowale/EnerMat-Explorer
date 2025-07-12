# app.py – EnerMat Perovskite Explorer v9.6  (final hot-fix 2025-07-12)

import io, os, datetime, streamlit as st
import pandas as pd
import plotly.express as px
from docx import Document

# ── MP API key ──────────────────────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    st.error("🛑 Please set a valid 32-character MP_API_KEY in Streamlit Secrets."); st.stop()

# ── Backend helpers ─────────────────────────────────────────────────
from backend.perovskite_utils import (
    mix_abx3 as screen,
    screen_ternary,
    END_MEMBERS,
    fetch_mp_data as _summary,
)

# ── Streamlit config ───────────────────────────────────────────────
st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.6")

# ── Persistent session state ───────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar controls ───────────────────────────────────────────────
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose screening type", ["Binary A–B", "Ternary A–B–C"])

    st.header("End-members")
    preset_A = st.selectbox("Preset A", END_MEMBERS, 0)
    preset_B = st.selectbox("Preset B", END_MEMBERS, 1)
    custom_A = st.text_input("Custom A (optional)", "").strip()
    custom_B = st.text_input("Custom B (optional)", "").strip()
    A, B = custom_A or preset_A, custom_B or preset_B
    if mode == "Ternary A–B–C":
        preset_C = st.selectbox("Preset C", END_MEMBERS, 2)
        custom_C = st.text_input("Custom C (optional)", "").strip()
        C = custom_C or preset_C

    st.header("Environment")
    rh   = st.slider("Humidity [%]", 0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)

    st.header("Target Band-Gap")
    bg_lo, bg_hi = st.slider("Gap window [eV]", 0.5, 3.0, (1.0, 1.4), 0.01)

    st.header("Model settings")
    bow = st.number_input("Bowing (eV, neg ⇒ gap ↑)", -1.0, 1.0, -0.15, 0.05)
    dx  = st.number_input("x-step", 0.01, 0.50, 0.05, 0.01)
    if mode == "Ternary A–B–C":
        dy = st.number_input("y-step", 0.01, 0.50, 0.05, 0.01)

    if st.button("🗑 Clear history"):
        st.session_state.history.clear(); st.experimental_rerun()

    st.caption(f"⚙️ Version: `{st.secrets.get('GIT_SHA','dev')}` • ⏱ {datetime.datetime.now():%Y-%m-%d %H:%M}")
    st.caption("© 2025 Dr Gbadebo Taofeek Yusuf")

# ── Cache wrapper ──────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Running screening…", max_entries=20)
def run_screen(A, B, rh, temp, bg, bow, dx):
    return screen(formula_A=A, formula_B=B, rh=rh, temp=temp,
                  bg_window=bg, bowing=bow, dx=dx)

# ── Control buttons ────────────────────────────────────────────────
col_run, col_back = st.columns([3,1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=not st.session_state.history)

# ── Restore previous result ────────────────────────────────────────
if do_back:
    st.session_state.history.pop()
    df = st.session_state.history[-1]["df"]

# ── Execute a new run ──────────────────────────────────────────────
elif do_run:
    if mode == "Binary A–B":
        df = run_screen(A, B, rh, temp, (bg_lo, bg_hi), bow, dx)
    else:
        df = screen_ternary(A, B, C, rh, temp, (bg_lo, bg_hi),
                            {"AB":bow,"AC":bow,"BC":bow}, dx, dy, 200)

    # ensure Plot tab always sees "stability"
    if "stability" not in df.columns and "Ehull" in df.columns:
        df["stability"] = df["Ehull"]

    # keep for navigation
    st.session_state.history.append({"df": df})

# ── No data yet ────────────────────────────────────────────────────
elif not st.session_state.history:
    st.info("Press ▶ Run screening to begin."); st.stop()
else:
    df = st.session_state.history[-1]["df"]

# ─── Tabs ──────────────────────────────────────────────────────────
tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table", "📈 Plot", "📥 Download"])

# ─── Table tab ─────────────────────────────────────────────────────
with tab_tbl:
    st.dataframe(df, use_container_width=True, height=400)
    st.caption("‘stability’ column = raw Ehull (eV atom⁻¹)")

# ─── Plot tab ──────────────────────────────────────────────────────
with tab_plot:
    need = {"stability", "Eg", "score"}
    if not need.issubset(df.columns):
        st.error("❌ Missing required columns for plotting."); st.stop()

    fig = px.scatter(df, x="stability", y="Eg", color="score",
                     color_continuous_scale="Turbo",
                     hover_data=[c for c in df.columns if c not in {"stability"}],
                     width=1200, height=800)
    fig.update_traces(marker=dict(size=12, opacity=0.9,
                                  line=dict(width=1, color="black")))
    st.plotly_chart(fig, use_container_width=True)

# ─── Download tab ──────────────────────────────────────────────────
with tab_dl:
    st.download_button("📥 Download CSV",
                       df.to_csv(index=False).encode(),
                       "EnerMat_results.csv", "text/csv")

    top = df.iloc[0]
    txt = (
        f"EnerMat report ({datetime.date.today()})\n"
        f"Top candidate : {top.formula if 'formula' in top else 'N/A'}\n"
        f"Band-gap      : {top.Eg}\n"
        f"Ehull (eV)    : {top.get('Ehull', top.stability)}\n"
        f"Score         : {top.score}\n"
    )
    st.download_button("📄 Download TXT", txt, "EnerMat_report.txt", "text/plain")

    doc = Document()
    doc.add_heading("EnerMat Report", 0)
    for line in txt.splitlines():
        doc.add_paragraph(line)
    buf = io.BytesIO(); doc.save(buf); buf.seek(0)
    st.download_button("📝 Download DOCX", buf, "EnerMat_report.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
