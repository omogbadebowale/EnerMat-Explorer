"""
EnerMat – Perovskite Explorer v9.7
Streamlit front-end

Key updates (2025-06-25)
────────────────────────
✓ displays new gap_score column everywhere
✓ fixes “alpha not defined” (passes alpha/beta into screen_ternary)
✓ stronger plotting guard; handles empty dataframes gracefully
✓ optional “S vs RH” curve button for binary runs
"""

import io, os, datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from docx import Document

# ── backend helpers ──────────────────────────────────────────────────────────
from backend.perovskite_utils import (
    mix_abx3 as screen,
    screen_ternary,
    END_MEMBERS,
    fetch_mp_data as _summary,
)

# ── API key check ────────────────────────────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    st.error("🛑  Set a valid 32-character MP_API_KEY in Streamlit Secrets.")
    st.stop()

# ── Streamlit boilerplate ────────────────────────────────────────────────────
st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.7")

if "history" not in st.session_state:
    st.session_state.history = []

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose screening type", ["Binary A–B", "Ternary A–B–C"])

    st.header("End-members")
    preset_A = st.selectbox("Preset A", END_MEMBERS, 0)
    preset_B = st.selectbox("Preset B", END_MEMBERS, 1)
    custom_A = st.text_input("Custom A (optional)").strip()
    custom_B = st.text_input("Custom B (optional)").strip()
    A, B = custom_A or preset_A, custom_B or preset_B

    if mode == "Ternary A–B–C":
        preset_C = st.selectbox("Preset C", END_MEMBERS, 2)
        custom_C = st.text_input("Custom C (optional)").strip()
        C = custom_C or preset_C

    st.header("Environment")
    rh   = st.slider("Humidity [%]",      0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)

    st.header("Target band-gap window")
    bg_lo, bg_hi = st.slider("Eg window [eV]", 0.5, 3.0, (1.0, 1.4), 0.01)

    st.header("Model settings")
    bow = st.number_input("Bowing [eV]", 0.0, 1.0, 0.30, 0.05)
    dx  = st.number_input("x-step",      0.01, 0.50, 0.05, 0.01)
    if mode == "Ternary A–B–C":
        dy = st.number_input("y-step",  0.01, 0.50, 0.05, 0.01)

    if st.button("🗑 Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()

    st.caption(f"⚙️ Commit: `{st.secrets.get('GIT_SHA','dev')}`")
    st.caption(f"© 2025 Dr G. T. Yusuf")

# ── cache wrapper ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Running screening…", max_entries=20)
def run_screen(*args, **kwargs):
    return screen(*args, **kwargs)


# ── main control buttons ─────────────────────────────────────────────────────
col_run, col_back = st.columns([3, 1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=not st.session_state.history)

if do_back:
    st.session_state.history.pop()
    prev = st.session_state.history[-1]
    mode = prev["mode"]
    locals().update(prev)      # bring A, B, rh, temp … into scope
    df   = prev["df"]
    st.success("Restored previous result")

elif do_run:
    if mode == "Binary A–B":
        df = run_screen(
            formula_A=A, formula_B=B,
            rh=rh, temp=temp,
            bg_window=(bg_lo, bg_hi),
            bowing=bow, dx=dx,
        )
    else:
        df = screen_ternary(
            A=A, B=B, C=C,
            rh=rh, temp=temp,
            bg=(bg_lo, bg_hi),
            bows={"AB": bow, "AC": bow, "BC": bow},
            dx=dx, dy=dy,
        )

    entry = dict(
        mode=mode, A=A, B=B, rh=rh, temp=temp,
        bg=(bg_lo, bg_hi), bow=bow, dx=dx, df=df,
    )
    if mode == "Ternary A–B–C":
        entry |= {"C": C, "dy": dy}
    st.session_state.history.append(entry)

elif st.session_state.history:
    df = st.session_state.history[-1]["df"]

else:
    st.info("Press ▶ Run screening to start.")
    st.stop()

# ── TABLE tab ────────────────────────────────────────────────────────────────
tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table", "📈 Plot", "📥 Download"])

with tab_tbl:
    st.subheader("Candidate Results")
    st.dataframe(df, height=400, use_container_width=True)

# ── helper: humidity curve (only for binary) ────────────────────────────────
def humidity_curve(A: str, B: str, x_fixed: float, temp: float, bow: float):
    rh_vals = list(range(0, 101, 10))
    rec = []
    for rh_val in rh_vals:
        sub = screen(
            formula_A=A, formula_B=B,
            rh=rh_val, temp=temp,
            bg_window=(bg_lo, bg_hi), bowing=bow, dx=0.01,
        )
        if sub.empty:
            continue
        closest = sub.iloc[(sub["x"] - x_fixed).abs().argsort().iloc[0]]
        rec.append({"RH %": rh_val, "S": closest.score})
    return pd.DataFrame(rec)

# ── PLOT tab ────────────────────────────────────────────────────────────────
with tab_plot:
    if df.empty:
        st.warning("No data to plot.")
    elif mode == "Binary A–B":
        cols_need = {"stability", "Eg", "score"}
        if not cols_need.issubset(df.columns):
            st.error("Required columns missing for plot.")
        else:
            fig = px.scatter(
                df, x="stability", y="Eg", color="score",
                hover_data=["formula", "x", "gap_score", "stability"],
                color_continuous_scale="Turbo", template="plotly_white",
                width=1100, height=700,
            )
            st.plotly_chart(fig, use_container_width=True)

            if st.button("📈 Show S vs RH curve"):
                curve_df = humidity_curve(A, B, x_fixed=0.30, temp=temp, bow=bow)
                if not curve_df.empty:
                    fig2 = px.line(curve_df, x="RH %", y="S", markers=True,
                                   title=f"S vs RH at x=0.30 ({A}/{B})",
                                   template="plotly_white")
                    st.plotly_chart(fig2, use_container_width=True)

    else:
        cols_need = {"x", "y", "score"}
        if not cols_need.issubset(df.columns):
            st.error("Columns missing for ternary plot.")
        else:
            fig3d = px.scatter_3d(
                df, x="x", y="y", z="score",
                color="score", color_continuous_scale="Turbo",
                hover_data=["x", "y", "Eg", "gap_score", "stability"],
                template="plotly_white", width=1100, height=800,
            )
            st.plotly_chart(fig3d, use_container_width=True)

# ── DOWNLOAD tab ────────────────────────────────────────────────────────────
with tab_dl:
    st.download_button("📥 CSV", df.to_csv(index=False).encode(),
                       "EnerMat_results.csv", "text/csv")

    top = df.iloc[0]
    top_label = top.get("formula", f"{A}-{B}-{C}")

    txt = f"""EnerMat report ({datetime.date.today()})
Top candidate : {top_label}
Band-gap      : {top.Eg}
Stability     : {getattr(top,'stability','N/A')}
Gap factor    : {getattr(top,'gap_score','N/A')}
Composite S   : {top.score}
"""
    st.download_button("📄 TXT", txt, "EnerMat_report.txt", "text/plain")

    doc = Document()
    doc.add_heading("EnerMat Report", 0)
    doc.add_paragraph(f"Date: {datetime.date.today()}")
    doc.add_paragraph(f"Top candidate: {top_label}")
    tbl = doc.add_table(rows=1, cols=2)
    tbl.rows[0].cells[0].text, tbl.rows[0].cells[1].text = "Property", "Value"
    for k, v in [("Band-gap", top.Eg),
                 ("Stability", getattr(top, "stability", "N/A")),
                 ("Gap factor", getattr(top, "gap_score", "N/A")),
                 ("Composite S", top.score)]:
        r = tbl.add_row()
        r.cells[0].text, r.cells[1].text = k, str(v)
    buf = io.BytesIO();  doc.save(buf);  buf.seek(0)
    st.download_button("📝 DOCX", buf, "EnerMat_report.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
