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
col1, col2 = st.columns([1, 12])
with col1:
    st.empty()  # logo removed for now
with col2:
    st.markdown("## **EnerMat Perovskite Explorer v9.6**")

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
@st.cache_data(show_spinner="⏳ Running screening…")
def run_screen(**kw):
    return screen(**kw)

# ─── Execution Control ────────────────────────────────────────────────────────
col_run, col_back = st.columns([3, 1])
do_run = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=len(st.session_state.history) < 1)

if do_back and st.session_state.history:
    st.session_state.history.pop()
    A, B, rh, temp, df = st.session_state.history[-1]
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
        df = run_screen(formula_A=A, formula_B=B, rh=rh, temp=temp, bg_window=(bg_lo, bg_hi), bowing=bow, dx=dx)
    else:
        try:
            df = screen_ternary(
                A=A, B=B, C=C,
                rh=rh, temp=temp,
                bg=(bg_lo, bg_hi),
                bows={"AB": bow, "AC": bow, "BC": bow},
                dx=dx, dy=dy, n_mc=200
            )
        except Exception as e:
            st.error(f"❌ Ternary error: {e}")
            st.stop()

    if df.empty:
        st.error("No candidates found – try widening your gap or composition window.")
        st.stop()

    st.session_state.history.append((A, B, rh, temp, df))
elif st.session_state.history:
    A, B, rh, temp, df = st.session_state.history[-1]
else:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table", "📈 Plot", "📥 Download"])

# ─── Table Tab ───────────────────────────────────────────────────────────────
with tab_tbl:
    st.markdown("**Run parameters**")
    param_data = {
        "Parameter": ["Humidity [%]", "Temperature [°C]", "Gap window [eV]", "Bowing [eV]", "x-step"],
        "Value": [rh, temp, f"{bg_lo:.2f}–{bg_hi:.2f}", bow, dx]
    }
    if mode == "Ternary A–B–C":
        param_data["Parameter"].append("y-step")
        param_data["Value"].append(dy)

    st.table(pd.DataFrame(param_data))

    st.subheader("Candidate Results")
    st.dataframe(df, use_container_width=True, height=400)

# ─── Plot Tab ────────────────────────────────────────────────────────────────
with tab_plot:
    if mode == "Binary A–B":
        required = [col for col in ["stability", "Eg", "score"] if col in df.columns]
        if not required:
            st.warning("❗ Required binary plot columns missing from results. Skipping plot.")
            st.stop()
        plot_df = df.dropna(subset=required).copy()
        for col in required:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df = plot_df.dropna(subset=required)

        top_cut = plot_df["score"].quantile(0.80)
        plot_df["is_top"] = plot_df["score"] >= top_cut

        try:
            fig = px.scatter(
                plot_df,
                x="x",
                y="Eg",
                color="score",
                color_continuous_scale="plasma",
                hover_data=["formula", "x", "Eg", "stability", "score"]
            )
            fig.update_traces(marker=dict(size=14, line_width=1), opacity=0.85)
            fig.add_trace(
                go.Scatter(
                    x=plot_df.loc[plot_df["is_top"], "stability"],
                    y=plot_df.loc[plot_df["is_top"], "Eg"],
                    mode="markers",
                    marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(width=2, color="black")),
                    hoverinfo="skip", showlegend=False
                )
            )
            fig.update_layout(template="simple_white", margin=dict(l=60, r=30, t=30, b=60))
            fig.update_xaxes(title="Stability")
            fig.update_yaxes(title="Band Gap (eV)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Plot error: {e}")

    else:  # Ternary A–B–C
        required = [col for col in ["x", "y", "score"] if col in df.columns]
        if not required:
            st.warning("❗ No required ternary plot columns found. Cannot generate 3D plot.")
            st.stop()
        plot_df = df.dropna(subset=required).copy()
        for col in required:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df = plot_df.dropna(subset=required)

        try:
            fig3d = px.scatter_3d(
                plot_df,
                x="x",
                y="y",
                z="score",
                color="score",
                hover_data={k: True for k in ["x", "y", "Eg", "score"] if k in plot_df.columns},
                height=600
            )
            fig3d.update_layout(template="simple_white")
            st.plotly_chart(fig3d, use_container_width=True)
        except Exception as e:
            st.error(f"3D plot error: {e}")

# ─── Download Tab ────────────────────────────────────────────────────────────
with tab_dl:
    csv = df.to_csv(index=False).encode()
    st.download_button("📥 Download CSV", csv, "EnerMat_results.csv", "text/csv")

    # Safely get top candidate
    top = df.iloc[0]

    if mode == "Binary A–B":
        top_label = getattr(top, "formula", f"{A}-{B}")
    else:
        # Safe retrieval of x and y, even if columns are missing
        if isinstance(top, pd.Series):
            x_val = top["x"] if "x" in top else 0.0
            y_val = top["y"] if "y" in top else 0.0
        else:
            x_val = getattr(top, "x", 0.0)
            y_val = getattr(top, "y", 0.0)
        top_label = f"{A}-{B}-{C} x={x_val:.2f} y={y_val:.2f}"

    # Compose report text
    txt = f"""
EnerMat report ({datetime.date.today()})
Top candidate : {top_label}
Band-gap     : {top.Eg}
Stability    : {getattr(top, 'stability', 'N/A')}
Score        : {top.score}
"""
    st.download_button("📄 Download TXT", txt, "EnerMat_report.txt", "text/plain")

    # DOCX report
    doc = Document()
    doc.add_heading("EnerMat Report", 0)
    doc.add_paragraph(f"Date: {datetime.date.today()}")
    doc.add_paragraph(f"Top candidate: {top_label}")
    tbl = doc.add_table(rows=1, cols=2)
    rows = [("Band-gap", top.Eg), ("Score", top.score)]
    if hasattr(top, 'stability'):
        rows.insert(1, ("Stability", top.stability))
    for k, v in rows:
        row = tbl.add_row()
        row.cells[0].text = k
        row.cells[1].text = str(v)
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    st.download_button(
        "📝 Download DOCX", buf, "EnerMat_report.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown(
    "<hr style='margin-top:2em;margin-bottom:0.5em;'>"
    "<div style='text-align:center; font-size:0.85em;'>"
    "EnerMat Explorer © 2025 Dr. Gbadebo Taofeek Yusuf · "
    "Built with ❤️ using Streamlit + Materials Project API"
    "</div>",
    unsafe_allow_html=True
)
