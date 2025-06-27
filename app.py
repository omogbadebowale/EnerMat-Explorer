EnerMat Perovskite Explorer – Streamlit app (v9.6)
Updated plotting aesthetics for journal‑quality figures
"""

import io
import os
import datetime
from pathlib import Path

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
@st.cache_data(show_spinner="⏳ Running screening…", max_entries=20)
def run_screen(formula_A, formula_B, rh, temp, bg_window, bowing, dx):
    return screen(
        formula_A=formula_A,
        formula_B=formula_B,
        rh=rh,
        temp=temp,
        bg_window=bg_window,
        bowing=bowing,
        dx=dx,
    )

# ─── Execution Control ────────────────────────────────────────────────────────
col_run, col_back = st.columns([3, 1])
do_run = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=not st.session_state.history)

if do_back:
    st.session_state.history.pop()
    prev = st.session_state.history[-1]
    mode = prev["mode"]
    A, B, rh, temp = prev["A"], prev["B"], prev["rh"], prev["temp"]
    bg_lo, bg_hi = prev["bg"]
    bow, dx = prev["bow"], prev["dx"]
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
        )
    else:
        try:
            df = screen_ternary(
                A=A,
                B=B,
                C=C,
                rh=rh,
                temp=temp,
                bg=(bg_lo, bg_hi),
                bows={"AB": bow, "AC": bow, "BC": bow},
                dx=dx,
                dy=dy,
                n_mc=200,
            )
        except Exception as e:
            st.error(f"❌ Ternary error: {e}")
            st.stop()

    df = df.rename(columns={
        "energy_above_hull": "stability",
        "band_gap": "Eg",
    })

    entry = {
        "mode": mode,
        "A": A,
        "B": B,
        "rh": rh,
        "temp": temp,
        "bg": (bg_lo, bg_hi),
        "bow": bow,
        "dx": dx,
        "df": df,
    }
    if mode == "Ternary A–B–C":
        entry["C"] = C
        entry["dy"] = dy
    st.session_state.history.append(entry)

elif st.session_state.history:
    prev = st.session_state.history[-1]
    mode = prev["mode"]
    A, B, rh, temp = prev["A"], prev["B"], prev["rh"], prev["temp"]
    bg_lo, bg_hi = prev["bg"]
    bow, dx = prev["bow"], prev["dx"]
    if mode == "Ternary A–B–C":
        C, dy = prev["C"], prev["dy"]
    df = prev["df"]

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
        "Value": [rh, temp, f"{bg_lo:.2f}–{bg_hi:.2f}", bow, dx],
    }
    if mode == "Ternary A–B–C":
        param_data["Parameter"].append("y-step")
        param_data["Value"].append(dy)

    st.table(pd.DataFrame(param_data))

    st.subheader("Candidate Results")
    st.dataframe(df, use_container_width=True, height=400)

# ─── Plot Tab ────────────────────────────────────────────────────────────────
with tab_plot:
    # Figure-wide styling constants for journal‑ready plots
    FIG_KW = dict(width=1200, height=800, template="simple_white")
    FONT_KW = dict(family="Times New Roman", size=18, color="#000")
    MARKER_LINE = dict(width=1.2, color="#000")  # crisp, high‑contrast edge
    C_SCALE = "Viridis"  # perceptually uniform & colour‑blind friendly

    if mode == "Binary A–B":
        required = {"stability", "Eg", "score"}
        if not required.issubset(df.columns):
            st.error("❌ Missing required columns: " + ", ".join(required - set(df.columns)))
            st.stop()

        plot_df = df.dropna(subset=required)

        fig = px.scatter(
            plot_df,
            x="stability",
            y="Eg",
            color="score",
            color_continuous_scale=C_SCALE,
            hover_data=["formula", "x", "Eg", "stability", "score"],
            **FIG_KW,
        )
        fig.update_traces(marker=dict(size=11, opacity=0.9, line=MARKER_LINE))

        # Highlight the top 20 % of candidates
        q80 = plot_df["score"].quantile(0.80)
        top = plot_df.query("score >= @q80")
        fig.add_scatter(
            x=top["stability"],
            y=top["Eg"],
            mode="markers",
            marker=dict(size=18, symbol="circle-open", line=dict(width=2), color="black"),
            hoverinfo="skip",
            showlegend=False,
        )

        fig.update_layout(
            font=FONT_KW,
            xaxis_title="Stability (eV atom⁻¹)",
            yaxis_title="Band gap / eV",
            coloraxis_colorbar=dict(
                title="Screening score",
                title_side="right",
                titlefont_size=18,
                tickfont_size=16,
                len=0.70,
                thickness=22,
                outlinewidth=1,
            ),
            margin=dict(l=90, r=40, t=30, b=80)
        )

        st.plotly_chart(fig, use_container_width=True)

    else:  # ── Ternary ───────────────────────────────────────────────────────
        required = {"x", "y", "score"}
        if not required.issubset(df.columns):
            st.warning("❗ Not enough columns for ternary 3‑D plot.")
            st.stop()

        plot_df = df.dropna(subset=required)

        fig3d = px.scatter_3d(
            plot_df,
            x="x", y="y", z="score",
            color="score",
            color_continuous_scale=C_SCALE,
            hover_data=["x", "y", "Eg", "score"],
            width=1200,
            height=900,
            template="simple_white",
        )
        fig3d.update_traces(marker=dict(size=5, opacity=0.9, line=MARKER_LINE))

        fig3d.update_layout(
            font=FONT_KW,
            scene=dict(
                xaxis_title="A fraction",
                yaxis_title="B fraction",
                zaxis_title="Screening score",
                xaxis=dict(showspikes=False),
                yaxis=dict(showspikes=False),
                zaxis=dict(showspikes=False),
            ),
            coloraxis_colorbar=dict(
                title="Score",
                title_side="right",
                titlefont_size=16,
                tickfont_size=14,
                len=0.60,
                thickness=20,
                outlinewidth=1,
            ),
            margin=dict(l=60, r=60, t=30, b=60),
        )

        st.plotly_chart(fig3d, use_container_width=True)

