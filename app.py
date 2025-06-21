import io
import os
import datetime
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from docx import Document
from mp_api.client import MPRester

from backend.perovskite_utils import (
    mix_abx3 as screen,
    screen_ternary,
    END_MEMBERS,
    fetch_mp_data as _summary
)

# ───────────────────────────────────── App Config ─────────────────────────────
st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.6")

# ─────────────────────────────────── Session History ──────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ───────────────────────────────────── Sidebar ────────────────────────────────
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose screening type", ["Binary A–B", "Ternary A–B–C"], key="mode")

    st.header("End-members")
    preset_A = st.selectbox("Preset A", END_MEMBERS, index=0, key="preset_A")
    preset_B = st.selectbox("Preset B", END_MEMBERS, index=1, key="preset_B")
    custom_A = st.text_input("Custom A (optional)", "", key="custom_A").strip()
    custom_B = st.text_input("Custom B (optional)", "", key="custom_B").strip()
    A = custom_A or preset_A
    B = custom_B or preset_B

    C = None
    if mode == "Ternary A–B–C":
        preset_C = st.selectbox("Preset C", END_MEMBERS, index=2, key="preset_C")
        custom_C = st.text_input("Custom C (optional)", "", key="custom_C").strip()
        C = custom_C or preset_C

    st.header("Environment")
    rh = st.slider("Humidity [%]", 0, 100, 50, key="rh")
    temp = st.slider("Temperature [°C]", -20, 100, 25, key="temp")

    st.header("Target gap [eV]")
    bg_lo, bg_hi = st.slider("Gap window [eV]", 0.5, 3.0, (1.0, 1.4), 0.01, key="bg")

    st.header("Model knobs")
    bow = st.number_input("Bowing [eV]", 0.0, 1.0, 0.30, 0.05, key="bow")
    dx = st.number_input("x-step", 0.01, 0.50, 0.05, 0.01, key="dx")
    dy = None
    if mode == "Ternary A–B–C":
        dy = st.number_input("y-step", 0.01, 0.50, 0.05, 0.01, key="dy")

    if st.button("🗑 Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()

    st.caption("© 2025 Dr Gbadebo Taofeek Yusuf")
    GIT_SHA = st.secrets.get("GIT_SHA", "dev")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    st.caption(f"⚙️ Version: `{GIT_SHA}` • ⏱ {ts}")

# ────────────────────────────────── Backend Calls ─────────────────────────────
@st.cache_data(show_spinner="⏳ Screening …")
def run_screen(*, A: str, B: str, rh: float, temp: float,
               bg: tuple[float,float], bow: float, dx: float) -> pd.DataFrame:
    return screen(formula_A=A, formula_B=B,
                  rh=rh, temp=temp,
                  bg_window=bg, bowing=bow, dx=dx)

@st.cache_data(show_spinner="⏳ Screening …")
def run_ternary(*, A: str, B: str, C: str, rh: float, temp: float,
                bg: tuple[float,float], bows: dict[str,float],
                dx: float, dy: float, n_mc: int=200) -> pd.DataFrame:
    return screen_ternary(A=A, B=B, C=C,
                          rh=rh, temp=temp,
                          bg=bg, bows=bows,
                          dx=dx, dy=dy, n_mc=n_mc)

# ───────────────────────────────── Run / Back Logic ───────────────────────────
col_run, col_back = st.columns([3, 1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=len(st.session_state.history) < 1)

if do_back and st.session_state.history:
    params = st.session_state.history.pop()
    # restore everything
    mode    = params["mode"]
    A, B, C = params["A"], params["B"], params.get("C")
    rh, temp = params["rh"], params["temp"]
    bg_lo, bg_hi = params["bg_lo"], params["bg_hi"]
    bow, dx, dy = params["bow"], params["dx"], params.get("dy")
    df      = params["df"]
    docA, docB = params["docA"], params["docB"]
    st.success("▶ Restored previous run")

elif do_run:
    # fetch end-member data
    try:
        docA = _summary(A, ["band_gap","energy_above_hull"])
        docB = _summary(B, ["band_gap","energy_above_hull"])
    except Exception as e:
        st.error(f"❌ MP query failed: {e}")
        st.stop()
    if not docA or not docB:
        st.error("❌ Invalid formula(s)")
        st.stop()

    # run the appropriate screening
    if mode == "Binary A–B":
        df = run_screen(A=A, B=B, rh=rh, temp=temp,
                        bg=(bg_lo,bg_hi), bow=bow, dx=dx)
    else:
        df = run_ternary(A=A, B=B, C=C,
                         rh=rh, temp=temp,
                         bg=(bg_lo,bg_hi),
                         bows={"AB":bow,"AC":bow,"BC":bow},
                         dx=dx, dy=dy, n_mc=200)

    if df.empty:
        st.error("No candidates found – widen your window or steps.")
        st.stop()

    # unify column names
    df = df.rename(columns={"Eg":"band_gap"})

    # store for “Previous”
    st.session_state.history.append({
        "mode": mode, "A": A, "B": B, "C": C,
        "rh": rh, "temp": temp,
        "bg_lo": bg_lo, "bg_hi": bg_hi,
        "bow": bow, "dx": dx, "dy": dy,
        "docA": docA, "docB": docB,
        "df": df
    })

elif st.session_state.history:
    last = st.session_state.history[-1]
    df, docA, docB = last["df"], last["docA"], last["docB"]
else:
    st.info("▶ Click **Run screening** to begin.")
    st.stop()

# ─────────────────────────────────── Tabs ─────────────────────────────────────
tab_tbl, tab_plot, tab_dl, tab_bench, tab_results = st.tabs([
    "📊 Table", "📈 Plot", "📥 Download", "⚖ Benchmark", "📑 Results Summary"
])

# ─────────────────────────────── Table Tab ──────────────────────────────────
with tab_tbl:
    params = pd.DataFrame({
        "Parameter": [
            "Humidity [%]", "Temperature [°C]",
            "Gap window [eV]", "Bowing [eV]", "x-step"
        ] + (["y-step"] if mode=="Ternary A–B–C" else []),
        "Value": [
            rh, temp,
            f"{bg_lo:.2f}–{bg_hi:.2f}", bow, dx
        ] + ([dy] if mode=="Ternary A–B–C" else [])
    })
    st.markdown("**Run parameters**")
    st.table(params)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**A-endmember: {A}**")
        st.write(f"Band-gap: {docA['band_gap']:.2f} eV")
        st.write(f"E_above_hull: {docA['energy_above_hull']:.3f} eV/atom")
    with c2:
        st.markdown(f"**B-endmember: {B}**")
        st.write(f"Band-gap: {docB['band_gap']:.2f} eV")
        st.write(f"E_above_hull: {docB['energy_above_hull']:.3f} eV/atom")

    st.dataframe(df, use_container_width=True, height=400)

# ─────────────────────────────── Plot Tab ───────────────────────────────────
with tab_plot:
    if mode == "Binary A–B":
        st.caption("ℹ️ Hover for details; zoom & pan")
        top_cut = df.score.quantile(0.8)
        df["is_top"] = df.score >= top_cut
        fig = px.scatter(
            df, x="stability", y="band_gap",
            color="score", color_continuous_scale="plasma",
            hover_data=["formula","x","band_gap","stability","score"],
            height=450
        )
        fig.add_trace(go.Scatter(
            x=df.loc[df.is_top,"stability"],
            y=df.loc[df.is_top,"band_gap"],
            mode="markers",
            marker=dict(size=22,color="rgba(0,0,0,0)",line=dict(width=2,color="black")),
            hoverinfo="skip", showlegend=False
        ))
        fig.update_layout(template="simple_white")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.caption("ℹ️ Hover for details; rotate & zoom")
        fig3d = px.scatter_3d(
            df, x="x", y="y", z="score",
            color="score",
            hover_data=["x","y","band_gap","score"],
            height=600
        )
        fig3d.update_layout(template="simple_white")
        st.plotly_chart(fig3d, use_container_width=True)

# ────────────────────────────── Download Tab ─────────────────────────────────
with tab_dl:
    csv = df.to_csv(index=False).encode()
    st.download_button("CSV", csv, "EnerMat_results.csv", "text/csv")
    # … your existing TXT & DOCX buttons …

# ─────────────────────────── Benchmark Tab ─────────────────────────────────
with tab_bench:
    # … your existing benchmark code …

# ─────────────────────── Results Summary Tab ───────────────────────────────
with tab_results:
    # … your existing results-summary code …
