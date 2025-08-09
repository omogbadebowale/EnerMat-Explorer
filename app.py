# ===============================
# File: app.py
# ===============================
import datetime
import io
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from docx import Document

from backend.perovskite_utils import (
    screen_binary,
    screen_ternary,
    END_MEMBERS,
)

# ─────────── STREAMLIT PAGE CONFIG ───────────
st.set_page_config("EnerMat Explorer – Lead-Free Perovskite PV Discovery Tool", layout="wide")
st.title("☀️ EnerMat Explorer | Lead-Free Perovskite PV Discovery Tool (Physics-Tightened)")

# ─────────── SESSION STATE ───────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────── SIDEBAR ───────────
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose screening type", ["Binary A–B", "Ternary A–B–C"]) 

    st.header("End-members")
    preset_A = st.selectbox("Preset A", END_MEMBERS, 0)
    preset_B = st.selectbox("Preset B", END_MEMBERS, 1)
    custom_A = st.text_input("Custom A (optional)").strip()
    custom_B = st.text_input("Custom B (optional)").strip()
    A = custom_A or preset_A
    B = custom_B or preset_B
    if mode.startswith("Ternary"):
        preset_C = st.selectbox("Preset C", END_MEMBERS, 2)
        custom_C = st.text_input("Custom C (optional)").strip()
        C = custom_C or preset_C

    st.header("Application")
    application = st.selectbox("Select application", ["single", "tandem", "indoor", "detector", "(manual)"])
    manual_eg = application == "(manual)"

    st.header("Environment")
    rh = st.slider("Humidity [%]", 0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)

    st.header("Target band-gap [eV]")
    bg_lo, bg_hi = st.slider("Gap window", 0.50, 3.00, (1.00, 1.40), 0.01)

    st.header("Model settings")
    bow = st.number_input("Bowing (eV, negative ⇒ gap↑)", -1.0, 1.0, -0.15, 0.05)
    dx = st.number_input("x-step", 0.01, 0.50, 0.05, 0.01)
    if mode.startswith("Ternary"):
        dy = st.number_input("y-step", 0.01, 0.50, 0.05, 0.01)

    z = st.slider("Ge fraction z", 0.00, 0.80, 0.10, 0.05, help="B-site Ge²⁺ in CsSn₁₋zGe_zX₃")

    # Structural penalty sharpness
    beta_struct = st.slider("Structural penalty β", 0.1, 3.0, 1.0, 0.1)

    # Clear history
    if st.button("🗑 Clear history"):
        st.session_state.history = []
        st.rerun()

# ─────────── INFO BOXES ───────────
st.info(
    "This version couples Temperature to the thermodynamic penalty (kT), applies Humidity to the oxidation enthalpy (ΔEₒₓ → ΔEₒₓ − λ·RH), and uses composition-weighted radii for the tolerance factor. Choose ‘(manual)’ to fully override application gap presets.")

# ─────────── RUN ───────────
col_run, col_prev = st.columns([3, 1])
do_run = col_run.button("▶ Run screening", type="primary")
do_prev = col_prev.button("⏪ Previous", disabled=not st.session_state.history)

if do_prev:
    st.session_state.history.pop()
    prev = st.session_state.history[-1]
    df, mode = prev["df"], prev["mode"]
    st.success("Showing previous result")

elif do_run:
    # sanity-check: restrict to known end-members unless you extend backend to parse more
    for f in ([A, B] if mode.startswith("Binary") else [A, B, C]):
        if f not in END_MEMBERS:
            st.error(f"❌ Unknown end-member: {f}")
            st.stop()

    if mode.startswith("Binary"):
        df = screen_binary(
            A, B, rh, temp, (bg_lo, bg_hi), bow, dx,
            z=z,
            application=None if manual_eg else application,
            manual_eg=manual_eg,
            beta_struct=beta_struct,
        )
    else:
        df = screen_ternary(
            A, B, C, rh, temp, (bg_lo, bg_hi), {"AB": bow, "AC": bow, "BC": bow},
            dx=dx, dy=dy, z=z,
            application=None if manual_eg else application,
            manual_eg=manual_eg,
            beta_struct=beta_struct,
        )
    st.session_state.history.append({"mode": mode, "df": df})

elif not st.session_state.history:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

# ─────────── DISPLAY ───────────
df = st.session_state.history[-1]["df"]
mode = st.session_state.history[-1]["mode"]

# Tabs: results, plot, provenance, download

with st.tabs(["📊 Table", "📈 Plot", "ℹ️ Provenance", "📥 Download"]) as (tab_tbl, tab_plot, tab_prov, tab_dl):
    with tab_tbl:
        st.dataframe(df, use_container_width=True, height=500)

    with tab_plot:
        if mode.startswith("Binary") and {"Ehull", "Eg"}.issubset(df.columns):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["Ehull"], y=df["Eg"], mode="markers",
                marker=dict(
                    size=8 + 12 * df["score"], color=df["score"],
                    colorscale="Viridis", cmin=0, cmax=1,
                    colorbar=dict(title="Score"), line=dict(width=0.5, color="black")
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Eg=%{y:.3f} eV\nEhull=%{x:.4f} eV/at\n"
                    "Score=%{marker.color:.3f}\n"
                    "Eox (env)=%{customdata[1]:.3f} eV\n"
                    "t=%{customdata[2]:.3f}\nPCEmax=%{customdata[3]:.1f}%<extra></extra>"
                ),
                customdata=df[["formula", "Eox_env", "t_factor", "PCE_max (%)"]].values
            ))
            fig.add_shape(type="rect", x0=0, x1=0.05, y0=bg_lo, y1=bg_hi,
                          line=dict(color="LightSeaGreen", dash="dash"), fillcolor="LightSeaGreen", opacity=0.1)
            fig.update_layout(
                title="EnerMat Binary Screen",
                xaxis_title="Ehull (eV/atom)", yaxis_title="Eg (eV)", template="simple_white",
                font=dict(family="Arial", size=12, color="black"), width=720, height=540,
                margin=dict(l=60, r=60, t=60, b=60),
                coloraxis_colorbar=dict(title=dict(text="Score", font=dict(size=12)), tickfont=dict(size=12))
            )
            st.plotly_chart(fig, use_container_width=True)
        elif mode.startswith("Ternary") and {"x", "y", "score"}.issubset(df.columns):
            fig = px.scatter_3d(df, x="x", y="y", z="score", color="score",
                                color_continuous_scale="Viridis",
                                labels={"x": "B2 fraction", "y": "B3 fraction"}, height=500)
            st.plotly_chart(fig, use_container_width=True)

    with tab_prov:
        st.subheader("Data provenance & calibration")
        # Condense unique provenance rows
        prov_cols = [c for c in ["A_mpid", "B_mpid", "C_mpid", "A_gap_route", "B_gap_route", "C_gap_route", "T_K", "RH_%"] if c in df.columns]
        if prov_cols:
            st.dataframe(df[prov_cols].drop_duplicates(), use_container_width=True)
        st.caption("Gap routes: 'calibrated' uses curated experimental gaps if provided; 'offset+X' applies a halide-specific offset (I/Br/Cl) aligned with manuscript.")

    with tab_dl:
        st.download_button("📥 Download CSV", df.to_csv(index=False).encode(), "EnerMat_results.csv", "text/csv")

        # Auto-report (TXT / DOCX)
        _top = df.iloc[0]
        formula = str(_top["formula"])
        coords = ", ".join(
            f"{c}={_top[c]:.2f}" for c in ("x", "y", "z") if c in _top and pd.notna(_top[c])
        )
        label = formula if len(df) == 1 else f"{formula} ({coords})"

        _txt = (
            "EnerMat auto-report  "
            f"{datetime.date.today()}\n"
            f"Top candidate   : {label}\n"
            f"Band-gap [eV]   : {_top['Eg']}\n"
            f"Ehull [eV/at.]  : {_top['Ehull']}\n"
            f"Eox [eV per Sn] : {_top.get('Eox', 'N/A')}\n"
            f"Eox_env [eV/Sn] : {_top.get('Eox_env', 'N/A')}\n"
            f"t_factor        : {_top.get('t_factor', 'N/A')}\n"
            f"PCE_max [%]     : {_top.get('PCE_max (%)', 'N/A')}\n"
            f"Score           : {_top['score']}\n"
        )
        st.download_button("📄 Download TXT", _txt, "EnerMat_report.txt", mime="text/plain")

        _doc = Document()
        _doc.add_heading("EnerMat Report", level=0)
        _doc.add_paragraph(f"Date : {datetime.date.today()}")
        _doc.add_paragraph(f"Top candidate : {label}")
        table = _doc.add_table(rows=1, cols=2)
        table.style = "LightShading-Accent1"
        hdr = table.rows[0].cells
        hdr[0].text, hdr[1].text = "Property", "Value"
        for k in ("Eg", "Ehull", "Eox", "Eox_env", "t_factor", "PCE_max (%)", "score"):
            if k in _top:
                row = table.add_row().cells
                row[0].text, row[1].text = k, str(_top[k])
        buf = io.BytesIO()
        _doc.save(buf)
        buf.seek(0)
        st.download_button("📝 Download DOCX", buf, "EnerMat_report.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# (Optional) Session export/import
with st.expander("💾 Session export/import"):
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Export last result as JSON"):
            st.download_button(
                "Download JSON",
                json.dumps(st.session_state.history[-1]["df"].to_dict(orient="records"), indent=2).encode(),
                file_name="EnerMat_results.json",
                mime="application/json",
            )
    with c2:
        up = st.file_uploader("Import results JSON", type=["json"], accept_multiple_files=False)
        if up:
            recs = json.load(up)
            df_imp = pd.DataFrame.from_records(recs)
            st.session_state.history.append({"mode": "Imported", "df": df_imp})
            st.success("Imported. Switch to tabs above to view.")
