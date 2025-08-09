import datetime
import io

import streamlit as st
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from docx import Document

from backend.perovskite_utils import (
    screen_binary,
    screen_ternary,
    END_MEMBERS,
    APPLICATION_CONFIG,   # ← import presets to override slider when applicable
)

# ─────────── STREAMLIT PAGE CONFIG ───────────
st.set_page_config("EnerMat Explorer – Lead-Free Perovskite PV Discovery Tool", layout="wide")
st.title("☀️ EnerMat Explorer | Lead-Free Perovskite PV Discovery Tool")
st.markdown(
    """
    <style>
      .css-1d391kg { border-right: 3px solid #0D47A1 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    application = st.selectbox(
        "Select application",
        ["single", "tandem", "indoor", "detector"],
        help="Preset band-gap preference and scoring. Score is normalized so the top candidate = 1.0 for the current settings."
    )

    st.header("Environment")
    rh = st.slider("Humidity [%]", 0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)

    st.header("Target band-gap [eV]")
    bg_lo, bg_hi = st.slider("Gap window (shown on plot)", 0.50, 3.00, (1.00, 1.40), 0.01)

    st.header("Model settings")
    bow = st.number_input("Bowing (eV, negative ⇒ gap↑)", -1.0, 1.0, -0.15, 0.05)
    dx = st.number_input("x-step", 0.01, 0.50, 0.05, 0.01)
    if mode.startswith("Ternary"):
        dy = st.number_input("y-step", 0.01, 0.50, 0.05, 0.01)

    z = st.slider("Ge fraction z", 0.00, 0.80, 0.10, 0.05, help="B-site Ge²⁺ in CsSn₁₋zGezX₃")

    if st.button("🗑 Clear history"):
        st.session_state.history = []
        st.rerun()

# Small note on scoring (keeps users oriented)
st.info("Score = SQ(Eg) × exp(−Ehull/kT) × exp(ΔEox/kTeff) × tolerance. "
        "Scores are normalized so the best entry under the current settings is 1.0. "
        "Application presets fix the target Eg preference (Single = 1.10–1.40 eV).")

# ─────────── CACHE WRAPPERS ───────────
@st.cache_data(show_spinner="⏳ Screening …", max_entries=20)
def _run_binary(*args, **kwargs):
    return screen_binary(*args, **kwargs)

@st.cache_data(show_spinner="⏳ Screening …", max_entries=10)
def _run_ternary(*args, **kwargs):
    return screen_ternary(*args, **kwargs)

# ─────────── RUNNING SCREEN ───────────
col_run, col_prev = st.columns([3, 1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_prev = col_prev.button("⏪ Previous", disabled=not st.session_state.history)

if do_prev:
    st.session_state.history.pop()
    if not st.session_state.history:
        st.stop()
    st.success("Showing previous result")

elif do_run:
    # sanity-check
    for f in ([A, B] if mode.startswith("Binary") else [A, B, C]):
        if f not in END_MEMBERS:
            st.error(f"❌ Unknown end-member: {f}")
            st.stop()

    # Use application preset range (overrides slider) for consistent scoring/plot shading
    plot_lo, plot_hi = bg_lo, bg_hi
    if application in APPLICATION_CONFIG and APPLICATION_CONFIG[application]["range"]:
        bg_lo, bg_hi = APPLICATION_CONFIG[application]["range"]  # scoring window
    # else keep user slider values

    if mode.startswith("Binary"):
        df = _run_binary(A, B, rh, temp, (bg_lo, bg_hi), bow, dx, z=z, application=application)
    else:
        df = _run_ternary(A, B, C, rh, temp, (bg_lo, bg_hi), {"AB": bow, "AC": bow, "BC": bow},
                          dx=dx, dy=dy, z=z, application=application)

    if df.empty:
        st.error("No data returned for the chosen inputs.")
        st.stop()

    st.session_state.history.append({
        "mode": mode,
        "df": df,
        "params": {"A": A, "B": B, **({"C": C} if mode.startswith("Ternary") else {}),
                   "application": application, "bow": bow, "dx": dx, **({"dy": dy} if mode.startswith("Ternary") else {}),
                   "z": z, "scoring_range": (bg_lo, bg_hi), "plot_range": (plot_lo, plot_hi)}
    })

if not st.session_state.history:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

# ─────────── DISPLAY RESULTS ───────────
h = st.session_state.history[-1]
df = h["df"]
mode = h["mode"]
bg_lo, bg_hi = h["params"]["plot_range"]

tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table", "📈 Plot", "📥 Download"])

with tab_tbl:
    st.dataframe(df, use_container_width=True, height=440)

with tab_plot:
    if mode.startswith("Binary") and {"Ehull", "Eg", "score"}.issubset(df.columns):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Ehull"], y=df["Eg"], mode="markers",
            marker=dict(
                size=8 + 12 * df["score"], color=df["score"],
                colorscale="Viridis", cmin=0, cmax=1,
                colorbar=dict(title="Score"), line=dict(width=0.5, color="black")
            ),
            hovertemplate="<b>%{customdata[0]}</b><br>Eg=%{y:.3f} eV<br>Ehull=%{x:.4f} eV/at<br>"
                          "Raw=%{customdata[1]:.3e}<br>Score=%{marker.color:.3f}<br>"
                          "PCE_max=%{customdata[2]:.1f}%<extra></extra>",
            customdata=pd.DataFrame({
                "formula": df["formula"], "raw": df["raw_score"], "pce": df["PCE_max (%)"]
            }).to_numpy()
        ))
        # shaded Eg window on plot only
        if bg_lo is not None and bg_hi is not None:
            fig.add_shape(type="rect", x0=0, x1=0.05, y0=bg_lo, y1=bg_hi,
                          line=dict(color="LightSeaGreen", dash="dash"),
                          fillcolor="LightSeaGreen", opacity=0.1)
        fig.update_layout(
            title=f"EnerMat Binary Screen – {h['params']['application']}",
            xaxis_title="Ehull (eV/atom)", yaxis_title="Eg (eV)",
            template="simple_white", width=720, height=540,
            margin=dict(l=60, r=60, t=60, b=60)
        )
        st.plotly_chart(fig, use_container_width=True)

    elif mode.startswith("Ternary") and {"x", "y", "score"}.issubset(df.columns):
        fig = px.scatter_3d(df, x="x", y="y", z="score", color="score",
                            color_continuous_scale="Viridis",
                            labels={"x": "B2 fraction", "y": "B3 fraction"}, height=500)
        st.plotly_chart(fig, use_container_width=True)

with tab_dl:
    st.download_button("📥 Download current CSV",
                       df.to_csv(index=False).encode(),
                       "EnerMat_results.csv", "text/csv")

    # Simple TXT + DOCX (top row)
    _top = df.iloc[0]
    label = str(_top["formula"]) if len(df) == 1 else f"{_top['formula']} (x={_top.get('x','')}, z={_top.get('z','')})"
    _txt = (
        "EnerMat auto-report  "
        f"{datetime.date.today()}\n"
        f"Top candidate   : {label}\n"
        f"Band-gap [eV]   : {_top['Eg']}\n"
        f"Ehull [eV/at.]  : {_top['Ehull']}\n"
        f"Eox_e [eV/e⁻]   : {_top.get('Eox_e', 'N/A')}\n"
        f"PCE_max [%]     : {_top.get('PCE_max (%)', 'N/A')}\n"
        f"Raw score       : {_top.get('raw_score', 'N/A')}\n"
        f"Score (norm)    : {_top['score']}\n"
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
    for k in ("Eg", "Ehull", "Eox_e", "PCE_max (%)", "raw_score", "score"):
        if k in _top:
            row = table.add_row().cells
            row[0].text, row[1].text = k, str(_top[k])

    buf = io.BytesIO()
    _doc.save(buf)
    buf.seek(0)
    st.download_button("📝 Download DOCX", buf,
                       "EnerMat_report.docx",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
