import datetime
import io
from pathlib import Path
import base64

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
st.title("☀️ EnerMat Explorer | Lead-Free Perovskite PV Discovery Tool")
st.markdown(
    """
    <style>
      /* Target the sidebar wrapper and give it a colored left border */
      .css-1d391kg {
        border-right: 3px solid #0D47A1 !important;
      }
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
    mode = st.radio(
        "Choose screening type",
        ["Binary A–B", "Ternary A–B–C"]
    )

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
        ["single", "tandem", "indoor", "detector"]
    )

    st.header("Environment")
    rh = st.slider("Humidity [%]", 0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)

    st.header("Target band-gap [eV]")
    bg_lo, bg_hi = st.slider(
        "Gap window", 0.50, 3.00, (1.00, 1.40), 0.01
    )

    st.header("Model settings")
    bow = st.number_input(
        "Bowing (eV, negative ⇒ gap↑)",
        -1.0, 1.0, -0.15, 0.05
    )
    dx = st.number_input("x-step", 0.01, 0.50, 0.05, 0.01)
    if mode.startswith("Ternary"):
        dy = st.number_input("y-step", 0.01, 0.50, 0.05, 0.01)

    z = st.slider(
        "Ge fraction z", 0.00, 0.80, 0.10, 0.05,
        help="B-site Ge²⁺ in CsSn₁₋zGeₓX₃"
    )

   # ── Clear history button ──
    if st.button("🗑 Clear history"):
        # Safely clear
        if "history" in st.session_state:
            st.session_state.history = []
        # Re-run with clean state
        st.rerun()

    # ── Developer credit in sidebar footer ──
    st.markdown(
        """
        <div style="font-size:0.8rem; color:grey; margin-top:2rem;">
          Developed by <strong>Dr. Gbadebo Taofeek Yusuf</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(f"⚙️ Build SHA : dev • 🕒 {datetime.datetime.now():%Y-%m-%d %H:%M}")


# ─────────── CACHE WRAPPERS ───────────
@st.cache_data(show_spinner="⏳ Screening …", max_entries=20)
def _run_binary(*args, **kwargs):
    return screen_binary(*args, **kwargs)

@st.cache_data(show_spinner="⏳ Screening …", max_entries=10)
def _run_ternary(*args, **kwargs):
    return screen_ternary(*args, **kwargs)

# ─────────── OVERVIEW & RESEARCH OPPORTUNITIES ───────────
st.markdown(
    """
    <style>
      .overview-box {
        background-color: #ffffff;       /* white for max contrast */
        border: 1px solid #dddddd;       /* light grey border */
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 32px;
        color: #333333;
        font-family: Arial, sans-serif;
      }
      .overview-box h2 {
        margin-top: 0;
        color: #005FAD;                  /* deep brand-blue */
        font-size: 1.8rem;
      }
      .overview-box p {
        font-size: 1rem;
        line-height: 1.5;
        margin-bottom: 16px;
      }
      .overview-box ul {
        margin: 0 0 16px 1.2em;
        font-size: 0.95rem;
        line-height: 1.4;
      }
      .overview-box ul li {
        margin-bottom: 8px;
      }
    </style>

    <div class="overview-box">
      <h2>Context &amp; Scientific Justification</h2>
      <p>
        Lead–halide perovskites deliver record solar efficiencies but suffer from environmental toxicity and rapid degradation under heat, moisture, or oxygen.
        Tin-based, lead-free analogues offer a safer path, yet optimising their key metrics remains a major hurdle:
      </p>
      <ul>
        <li><strong>Eg</strong> (band gap): ideal ≈ 1.3 eV for single-junction PV absorption.</li>
        <li><strong>E<sub>hull</sub></strong> (phase stability): &lt; 0.05 eV / atom ⇒ likely synthesizable.</li>
        <li><strong>ΔE<sub>ox</sub></strong> (oxidation resistance): positive values resist Sn²⁺ → Sn⁴⁺.</li>
        <li><strong>PCE<sub>max</sub></strong> (Shockley–Queisser limit): theoretical upper bound on efficiency.</li>
      </ul>
      <p>
        <em>EnerMat Explorer</em> translates these metrics into an interactive, high-throughput screening platform:
        instantly ranking compositions by PCE<sub>max</sub>, phase stability and oxidation resistance—no DFT runs required.
      </p>

      <h2>Halides Covered &amp; Research Pathways</h2>
      <ul>
        <li><strong>Tin-based perovskites:</strong> CsSnX₃, MASnX₃, FASnX₃ (X = I, Br, Cl).</li>
        <li><strong>Germanium analogues:</strong> CsGeBr₃, CsGeCl₃.</li>
        <li><strong>Vacancy-ordered phases:</strong> Cs₂SnI₆.</li>
        <li><strong>Layered Bi/Sb phases:</strong> Cs₃Bi₂Br₉, Cs₃Sb₂I₉.</li>
        <li><strong>Double perovskites:</strong> Cs₂AgBiBr₆, Cs₂AgInCl₆.</li>
      </ul>
      <p>
        <strong>Explore opportunities:</strong><br>
        • Map how A-site mixing (Cs⁺ vs. MA⁺/FA⁺) shifts Eg & stability.<br>
        • Screen Bi/Sb phases for moisture and carrier-lifetime enhancements.<br>
        • Investigate vacancy ordering to boost oxidation resilience.<br>
        • Tune double-perovskite alloys for next-gen tandem cells.<br>
        • Visualize humidity/temperature effects on E<sub>hull</sub> in real time.
      </p>
      <p>
        Whether you’re a student learning perovskite design or an engineer scouting new compositions, EnerMat Explorer turns weeks of trial-and-error into seconds of interactive insight.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────── HOW TO USE ───────────
st.markdown(
    """
    <style>
      .usage-box {
        background-color: #ffffff;
        border: 1px solid #dddddd;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 32px;
        color: #333333;
        font-family: Arial, sans-serif;
      }
      .usage-box h2 {
        margin-top: 0;
        color: #005FAD;
        font-size: 1.6rem;
      }
      .usage-box ul {
        margin: 0 0 16px 1.2em;
        font-size: 0.95rem;
        line-height: 1.4;
      }
    </style>

    <div class="usage-box">
      <h2>🔧 How to Use EnerMat Explorer</h2>
      <ul>
        <li><strong>Select End-members</strong> (A &amp; B, optionally C) from the sidebar dropdowns.</li>
        <li><strong>Set Environment:</strong> adjust Humidity (%) and Temperature (°C) sliders.</li>
        <li><strong>Tune Model:</strong> define your Band-gap window, Bowing parameter, x-step (and y-step), and Ge-fraction z.</li>
        <li>Hit the <code>▶ Run screening</code> button to compute an interactive table and scatter/3D plot of Eg vs. Ehull vs. Score.</li>
        <li>Use <code>⏪ Previous</code> to step back through your history, and grab your CSV/TXT/DOCX via the Download tab.</li>
      </ul>
      <p><em>
        Instantly visualize how band-gap, phase-stability and oxidation-resistance trade off with theoretical efficiency—no DFT required! 
      </em></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────── RUNNING SCREEN ───────────
col_run, col_prev = st.columns([3,1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_prev = col_prev.button("⏪ Previous", disabled=not st.session_state.history)

if do_prev:
    st.session_state.history.pop()
    prev = st.session_state.history[-1]
    df, mode = prev["df"], prev["mode"]
    st.success("Showing previous result")

elif do_run:
    # sanity-check
    for f in ([A, B] if mode.startswith("Binary") else [A, B, C]):
        if f not in END_MEMBERS:
            st.error(f"❌ Unknown end-member: {f}")
            st.stop()

    if mode.startswith("Binary"):
        df = _run_binary(
            A, B, rh, temp, (bg_lo, bg_hi), bow, dx,
            z=z, application=application
        )
    else:
        df = _run_ternary(
            A, B, C, rh, temp,
            (bg_lo, bg_hi), {"AB":bow,"AC":bow,"BC":bow},
            dx=dx, dy=dy, z=z, application=application
        )
    st.session_state.history.append({"mode":mode, "df":df})

elif not st.session_state.history:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

  # ─────────── DISPLAY RESULTS ───────────
df = st.session_state.history[-1]["df"]
mode = st.session_state.history[-1]["mode"]

tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table","📈 Plot","📥 Download"])

with tab_tbl:
    st.dataframe(df, use_container_width=True, height=440)

with tab_plot:
    if mode.startswith("Binary") and {"Ehull","Eg"}.issubset(df.columns):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Ehull"], y=df["Eg"], mode="markers",
            marker=dict(
                size=8+12*df["score"], color=df["score"],
                colorscale="Viridis", cmin=0, cmax=1,
                colorbar=dict(title="Score"), line=dict(width=0.5, color="black")
            ),
            hovertemplate="<b>%{customdata[6]}</b><br>Eg=%{y:.3f} eV<br>Ehull=%{x:.4f} eV/at<br>Score=%{marker.color:.3f}<extra></extra>",
            customdata=df.to_numpy()
        ))
        fig.add_shape(type="rect", x0=0, x1=0.05, y0=bg_lo, y1=bg_hi,
                      line=dict(color="LightSeaGreen",dash="dash"), fillcolor="LightSeaGreen", opacity=0.1)
        fig.update_layout(
    title="EnerMat Binary Screen",
    xaxis_title="Ehull (eV/atom)",
    yaxis_title="Eg (eV)",
    template="simple_white",
    font=dict(
        family="Arial",
        size=12,
        color="black"
    ),
    width=720,
    height=540,
    margin=dict(l=60, r=60, t=60, b=60),
    coloraxis_colorbar=dict(
        title=dict(text="Score", font=dict(size=12)),  # ✅ Fix applied here
        tickfont=dict(size=12)
    )
)
        st.plotly_chart(fig, use_container_width=True)
    elif mode.startswith("Ternary") and {"x","y","score"}.issubset(df.columns):
        fig = px.scatter_3d(df, x="x", y="y", z="score", color="score",
                            color_continuous_scale="Viridis",
                            labels={"x":"B2 fraction","y":"B3 fraction"}, height=500)
        st.plotly_chart(fig, use_container_width=True)

with tab_dl:
    st.download_button("📥 Download CSV", df.to_csv(index=False).encode(), "EnerMat_results.csv", "text/csv")

# ╭─────────────────────  AUTO-REPORT  (TXT / DOCX)  ────────────────╮
_top = df.iloc[0]
formula = str(_top["formula"])
coords  = ", ".join(
    f"{c}={_top[c]:.2f}"
    for c in ("x", "y", "z", "ge_frac") if c in _top and pd.notna(_top[c])
)
label = formula if len(df) == 1 else f"{formula} ({coords})"

_txt = (
    "EnerMat auto-report  "
    f"{datetime.date.today()}\n"
    f"Top candidate   : {label}\n"
    f"Band-gap [eV]   : {_top['Eg']}\n"
    f"Ehull [eV/at.]  : {_top['Ehull']}\n"
    f"Eox_e [eV/e⁻]   : {_top.get('Eox_e', 'N/A')}\n"
    f"Score           : {_top['score']}\n"
)

st.download_button("📄 Download TXT", _txt,
                   "EnerMat_report.txt", mime="text/plain")

_doc = Document()
_doc.add_heading("EnerMat Report", level=0)
_doc.add_paragraph(f"Date : {datetime.date.today()}")
_doc.add_paragraph(f"Top candidate : {label}")

table = _doc.add_table(rows=1, cols=2)
table.style = "LightShading-Accent1"
hdr = table.rows[0].cells
hdr[0].text, hdr[1].text = "Property", "Value"
for k in ("Eg", "Ehull", "Eox_e", "score"):
    if k in _top:
        row = table.add_row().cells
        row[0].text, row[1].text = k, str(_top[k])

buf = io.BytesIO()
_doc.save(buf)
buf.seek(0)
st.download_button("📝 Download DOCX", buf,
                   "EnerMat_report.docx",
                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
# ─────────────────────────────────────────────────────────────────────────────
