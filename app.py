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

# ─── Branded header with large, responsive SVG/PNG logo ──────────
ASSETS_DIR = Path(__file__).with_name("assets")
for name, mime in [
    ("enermat_logo.svg", "image/svg+xml"),
    ("enermat_logo.png", "image/png"),
    ("enermat_logo.png.webp", "image/webp"),
]:
    p = ASSETS_DIR / name
    if p.exists():
        logo_b64 = base64.b64encode(p.read_bytes()).decode()
        logo_mime = mime
        break
else:
    logo_b64, logo_mime = "", "image/png"      # graceful fallback

st.markdown(
    f"""
    <style>
      .em-logo {{
        text-align:center;
        margin:0 0 1.4rem 0;
        padding-top:.4rem;
      }}
      .em-logo img {{
        height:min(45vw,320px);   /* responsive, caps at 220 px */
        width:auto;
      }}
      .em-logo h1 {{
        margin:.5rem 0 0 0;
        font-size:2.3rem;
        font-weight:700;
      }}
      .em-logo small {{
        opacity:.75;
        font-size:.95rem;
      }}
    </style>

    <div class="em-logo">
        <img src="data:{logo_mime};base64,{logo_b64}" alt="EnerMat logo">
        <h1>EnerMat Explorer</h1>
        <small>Lead-free PV discovery tool</small>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────── STREAMLIT PAGE CONFIG ───────────
st.set_page_config("EnerMat Explorer", layout="wide")
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
# NEW — show only if a Bi or Sb preset is selected
if any(elem in f for elem in ("Bi", "Sb") for f in (A, B)):
    y = st.slider(
        "Sb fraction y", 0.00, 1.00, 0.00, 0.05,
        help="B-site Sb³⁺ in Cs₃(Bi₁₋ySb_y)₂X₉ or Cs₂Ag(Bi₁₋ySb_y)X₆"
    )
else:
    y = 0.00          # keeps legacy path unchanged
    if st.button("🗑 Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()

    st.caption(f"⚙️ Build SHA : dev • 🕒 {datetime.datetime.now():%Y-%m-%d %H:%M}")

# ─────────── CACHE WRAPPERS ───────────
@st.cache_data(show_spinner="⏳ Screening …", max_entries=20)
def _run_binary(*args, **kwargs):
    return screen_binary(*args, **kwargs)

@st.cache_data(show_spinner="⏳ Screening …", max_entries=10)
def _run_ternary(*args, **kwargs):
    return screen_ternary(*args, **kwargs)

# ─────────── PROBLEM STATEMENT / SIGNIFICANCE ───────────
st.markdown(
    """
    <style>
    .problem-box {
      background: linear-gradient(135deg, #FFD54F 0%, #FFB74D 100%);
      padding: 20px;
      border-radius: 12px;
      margin-bottom: 24px;
      color: #333;
      font-family: Arial, sans-serif;
    }
    .problem-box h2 {
      margin: 0 0 8px;
      font-size: 1.8rem;
      color: #fff;
      text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .problem-box p {
      margin: 0 0 12px;
      font-size: 1rem;
      line-height: 1.5;
      color: #333;
    }
    .problem-box ul {
      margin: 0;
      padding-left: 1.2em;
      font-size: 0.95rem;
      color: #333;
    }
    </style>
    <div class="problem-box">
      <h2>Context & Value</h2>
      <p>
        Lead–halide perovskites deliver record solar efficiencies but suffer from
        environmental toxicity and rapid degradation under heat, moisture, or oxygen.
        Tin–based, lead‑free analogues offer a safer path but balancing their
        <strong>Eg</strong> (band gap), <strong>Eₕᵤₗₗ</strong> (phase stability),
        <strong>ΔEₒₓ</strong> (oxidation resistance) and
        <strong>PCEₘₐₓ</strong> (theoretical solar efficiency) remains a major hurdle.
      </p>
      <ul>
        <li><strong>Eg</strong>: Direct band gap (eV) governing light absorption (ideal ~1.3 eV).</li>
        <li><strong>Eₕᵤₗₗ</strong>: Energy above the convex hull (eV/atom), a measure of thermodynamic stability.</li>
        <li><strong>ΔEₒₓ</strong>: Oxidation enthalpy (eV per Sn), quantifying Sn²⁺→Sn⁴⁺ drive.</li>
        <li><strong>PCEₘₐₓ</strong>: Shockley–Queisser power‑conversion efficiency (%), the detailed‑balance limit.</li>
      </ul>
      <p>
        This app will help materials scientists, solar‑cell engineers and educators to quickly pinpoint promising perovskite alloys, cutting down costly trial‑and‑error experiments and speeding up innovation.
      </p>
      <p>
        It will also assist Industry partners gain a transparent decision‑making tool to evaluate trade‑offs between efficiency and stability, while students enjoy an interactive platform to explore perovskite design principles hands‑on.
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
        background: linear-gradient(135deg, #BBDEFB 0%, #E3F2FD 100%);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 24px;
        color: #0D47A1;
        font-family: Arial, sans-serif;
      }
      .usage-box h2 {
        margin: 0 0 8px;
        font-size: 1.6rem;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.6);
      }
      .usage-box p {
        margin: 0 0 12px;
        font-size: 0.95rem;
        line-height: 1.5;
      }
      .usage-box ul {
        margin: 0;
        padding-left: 1.2em;
        font-size: 0.9rem;
      }
    </style>
    <div class="usage-box">
      <h2>🔧 How to Use EnerMat</h2>
      <p>
        Follow these simple steps to identify your optimal lead‑free perovskite:
      </p>
      <ul>
        <li>
          <strong>Select End‑members</strong> (A & B, and optionally C) from the dropdowns on the left.
        </li>
        <li>
          <strong>Set Environment</strong>: adjust humidity (%) and temperature (°C) sliders.
        </li>
        <li>
          <strong>Tune Model</strong>: choose your target band‑gap window, bowing parameter, composition step‑sizes, and Ge‑fraction.
        </li>
        <li>
          Hit the <code>▶ Run screening</code> button to generate an interactive table & plot of Eg vs. Ehull vs. Score.
        </li>
        <li>
          Use <code>⏪ Previous</code> to step back through prior results, and download CSV/TXT/DOCX via the Download tab.
        </li>
      </ul>
      <p style="font-style:italic; font-size:0.85rem;">
        Instantly visualize how band‐gap, phase‐stability, oxidation‐resistance, and theoretical efficiency trade off — no DFT runs required!
      </p>
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
            z=z, y=y, application=application
        )
    else:
        df = _run_ternary(
            A, B, C, rh, temp,
            (bg_lo, bg_hi), {"AB":bow,"AC":bow,"BC":bow},
            dx=dx, dy=dy, z=z, y=y, application=application
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
