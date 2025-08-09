import datetime
import io
import json

import streamlit as st
import pandas as pd
from plotly import graph_objects as go
import plotly.express as px
from docx import Document

from backend.perovskite_utils import (
    screen_binary,
    screen_ternary,
    END_MEMBERS,
    APPLICATION_CONFIG,
    offline_mode,
)

# ─────────── PAGE CONFIG ───────────
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
    st.session_state.history = []  # list of {"mode","df","params"}

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
        ["single", "tandem", "indoor", "detector", "custom"],
        help="Fixed, defensible presets. 'custom' reveals Gaussian controls."
    )

    # Custom Gaussian controls (optional)
    custom_center = custom_sigma = None
    if application == "custom":
        st.subheader("Custom target (optional Gaussian)")
        custom_center = st.number_input("Eg center (eV)", 0.5, 3.0, 1.30, 0.01)
        custom_sigma  = st.number_input("Eg sigma (eV)", 0.00, 0.50, 0.10, 0.01)

    st.header("Model settings")
    bow = st.number_input("Bowing (eV, negative ⇒ gap↑)", -1.0, 1.0, -0.15, 0.05)
    dx = st.number_input("x-step", 0.01, 0.50, 0.05, 0.01)
    if mode.startswith("Ternary"):
        dy = st.number_input("y-step", 0.01, 0.50, 0.05, 0.01)

    st.subheader("B-site dopant")
    dopant_element = st.selectbox("Choose dopant element", ["Ge", "Si", "Pb", "Mn", "Zn", "None"], index=0)
    dopant_fraction = st.slider(
        f"{dopant_element} fraction z in Sn₁₋z{dopant_element}z",
        0.00, 0.80, 0.10 if dopant_element != "None" else 0.00, 0.05,
    )

    st.header("Structural penalty")
    t0 = st.number_input("Target tolerance factor t₀", 0.80, 1.10, 0.95, 0.01)
    beta = st.number_input("Penalty stiffness β", 0.0, 5.0, 1.0, 0.1)

    if st.button("🗑 Clear history"):
        st.session_state.history = []
        st.rerun()

    st.markdown(
        """
        <div style="font-size:0.85rem; color:#555; margin-top:0.5rem;">
          <strong>Developer:</strong> Dr Gbadebo Taofeek Yusuf (Academic World)<br>
          📞 +44 7776 727237 &nbsp; ✉️ das@academicworld.co.uk
        </div>
        """,
        unsafe_allow_html=True,
    )

if offline_mode:
    st.warning("Running in offline/demo mode (no valid MP API key found). Some formulas may return no data.")

# ─────────── CACHE WRAPPERS ───────────
@st.cache_data(show_spinner="⏳ Screening …", max_entries=20)
def _run_binary(args: dict):
    return screen_binary(**args)

@st.cache_data(show_spinner="⏳ Screening …", max_entries=10)
def _run_ternary(args: dict):
    return screen_ternary(**args)

# ─────────── Overview (brief) ───────────
st.info("Score = SQ(Eg) × exp(−Ehull/kTmix) × exp(ΔEox/kTeff) × exp(−β|t−t₀|). Top candidate is normalized to 1.0.")

# ─────────── RUN / PREVIOUS ───────────
col_run, col_prev = st.columns([3, 1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_prev = col_prev.button("⏪ Previous", disabled=not st.session_state.history)

if do_prev:
    st.session_state.history.pop()
    st.success("Showing previous result")
    if not st.session_state.history:
        st.stop()

if do_run:
    base_params = dict(
        rh_unused=0.0, temp_unused=25.0, bg_unused=(0.0, 0.0),  # kept for API stability
        application=application,
        t0=t0, beta=beta,
        dopant_element=dopant_element,
        dopant_fraction=dopant_fraction,
        custom_center=custom_center, custom_sigma=custom_sigma,
    )
    if mode.startswith("Binary"):
        args = dict(A=A, B=B, bow=bow, dx=dx, z=dopant_fraction, **base_params)
        df = _run_binary(args)
    else:
        args = dict(A=A, B=B, C=C, bows={"AB": bow, "AC": bow, "BC": bow},
                    dx=dx, dy=dy, z=dopant_fraction, **base_params)
        df = _run_ternary(args)

    if df.empty:
        st.error("No data returned for the chosen inputs. Try different end-members.")
        st.stop()

    st.session_state.history.append({
        "mode": mode,
        "df": df,
        "params": {
            "A": A, "B": B, **({"C": C} if mode.startswith("Ternary") else {}),
            "application": application, "bow": bow, "dx": dx, **({"dy": dy} if mode.startswith("Ternary") else {}),
            "z": dopant_fraction, "dopant_element": dopant_element,
            "t0": t0, "beta": beta,
            "custom_center": custom_center, "custom_sigma": custom_sigma,
        }
    })

if not st.session_state.history:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

# ─────────── DISPLAY RESULTS ───────────
df = st.session_state.history[-1]["df"]
mode = st.session_state.history[-1]["mode"]
application = st.session_state.history[-1]["params"]["application"]

tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table", "📈 Plot", "📥 Download"])

with tab_tbl:
    st.dataframe(df, use_container_width=True, height=460)

with tab_plot:
    # choose a shading window for the plot
    y0 = y1 = None
    if application == "custom":
        cc = st.session_state.history[-1]["params"].get("custom_center")
        ss = st.session_state.history[-1]["params"].get("custom_sigma")
        if cc and ss and ss > 0:
            y0, y1 = cc - 2*ss, cc + 2*ss
    else:
        rng = APPLICATION_CONFIG.get(application, {}).get("range")
        if rng:
            y0, y1 = rng

    if mode.startswith("Binary") and {"Ehull","Eg","score"}.issubset(df.columns):
        fig = go.Figure()
        custom = pd.DataFrame({
            "Eg": df["Eg"],
            "PCE": df.get("PCE_max (%)", 0.0),
            "label": df["formula"],
            "dopant": df.get("dopant", "None"),
            "scope": df.get("scope", "Unknown"),
        }).to_numpy()

        fig.add_trace(go.Scatter(
            x=df["Ehull"], y=df["Eg"], mode="markers",
            marker=dict(
                size=8 + 12 * df["score"], color=df["score"],
                colorscale="Viridis", cmin=0, cmax=1,
                colorbar=dict(title="Score"),
                line=dict(width=0.5, color="black"),
            ),
            hovertemplate="<b>%{customdata[2]}</b><br>"
                          "Eg=%{customdata[0]:.3f} eV<br>"
                          "Ehull=%{x:.4f} eV/at<br>"
                          "Score=%{marker.color:.3f}<br>"
                          "PCE_max=%{customdata[1]:.1f}%<br>"
                          "Dopant=%{customdata[3]}<br>"
                          "Scope=%{customdata[4]}<extra></extra>",
            customdata=custom
        ))
        if y0 is not None and y1 is not None:
            fig.add_shape(
                type="rect", x0=0, x1=0.05, y0=y0, y1=y1,
                line=dict(color="LightSeaGreen", dash="dash"),
                fillcolor="LightSeaGreen", opacity=0.10
            )
        fig.update_layout(
            title=f"EnerMat Binary Screen – {application}",
            xaxis_title="Ehull (eV/atom)", yaxis_title="Eg (eV)",
            template="simple_white", width=720, height=540,
            margin=dict(l=60, r=60, t=60, b=60)
        )
        st.plotly_chart(fig, use_container_width=True)

    elif mode.startswith("Ternary") and {"x","y","score"}.issubset(df.columns):
        fig = px.scatter_3d(df, x="x", y="y", z="score", color="score",
                            color_continuous_scale="Viridis",
                            labels={"x":"B2 fraction","y":"B3 fraction"},
                            height=520)
        st.plotly_chart(fig, use_container_width=True)

with tab_dl:
    # Current run CSV (already trimmed by backend)
    st.download_button(
        "📥 Download current CSV",
        df.to_csv(index=False).encode(),
        "EnerMat_results.csv", "text/csv"
    )

    # ALL runs (keep relevant columns + params)
    if st.session_state.history:
        all_rows = []

        def _freeze(v):
            if isinstance(v, (dict, list, tuple, set)):
                try:
                    return json.dumps(v, separators=(",", ":"))
                except Exception:
                    return str(v)
            return v

        for i, h in enumerate(st.session_state.history, 1):
            dfi = pd.DataFrame(h["df"]).copy()
            for k, v in h.get("params", {}).items():
                dfi[f"param_{k}"] = _freeze(v)
            dfi["run_id"] = i
            all_rows.append(dfi)
        big = pd.concat(all_rows, ignore_index=True)
        st.download_button(
            "📦 Download ALL runs (CSV)",
            big.to_csv(index=False).encode(),
            "EnerMat_all_runs.csv", "text/csv"
        )

    # Auto TXT report (top row)
    _top = df.iloc[0]
    formula = str(_top["formula"])
    coords = ", ".join(
        f"{c}={_top[c]:.2f}"
        for c in ("x", "y", "z") if c in _top and pd.notna(_top[c])
    )
    label = formula if len(df) == 1 else f"{formula} ({coords})"

    _txt = (
        "EnerMat auto-report  "
        f"{datetime.date.today()}\n"
        f"Top candidate   : {label}\n"
        f"Dopant / z      : {_top.get('dopant','None')} / {_top.get('z','0.00')}\n"
        f"Band-gap [eV]   : {_top['Eg']}\n"
        f"Ehull [eV/at.]  : {_top['Ehull']}\n"
        f"Eox_e [eV/e⁻]   : {_top.get('Eox_e', 'N/A')}\n"
        f"PCE_max [%]     : {_top.get('PCE_max (%)', 'N/A')}\n"
        f"Score           : {_top['score']}\n"
        f"Scope           : {_top.get('scope','Unknown')}\n"
    )
    st.download_button("📄 Download TXT", _txt, "EnerMat_report.txt", mime="text/plain")

    # DOCX report
    _doc = Document()
    _doc.add_heading("EnerMat Report", level=0)
    _doc.add_paragraph(f"Date : {datetime.date.today()}")
    _doc.add_paragraph(f"Top candidate : {label}")

    table = _doc.add_table(rows=1, cols=2)
    table.style = "LightShading-Accent1"
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text = "Property", "Value"
    for k in ("dopant", "z", "Eg", "Ehull", "Eox_e", "PCE_max (%)", "score", "scope"):
        if k in _top:
            row = table.add_row().cells
            row[0].text, row[1].text = k, str(_top[k])

    buf = io.BytesIO()
    _doc.save(buf)
    buf.seek(0)
    st.download_button("📝 Download DOCX", buf,
                       "EnerMat_report.docx",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
