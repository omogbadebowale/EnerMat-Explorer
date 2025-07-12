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
   bowing = st.number_input(
    "Bowing b  (use negative for Br→Cl alloys)",
    min_value = -1.0,      # allow -1.0 eV
    max_value =  1.0,
    value     = -0.15,     # default for CsSnBr3↔CsSnCl3
    step      =  0.05,
    format    = "%.2f",
)
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
        dx=dx
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
            formula_A=A, formula_B=B,
            rh=rh, temp=temp,
            bg_window=(bg_lo, bg_hi), bowing=bow, dx=dx
        )
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

    df = df.rename(columns={
        "energy_above_hull": "stability",
        "band_gap": "Eg"
    })

    entry = {
        "mode": mode,
        "A": A, "B": B, "rh": rh, "temp": temp,
        "bg": (bg_lo, bg_hi), "bow": bow, "dx": dx,
        "df": df
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
        required = [c for c in ["stability", "Eg", "score"] if c in df.columns]
        if len(required) < 3:
            missing = set(["stability", "Eg", "score"]) - set(df.columns)
            st.error(f"❌ Missing required columns for plotting: {', '.join(missing)}")
            st.stop()
        plot_df = df.dropna(subset=required).copy()

        # Build a high-quality scatter
        fig = px.scatter(
            plot_df,
            x="stability",
            y="Eg",
            color="score",
            color_continuous_scale="Turbo",
            hover_data=["formula", "x", "Eg", "stability", "score"],
            width=1200,
            height=800
        )

        fig.update_traces(
            marker=dict(
                size=12,
                opacity=0.9,
                line=dict(width=1, color="black")
            )
        )

        # Highlight top 20%
        top_cut = plot_df["score"].quantile(0.80)
        top_mask = plot_df["score"] >= top_cut
        fig.add_trace(
            go.Scatter(
                x=plot_df.loc[top_mask, "stability"],
                y=plot_df.loc[top_mask, "Eg"],
                mode="markers",
                marker=dict(
                    size=20,
                    symbol="circle-open",
                    line=dict(width=2, color="black")
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

        fig.update_layout(
            template="plotly_white",
            margin=dict(l=80, r=40, t=60, b=80),
            font=dict(family="Times New Roman", size=16, color="#333"),
            xaxis=dict(title="Stability", title_font_size=18, tickfont_size=14),
            yaxis=dict(title="Band Gap (eV)", title_font_size=18, tickfont_size=14),
            coloraxis_colorbar=dict(
                title="Score",
                title_font_size=16,
                tickfont_size=14,
                thicknessmode="pixels", thickness=20, len=0.75,
                outlinewidth=1, outlinecolor="#666"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Uncomment to export a sharp PNG/SVG (requires `pip install kaleido`):
        # fig.write_image("binary_screening.png", scale=3)

    else:
        required = [c for c in ["x", "y", "score"] if c in df.columns]
        if len(required) < 3:
            st.warning("❗ Not enough columns for ternary 3D plot.")
            st.stop()
        plot_df = df.dropna(subset=required).copy()

        fig3d = px.scatter_3d(
            plot_df,
            x="x", y="y", z="score",
            color="score",
            color_continuous_scale="Turbo",
            hover_data={k: True for k in ["x", "y", "Eg", "score"] if k in plot_df.columns},
            width=1200,
            height=900
        )

        fig3d.update_traces(
            marker=dict(
                size=5,
                opacity=0.9,
                line=dict(width=1, color="black")
            )
        )

        fig3d.update_layout(
            template="plotly_white",
            margin=dict(l=60, r=60, t=60, b=60),
            font=dict(family="Calibri", size=14, color="#222"),
            scene=dict(
                xaxis=dict(title="A fraction", title_font_size=16, tickfont_size=12),
                yaxis=dict(title="B fraction", title_font_size=16, tickfont_size=12),
                zaxis=dict(title="Score",      title_font_size=16, tickfont_size=12)
            ),
            coloraxis_colorbar=dict(
                title="Score",
                title_font_size=14,
                tickfont_size=12,
                thickness=18, len=0.6,
                outlinewidth=1, outlinecolor="#444"
            )
        )

        st.plotly_chart(fig3d, use_container_width=True)

        # Uncomment to export a sharp 3D PNG/SVG:
        # fig3d.write_image("ternary_screening.png", scale=3)

# ─── Download Tab ────────────────────────────────────────────────────────────
with tab_dl:
    csv = df.to_csv(index=False).encode()
    st.download_button("📥 Download CSV", csv, "EnerMat_results.csv", "text/csv")

    top = df.iloc[0]
    if mode == "Binary A–B":
        top_label = top.formula
    else:
        top_label = f"{A}-{B}-{C} x={top.x:.2f} y={top.y:.2f}"

    txt = f"""EnerMat report ({datetime.date.today()})
Top candidate : {top_label}
Band-gap     : {top.Eg}
Stability    : {getattr(top, 'stability', 'N/A')}
Score        : {top.score}
"""
    st.download_button("📄 Download TXT", txt, "EnerMat_report.txt", "text/plain")

    doc = Document()
    doc.add_heading("EnerMat Report", 0)
    doc.add_paragraph(f"Date: {datetime.date.today()}")
    doc.add_paragraph(f"Top candidate: {top_label}")
    tbl = doc.add_table(rows=1, cols=2)
    hdr_cells = tbl.rows[0].cells
    hdr_cells[0].text = "Property"
    hdr_cells[1].text = "Value"
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
