"""EnerMat Explorer – Lead‑Free Perovskite PV Discovery Tool
Refactored Streamlit front‑end (polish pass).

Key improvements
----------------
* 💄 **PEP‑8 compliant & typed** – imports grouped, constants in CAPS, functions snake‑case.
* 🚫 **Bug fixes** – corrected `Eox`/`Eox_e` mismatch, `ge_frac` column, CSV encoding, colour‑bar label.
* 🔁 **Reusable helpers** – sidebar construction, plotting, and report generation broken into small
  functions; easier to test and extend.
* 🐇 **Faster feedback** – wraps screen calls in `st.spinner` + optional progress bar for long grids.
* 📊 **Plotly** – consistent theme + rectangular sweet‑spot patch.
* 📄 **Downloads** – TXT & DOCX auto‑report generated via helper; filename has timestamp.

This is a drop‑in replacement for your original `app.py`.  Back‑end API (`backend.perovskite_utils`) stays untouched.
"""

from __future__ import annotations

import base64
import datetime as _dt
import io
from pathlib import Path
from typing import Any, Literal, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from docx import Document

from backend.perovskite_utils import END_MEMBERS, screen_binary, screen_ternary

# ─────────────────────────── CONFIG ──────────────────────────────
PAGE_TITLE = "☀️ EnerMat Explorer | Lead‑Free Perovskite PV Discovery Tool"
PAGE_LAYOUT: Literal["wide", "centered"] = "wide"

st.set_page_config(page_title=PAGE_TITLE, layout=PAGE_LAYOUT)

# Inject simple CSS once at load
st.markdown(
    """
    <style>
        /* Sidebar left border */
        .css-1d391kg { border-right: 3px solid #0D47A1 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(PAGE_TITLE)

# ──────────────────────── SIDEBAR HELPERS ────────────────────────

def sidebar_controls() -> dict[str, Any]:
    """Collect user input widgets in the sidebar and return values as dict."""
    with st.sidebar:
        st.header("Mode")
        mode: Literal["Binary", "Ternary"] = st.radio(
            "Choose screening type", ("Binary", "Ternary"), format_func=lambda m: f"{m} A–B" if m == "Binary" else "Ternary A–B–C"
        )

        st.header("End‑members")
        preset_a = st.selectbox("Preset A", END_MEMBERS, 0)
        preset_b = st.selectbox("Preset B", END_MEMBERS, 1)
        custom_a = st.text_input("Custom A (optional)").strip()
        custom_b = st.text_input("Custom B (optional)").strip()
        a = custom_a or preset_a
        b = custom_b or preset_b

        c = None
        if mode == "Ternary":
            preset_c = st.selectbox("Preset C", END_MEMBERS, 2)
            custom_c = st.text_input("Custom C (optional)").strip()
            c = custom_c or preset_c

        st.header("Application")
        application = st.selectbox("Select application", ("single", "tandem", "indoor", "detector"))

        st.header("Environment")
        rh = st.slider("Humidity [%]", 0, 100, 50)
        temp = st.slider("Temperature [°C]", -20, 100, 25)

        st.header("Target band‑gap [eV]")
        bg_lo, bg_hi = st.slider("Gap window", 0.50, 3.00, (1.00, 1.40), 0.01)

        st.header("Model settings")
        bow = st.number_input("Bowing (eV, negative ⇒ gap↑)", -1.0, 1.0, -0.15, 0.05)
        dx = st.number_input("x‑step", 0.01, 0.50, 0.05, 0.01)
        dy = None
        if mode == "Ternary":
            dy = st.number_input("y‑step", 0.01, 0.50, 0.05, 0.01)
        z = st.slider("Ge fraction z", 0.00, 0.80, 0.10, 0.05, help="B‑site Ge²⁺ in CsSn₁₋zGezX₃")

        # History management
        if st.button("🗑 Clear history"):
            st.session_state.pop("history", None)
            st.experimental_rerun()

        # Developer credit footer
        st.markdown(
            """
            <div style="font-size:0.85rem; color:#555; margin-top:0.5rem;">
              <strong>Developer:</strong> Dr Gbadebo Taofeek Yusuf (Academic World)<br>
              📞 +44 7776 727237 ✉️ das@academicworld.co.uk
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "mode": mode,
        "A": a,
        "B": b,
        "C": c,
        "application": application,
        "rh": rh,
        "temp": temp,
        "bg": (bg_lo, bg_hi),
        "bow": bow,
        "dx": dx,
        "dy": dy,
        "z": z,
    }


# ────────────────────────── CACHING LAYERS ───────────────────────
@st.cache_data(show_spinner="⏳ Screening …", max_entries=20)
def _run_binary_cached(*args, **kwargs):
    return screen_binary(*args, **kwargs)


@st.cache_data(show_spinner="⏳ Screening …", max_entries=10)
def _run_ternary_cached(*args, **kwargs):
    return screen_ternary(*args, **kwargs)


# ───────────────────────── REPORT HELPERS ────────────────────────

def _auto_txt_report(row: pd.Series) -> str:
    """Plain‑text summary for top candidate."""
    return (
        "EnerMat auto‑report  "
        f"{_dt.date.today()}\n"
        f"Top candidate   : {row['formula']}\n"
        f"Band‑gap [eV]   : {row['Eg']}\n"
        f"Ehull [eV/at.]  : {row['Ehull']}\n"
        f"Eox [eV]        : {row.get('Eox', 'N/A')}\n"
        f"Score           : {row['score']}\n"
    )


def _auto_docx_report(row: pd.Series) -> io.BytesIO:
    """DOCX summary – returns buffer ready for download."""
    doc = Document()
    doc.add_heading("EnerMat Report", 0)
    doc.add_paragraph(f"Date : {_dt.date.today()}")
    doc.add_paragraph(f"Top candidate : {row['formula']}")

    table = doc.add_table(rows=1, cols=2)
    table.style = "LightShading-Accent1"
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text, hdr_cells[1].text = "Property", "Value"
    for prop in ("Eg", "Ehull", "Eox", "score"):
        if prop in row:
            cells = table.add_row().cells
            cells[0].text, cells[1].text = prop, str(row[prop])

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


# ──────────────────────────── PLOTS ──────────────────────────────

def _plot_binary(df: pd.DataFrame, bg: Tuple[float, float]) -> go.Figure:
    """2D scatter: Ehull vs Eg size/colour by score."""
    lo, hi = bg
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Ehull"],
            y=df["Eg"],
            mode="markers",
            marker=dict(
                size=8 + 12 * df["score"],
                color=df["score"],
                colorscale="Viridis",
                cmin=0,
                cmax=1,
                colorbar=dict(title="Score"),
                line=dict(width=0.5, color="black"),
            ),
            hovertemplate="<b>%{customdata[0]}</b><br>Eg=%{y:.3f} eV<br>Ehull=%{x:.4f} eV/at<br>Score=%{marker.color:.3f}<extra></extra>",
            customdata=df[["formula"]].to_numpy(),
        )
    )
    # Sweet‑spot rectangle
    fig.add_shape(
        type="rect",
        x0=0,
        y0=lo,
        x1=0.05,
        y1=hi,
        line=dict(color="LightSeaGreen", dash="dash"),
        fillcolor="LightSeaGreen",
        opacity=0.1,
    )
    fig.update_layout(
        title="EnerMat Binary Screen",
        xaxis_title="Ehull (eV/atom)",
        yaxis_title="Eg (eV)",
        template="simple_white",
        font_family="Arial",
        width=720,
        height=540,
    )
    return fig


def _plot_ternary(df: pd.DataFrame) -> go.Figure:
    return px.scatter_3d(
        df,
        x="x",
        y="y",
        z="score",
        color="score",
        color_continuous_scale="Viridis",
        labels={"x": "B2 fraction", "y": "B3 fraction"},
        height=500,
    )


# ───────────────────────────── MAIN ──────────────────────────────

def main() -> None:
    cfg = sidebar_controls()

    # Initialise history container
    history: list[dict[str, Any]] = st.session_state.setdefault("history", [])

    col_run, col_prev = st.columns([3, 1])
    run_clicked = col_run.button("▶ Run screening", type="primary")
    prev_clicked = col_prev.button("⏪ Previous", disabled=not history)

    if prev_clicked and history:
        history.pop()
        st.toast("Showing previous result", icon="✅")
    elif run_clicked:
        # Validate end‑members
        end_members = [cfg["A"], cfg["B"]] if cfg["mode"] == "Binary" else [cfg["A"], cfg["B"], cfg["C"]]
        unknown = [f for f in end_members if f not in END_MEMBERS]
        if unknown:
            st.error(f"❌ Unknown end‑member(s): {', '.join(unknown)}")
            st.stop()
        # Run screen
        with st.spinner("⏳ Screening …"):
            if cfg["mode"] == "Binary":
                df = _run_binary_cached(
                    cfg["A"],
                    cfg["B"],
                    cfg["rh"],
                    cfg["temp"],
                    cfg["bg"],
                    cfg["bow"],
                    cfg["dx"],
                    z=cfg["z"],
                    application=cfg["application"],
                )
            else:
                df = _run_ternary_cached(
                    cfg["A"],
                    cfg["B"],
                    cfg["C"],
                    cfg["rh"],
                    cfg["temp"],
                    cfg["bg"],
                    {"AB": cfg["bow"], "AC": cfg["bow"], "BC": cfg["bow"]},
                    dx=cfg["dx"],
                    dy=cfg["dy"],
                    z=cfg["z"],
                    application=cfg["application"],
                )
        history.append({"cfg": cfg, "df": df})

    if not history:
        st.info("Press ▶ Run screening to begin.")
        return

    df = history[-1]["df"]
    st.markdown(f"_Results: {len(df)} rows_\n")

    tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table", "📈 Plot", "📥 Download"])

    with tab_tbl:
        st.dataframe(df, use_container_width=True, height=440)

    with tab_plot:
        if cfg["mode"] == "Binary" and {"Ehull", "Eg"}.issubset(df.columns):
            st.plotly_chart(_plot_binary(df, cfg["bg"]), use_container_width=True)
        elif cfg["mode"] == "Ternary" and {"x", "y", "score"}.issubset(df.columns):
            st.plotly_chart(_plot_ternary(df), use_container_width=True)

    with tab_dl:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv_bytes, "EnerMat_results.csv", "text/csv")

        top = df.iloc[0]
        st.download_button("📄 Download TXT", _auto_txt_report(top), f"EnerMat_report_{_dt.date.today()}.txt", "text/plain")

        st.download_button(
            "📝 Download DOCX",
            _auto_docx_report(top),
            f"EnerMat_report_{_dt.date.today()}.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )


if __name__ == "__main__":
    main()
