# app.py  –  EnerMat Perovskite Explorer v9.6  (2025-07-13, “oxidation-fixed” edition)

import io, os, datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from docx import Document

# ── Materials-Project API key ────────────────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    st.error("🛑  You need a valid 32-character MP_API_KEY in Secrets.")
    st.stop()

# ── Backend helpers ─────────────────────────────────────────────────────────
from backend.perovskite_utils import (
    mix_abx3    as screen_binary,
    screen_ternary,
    END_MEMBERS,
    fetch_mp_data as _summary,
)

# ── Streamlit page config ───────────────────────────────────────────────────
st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.6")

# ── Session state ───────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar – I/O controls ──────────────────────────────────────────────────
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
    if mode == "Ternary A–B–C":
        preset_C = st.selectbox("Preset C", END_MEMBERS, 2)
        custom_C = st.text_input("Custom C (optional)").strip()
        C = custom_C or preset_C

    st.header("Environment")
    rh   = st.slider("Humidity [%]", 0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)

    st.header("Target band-gap [eV]")
    bg_lo, bg_hi = st.slider("Gap window", 0.5, 3.0, (1.0, 1.4), 0.01)

    st.header("Model settings")
    bow = st.number_input("Bowing (eV, negative ⇒ gap↑)", -1.0, 1.0, -0.15, 0.05)
    dx  = st.number_input("x-step", 0.01, 0.50, 0.05, 0.01)
    if mode == "Ternary A–B–C":
        dy = st.number_input("y-step", 0.01, 0.50, 0.05, 0.01)

    if st.button("🗑 Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()

    st.caption(f"⚙️  Build SHA: {st.secrets.get('GIT_SHA','dev')} • "
               f"🕒 {datetime.datetime.now():%Y-%m-%d %H:%M}")

# ── Cached runner wrappers (binary & ternary) ───────────────────────────────
@st.cache_data(show_spinner="⏳  Screening …", max_entries=20)
def _run_binary(*args, **kws):
    return screen_binary(*args, **kws)

@st.cache_data(show_spinner="⏳  Screening …", max_entries=10)
def _run_ternary(*args, **kws):
    return screen_ternary(*args, **kws)

# ── Control buttons ─────────────────────────────────────────────────────────
col_run, col_prev = st.columns([3, 1])
do_run   = col_run.button("▶ Run screening", type="primary")
do_prev  = col_prev.button("⏪ Previous", disabled=not st.session_state.history)

# ── Retrieve previous result ────────────────────────────────────────────────
if do_prev:
    st.session_state.history.pop()          # discard current
    prev = st.session_state.history[-1]     # get last
    mode, df = prev["mode"], prev["df"]
    (A, B, rh, temp, (bg_lo, bg_hi),
     bow, dx) = (prev[k] for k in
                 ("A","B","rh","temp","bg","bow","dx"))
    if mode == "Ternary A–B–C":
        C, dy = prev["C"], prev["dy"]
    st.success("Showing previous result")

# ── Run a fresh screen ──────────────────────────────────────────────────────
elif do_run:
    try:
        _summary(A, [])  # quick probe to ensure formula is valid
        _summary(B, [])
        if mode == "Ternary A–B–C":
            _summary(C, [])
    except Exception as e:
        st.error(f"❌  Formula error / MP lookup failed: {e}")
        st.stop()

    if mode == "Binary A–B":
        df = _run_binary(
            A, B, rh, temp,
            (bg_lo, bg_hi), bow, dx
        )
    else:
        df = _run_ternary(
            A, B, C, rh, temp,
            (bg_lo, bg_hi),
            bows={"AB": bow, "AC": bow, "BC": bow},
            dx=dx, dy=dy
        )

    st.session_state.history.append({
        "mode": mode, "A": A, "B": B, "rh": rh, "temp": temp,
        "bg": (bg_lo, bg_hi), "bow": bow, "dx": dx,
        "df": df, **({"C": C, "dy": dy} if mode.startswith("Ternary") else {})
    })

# ── If nothing to show yet ──────────────────────────────────────────────────
elif not st.session_state.history:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

# ── Current dataframe ───────────────────────────────────────────────────────
df = st.session_state.history[-1]["df"]

# ── Tabs: table · plot · download ───────────────────────────────────────────
tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table", "📈 Plot", "📥 Download"])

# ── Table view ──────────────────────────────────────────────────────────────
with tab_tbl:
    st.markdown("### Run parameters")
    param_rows = [
        ("Humidity [%]", rh),
        ("Temperature [°C]", temp),
        ("Gap window [eV]", f"{bg_lo:.2f} – {bg_hi:.2f}"),
        ("Bowing [eV]", bow),
        ("x-step", dx),
    ]
    if mode.startswith("Ternary"):
        param_rows.append(("y-step", dy))
    st.table(pd.DataFrame(param_rows, columns=["Parameter", "Value"]))

    st.markdown("### Candidate results")
    st.dataframe(df, use_container_width=True, height=400)

# ── Plot view ───────────────────────────────────────────────────────────────
with tab_plot:
    if mode.startswith("Binary"):
        have = set(df.columns)
        needed = {"Eg", "Ehull", "score"}
        if not needed.issubset(have):
            st.warning("Missing columns for binary plot.")
        else:
            fig = px.scatter(
                df, x="Ehull", y="Eg", color="score",
                color_continuous_scale="Turbo",
                hover_data=df.columns, width=1150, height=780
            )
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color="black")))
            st.plotly_chart(fig, use_container_width=True)
    else:
        needed = {"x", "y", "score"}
        if not needed.issubset(df.columns):
            st.warning("Missing columns for ternary plot.")
        else:
            fig3d = px.scatter_3d(
                df, x="x", y="y", z="score",
                color="score", color_continuous_scale="Turbo",
                hover_data=df.columns, width=1150, height=820
            )
            fig3d.update_traces(marker=dict(size=4, line=dict(width=0.5, color="black")))
            st.plotly_chart(fig3d, use_container_width=True)

# ── Download view ───────────────────────────────────────────────────────────
@@ with tab_dl:
-    top = df.iloc[0]
-    label = (top.formula if mode.startswith("Binary")
-             else f"{A}+{B}+{C} (x={top.x:.2f}, y={top.y:.2f})")
+    top = df.iloc[0]
+    if mode.startswith("Binary"):
+        label = top["formula"]
+    else:
+        label = f"{A}+{B}+{C} (x={top['x']:.2f}, y={top['y']:.2f})"

    st.download_button(
        "📥 Download CSV",
        df.to_csv(index=False).encode(),
        "EnerMat_results.csv", "text/csv"
    )

    top = df.iloc[0]
    label = (top.formula if mode.startswith("Binary")
             else f"{A}+{B}+{C} (x={top.x:.2f}, y={top.y:.2f})")

    txt = (f"EnerMat auto-report  {datetime.date.today()}\n"
           f"Top candidate   : {label}\n"
           f"Band-gap [eV]   : {top.Eg}\n"
           f"Ehull  [eV/atom]: {top.Ehull}\n"
           f"Eox   [eV/Sn]   : {getattr(top,'Eox','N/A')}\n"
           f"Score           : {top.score}\n")
    st.download_button("📄 Download TXT", txt, "EnerMat_report.txt", "text/plain")

    # DOCX
    doc = Document()
    doc.add_heading("EnerMat Report", 0)
    doc.add_paragraph(f"Date: {datetime.date.today()}")
    doc.add_paragraph(f"Top candidate: {label}")
    tbl = doc.add_table(rows=1, cols=2)
    hdr = tbl.rows[0].cells
    hdr[0].text, hdr[1].text = "Property", "Value"
    for k in ("Eg", "Ehull", "Eox", "score"):
        if k in top:
            row = tbl.add_row()
            row.cells[0].text, row.cells[1].text = k, str(top[k])
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    st.download_button(
        "📝 Download DOCX", buf, "EnerMat_report.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
