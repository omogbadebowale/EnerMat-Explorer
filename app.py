"""
EnerMat Perovskite Explorer – Streamlit front-end
Version : v9.6.1   (2025-06-24)
"""

import io, os, datetime
import streamlit as st
import pandas as pd
import plotly.express as px, plotly.graph_objects as go

from backend.perovskite_utils import (
    mix_abx3 as screen,
    screen_ternary,
    END_MEMBERS,
    fetch_mp_data as _summary,
)

# ── MP key check (front-end fail-fast) ───────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    st.error("🛑 MP_API_KEY missing or malformed in Streamlit secrets.")
    st.stop()

# ── Streamlit page config ───────────────────────────────────────────────
st.set_page_config(page_title="EnerMat Explorer", layout="wide")
st.title("🔬 **EnerMat Perovskite Explorer v9.6.1**")

# ── Session-state history ───────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar controls ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose screening type", ["Binary A–B", "Ternary A–B–C"])

    st.header("End-members")
    # default to Pb-free trio
    preset_A = st.selectbox("Preset A", END_MEMBERS, index=0)   # CsSnBr3
    preset_B = st.selectbox("Preset B", END_MEMBERS, index=1)   # CsSnCl3
    custom_A = st.text_input("Custom A (optional)").strip()
    custom_B = st.text_input("Custom B (optional)").strip()
    A, B = custom_A or preset_A, custom_B or preset_B

    if mode == "Ternary A–B–C":
        preset_C = st.selectbox("Preset C", END_MEMBERS, index=2)  # CsSnI3
        custom_C = st.text_input("Custom C (optional)").strip()
        C = custom_C or preset_C

    st.header("Environment")
    rh   = st.slider("Humidity [%]",      0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)

    st.header("Target Band Gap [eV]")
    bg_lo, bg_hi = st.slider("Gap window", 0.5, 3.0, (1.0, 1.4), 0.01)

    st.header("Model settings")
    bow = st.number_input("Bowing [eV]", 0.0, 1.0, 0.30, 0.05)
    dx  = st.number_input("x-step", 0.01, 0.50, 0.05, 0.01)
    if mode == "Ternary A–B–C":
        dy = st.number_input("y-step", 0.01, 0.50, 0.05, 0.01)

    if st.button("🗑 Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()

    st.caption("© 2025 Dr Gbadebo Taofeek Yusuf")

# ── Cached binary screen runner ─────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Running binary...", max_entries=20)
def run_screen(**kwargs):
    return screen(**kwargs)

# ── Execution control ───────────────────────────────────────────────────
col_run, col_back = st.columns([3,1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=not st.session_state.history)

if do_back:
    st.session_state.history.pop()
    prev = st.session_state.history[-1]
    mode, df = prev["mode"], prev["df"]
    A,B,rh,temp,bg_lo,bg_hi,bow,dx = prev["A"],prev["B"],prev["rh"],prev["temp"],*prev["bg"],prev["bow"],prev["dx"]
    if mode == "Ternary A–B–C": C,dy = prev["C"],prev["dy"]
    st.success("Showing previous result")

elif do_run:
    try:
        _ = _summary(A, ["band_gap"]) and _summary(B, ["band_gap"])  # quick sanity
        if mode == "Ternary A–B–C": _ = _summary(C, ["band_gap"])
    except Exception as e:
        st.error(f"❌ Materials-Project error: {e}"); st.stop()

    if mode == "Binary A–B":
        df = run_screen(
            formula_A=A, formula_B=B,
            rh=rh, temp=temp,
            bg_window=(bg_lo,bg_hi), bowing=bow, dx=dx
        )
    else:
        df = screen_ternary(
            A=A, B=B, C=C, rh=rh, temp=temp,
            bg=(bg_lo,bg_hi),
            bows={"AB":bow,"AC":bow,"BC":bow},
            dx=dx, dy=dy
        )

    if df.empty:
        st.warning("No valid compositions found – widen your parameters."); st.stop()

    # store run in history (cap at 10)
    st.session_state.history.append({
        "mode": mode, "A":A, "B":B, "rh":rh, "temp":temp,
        "bg":(bg_lo,bg_hi), "bow":bow, "dx":dx, "df":df,
        **({"C":C, "dy":dy} if mode.startswith("Ternary") else {})
    })
    if len(st.session_state.history) > 10:
        st.session_state.history.pop(0)

else:
    if not st.session_state.history:
        st.info("Press **Run screening** to begin."); st.stop()
    prev = st.session_state.history[-1]; df,mode = prev["df"],prev["mode"]

# Always sort so df.iloc[0] is best
df = df.sort_values("score", ascending=False).reset_index(drop=True)

# ── UI tabs ─────────────────────────────────────────────────────────────
tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table","📈 Plot","📥 Download"])

with tab_tbl:
    st.markdown("**Run parameters**")
    rows = [
        ("Humidity [%]", rh), ("Temperature [°C]", temp),
        ("Gap window [eV]", f"{bg_lo:.2f}–{bg_hi:.2f}"),
        ("Bowing [eV]", bow), ("x-step", dx)
    ]
    if mode.startswith("Ternary"): rows.append(("y-step", dy))
    st.table(pd.DataFrame(rows, columns=["Parameter","Value"]))

    st.subheader("Candidate results")
    st.dataframe(df, use_container_width=True, height=400)

with tab_plot:
    if mode == "Binary A–B":
        if not {"stability","Eg","score"}.issubset(df.columns):
            st.error("Required columns missing for plot."); st.stop()
        fig = px.scatter(
            df, x="stability", y="Eg", color="score",
            color_continuous_scale="Turbo",
            hover_data=["formula","x","Eg","stability","score"],
            width=1200, height=800
        )
        fig.update_traces(marker=dict(size=12,opacity=0.9,line=dict(width=1,color="black")))
        st.plotly_chart(fig, use_container_width=True)

    else:
        fig3d = px.scatter_3d(
            df, x="x", y="y", z="score", color="score",
            color_continuous_scale="Turbo",
            hover_data={"x":True,"y":True,"Eg":True,"score":True},
            width=1200, height=900
        )
        fig3d.update_traces(marker=dict(size=5,opacity=0.9,line=dict(width=1,color="black")))
        st.plotly_chart(fig3d, use_container_width=True)

with tab_dl:
    # CSV
    st.download_button("📥 Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "EnerMat_results.csv","text/csv"
    )

    # Plain-text report
    top = df.iloc[0]
    if "formula" in top:
        label = top.formula
    else:
        label = f"{A}-{B}-{C} x={top.x:.2f} y={top.y:.2f}"

    lines = [
        f"EnerMat report ({datetime.date.today()})",
        f"Top candidate : {label}",
        f"Band-gap     : {top.Eg}",
    ]
    if "stability" in top:
        lines.append(f"Stability    : {top.stability}")
    lines.append(f"Score        : {top.score}")
    txt = "\n".join(lines) + "\n"

    st.download_button("📄 Download TXT", txt,
        "EnerMat_report.txt","text/plain")

    # DOCX
    import docx, io
    doc = docx.Document()
    doc.add_heading("EnerMat Report", level=0)
    for ln in lines: doc.add_paragraph(ln)
    buf = io.BytesIO(); doc.save(buf); buf.seek(0)
    st.download_button("📝 Download DOCX", buf,
        "EnerMat_report.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
