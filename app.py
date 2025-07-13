# app.py  –  EnerMat Perovskite Explorer  v9.6  (2025-07-13, oxide-ready)

from __future__ import annotations
import datetime, io, os, textwrap

import pandas as pd
import plotly.express as px
import streamlit as st
from docx import Document

# ─── back-end helpers ──────────────────────────────────────────────
from backend.perovskite_utils import (            # ← your 2025-07-13 backend
    mix_abx3          as screen_binary,
    screen_ternary    as screen_ternary,
    END_MEMBERS,
    fetch_mp_data     as _summary,
)

# ─── Streamlit page & session ─────────────────────────────────────
st.set_page_config("EnerMat Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.6")

if "history" not in st.session_state:
    st.session_state.history = []

# ─── sidebar – input widgets ──────────────────────────────────────
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

    if mode.startswith("Ternary"):
        preset_C = st.selectbox("Preset C", END_MEMBERS, index=2)
        custom_C = st.text_input("Custom C (optional)", "").strip()
        C = custom_C or preset_C

    st.header("Environment")
    rh   = st.slider("Humidity [%]",      0, 100, 50)
    temp = st.slider("Temperature [°C]",-20, 100, 25)

    st.header("Target band-gap [eV]")
    bg_lo, bg_hi = st.slider("Gap window [eV]", 0.5, 3.0, (1.0, 1.4), 0.01)

    st.header("Model settings")
    bow = st.number_input("Bowing (eV, neg ⇒ gap ↑)",-1.0, 1.0,-0.15, 0.05)
    dx  = st.number_input("x-step",0.01,0.50,0.05,0.01)
    if mode.startswith("Ternary"):
        dy = st.number_input("y-step",0.01,0.50,0.05,0.01)

    if st.button("🗑 Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()

# ─── cached wrappers (keep UI snappy) ─────────────────────────────
@st.cache_data(show_spinner="⏳ Binary screen …", max_entries=20)
def _run_binary(*args, **kws):
    return screen_binary(*args, **kws)

@st.cache_data(show_spinner="⏳ Ternary screen …", max_entries=20)
def _run_ternary(*args, **kws):
    return screen_ternary(*args, **kws)

# ─── run / back buttons ───────────────────────────────────────────
col_run, col_back = st.columns([3,1])
run  = col_run.button("▶ Run screening", type="primary")
back = col_back.button("⏪ Previous", disabled=not st.session_state.history)

if back:
    entry = st.session_state.history.pop()
    df    = entry["df"]
    mode  = entry["mode"]
    A,B,rh,temp,bg_lo,bg_hi,bow,dx = (entry[k] for k in
        ("A","B","rh","temp","bg_lo","bg_hi","bow","dx"))
    if mode.startswith("Ternary"):
        C,dy = entry["C"], entry["dy"]
    st.success("Showing previous result")

elif run:
    # minimal MP sanity-check so we fail fast
    try:
        _ = _summary(A,["band_gap"])
        _ = _summary(B,["band_gap"])
        if mode.startswith("Ternary"):
            _ = _summary(C,["band_gap"])
    except Exception as e:
        st.error(f"❌ Error querying Materials Project: {e}")
        st.stop()

    if mode.startswith("Binary"):
        df = _run_binary(A,B,rh,temp,(bg_lo,bg_hi), bow, dx)
    else:
        bows = dict(AB=bow, AC=bow, BC=bow)
        df = _run_ternary(A,B,C,rh,temp,(bg_lo,bg_hi), bows, dx, dy)

    # keep a snapshot
    st.session_state.history.append(dict(
        mode=mode, A=A, B=B, C=C if mode.startswith("Ternary") else "",
        rh=rh,temp=temp,bg_lo=bg_lo,bg_hi=bg_hi,bow=bow,dx=dx,dy=dy if mode.startswith("Ternary") else 0,
        df=df
    ))

elif st.session_state.history:
    df   = st.session_state.history[-1]["df"]
    mode = st.session_state.history[-1]["mode"]
else:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

# ─── main layout: 3 tabs ──────────────────────────────────────────
tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table","📈 Plot","📥 Download"])

# — table —
with tab_tbl:
    st.markdown("### Run parameters")
    params = {
        "Parameter": ["Humidity [%]","Temperature [°C]",
                      "Gap window [eV]","Bowing [eV]",
                      "x-step"] + (["y-step"] if mode.startswith("Ternary") else []),
        "Value":     [rh,temp,f"{bg_lo:.2f}–{bg_hi:.2f}",bow,dx] + ([dy] if mode.startswith("Ternary") else [])
    }
    st.table(pd.DataFrame(params))
    st.markdown("### Candidate results")
    st.dataframe(df,use_container_width=True,height=420)

# — plot —
with tab_plot:
    if mode.startswith("Binary"):
        fig = px.scatter(df,x="Eox",y="Eg",
                         size="score",color="score",
                         hover_data=["x","Eg","Ehull","Eox","score"],
                         color_continuous_scale="Turbo")
        st.plotly_chart(fig,use_container_width=True)
    else:
        fig = px.scatter_3d(df,x="x",y="y",z="score",
                            color="score",color_continuous_scale="Turbo",
                            hover_data=["x","y","Eg","Ehull","Eox","score"])
        fig.update_traces(marker=dict(size=5,opacity=0.9,line=dict(width=0.5,color="black")))
        st.plotly_chart(fig,use_container_width=True)

# — downloads —
with tab_dl:
    st.download_button("⬇️ CSV", df.to_csv(index=False).encode(),
                       "EnerMat_results.csv","text/csv")

    # simple TXT & DOCX summaries (robust for both modes)
    top = df.iloc[0]
    label = top.formula if "formula" in top else (
        f"{A}-{B}" if mode.startswith("Binary") else f"{A}-{B}-{C}")

    txt = textwrap.dedent(f"""
        EnerMat auto-report ({datetime.date.today()})
        Top candidate : {label}
        Band-gap [eV] : {top.Eg}
        Ehull  [eV]   : {top.Ehull}
        ΔEox  [eV/Sn] : {top.Eox}
        Score         : {top.score}
    """).strip()
    st.download_button("⬇️ TXT", txt, "EnerMat_report.txt","text/plain")

    doc = Document()
    doc.add_heading("EnerMat auto-report", level=1)
    for line in txt.splitlines():
        doc.add_paragraph(line)
    buf = io.BytesIO(); doc.save(buf); buf.seek(0)
    st.download_button("⬇️ DOCX", buf, "EnerMat_report.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
