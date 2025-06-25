# app.py – EnerMat Explorer  v9.7-patched  (25 Jun 2025)
import io, os, datetime as dt
import streamlit as st, pandas as pd
import plotly.express as px, plotly.graph_objects as go
from docx import Document

# ——— MP key ————————————————————————————
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    st.error("Need valid 32-char MP_API_KEY in secrets or env")
    st.stop()

# ——— backend ————————————————————————————
from backend.perovskite_utils import (
    mix_abx3 as run_binary, screen_ternary,
    END_MEMBERS, fetch_mp_data as _mp,
)

st.set_page_config("EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.7")

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

# ——— Sidebar ————————————————————————————
with st.sidebar:
    mode = st.radio("Choose screening type", ["Binary A–B", "Ternary A–B–C"])
    preset_A = st.selectbox("Preset / custom A", END_MEMBERS, 0);  custom_A = st.text_input("", key="A")
    preset_B = st.selectbox("Preset / custom B", END_MEMBERS, 1);  custom_B = st.text_input("", key="B")
    A, B = (custom_A or preset_A).strip(), (custom_B or preset_B).strip()
    if mode == "Ternary A–B–C":
        preset_C = st.selectbox("Preset / custom C", END_MEMBERS, 2); custom_C = st.text_input("", key="C")
        C = (custom_C or preset_C).strip()

    rh   = st.slider("Humidity [%]", 0,100,50)
    temp = st.slider("Temperature [°C]", -20,100,25)
    bg_lo, bg_hi = st.slider("Target gap [eV]", 0.5,3.0,(1.0,1.4),0.01)
    bow  = st.number_input("Bowing [eV]", 0.0,1.0,0.30,0.05)
    dx   = st.number_input("x-step", 0.01,0.5,0.05,0.01)
    dy   = st.number_input("y-step", 0.01,0.5,0.05,0.01) if mode=="Ternary A–B–C" else None
    if st.button("🗑 Clear history"):  st.session_state.history.clear(); st.experimental_rerun()

# ——— Cached binary helper ——————————————————
@st.cache_data(show_spinner="⏳ Screening…", max_entries=20)
def _run_binary(a,b,rh,temp,lo,hi,bow,dx):
    return run_binary(a,b,rh,temp,(lo,hi),bow,dx)

# ——— Run button row ————————————————————
col_run, col_back = st.columns([3,1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=not st.session_state.history)

# ——— Core logic ————————————————————————
if do_back:
    st.session_state.history.pop()
    params = st.session_state.history[-1]
    df = params["df"]
    for k,v in params.items(): globals()[k]=v   # restore sidebar vars
    st.success("Showing previous result")

elif do_run:
    # sanity check formulas
    docs_ok = all(_mp(f,["band_gap"]) for f in ([A,B] if mode=="Binary A–B" else [A,B,C]))
    if not docs_ok: st.error("Invalid formula"); st.stop()
    df = (_run_binary(A,B,rh,temp,bg_lo,bg_hi,bow,dx) if mode=="Binary A–B"
          else screen_ternary(A,B,C,rh,temp,(bg_lo,bg_hi),{"AB":bow,"AC":bow,"BC":bow},dx,dy))
    if df.empty: st.warning("No data returned – check parameters"); st.stop()
    st.session_state.history.append(dict(mode=mode,A=A,B=B,C=C if mode!="Binary A–B" else None,
                                         rh=rh,temp=temp,bg=(bg_lo,bg_hi),bow=bow,dx=dx,dy=dy,df=df))

elif st.session_state.history:
    df = st.session_state.history[-1]["df"]
else:
    st.info("Press ▶ Run screening")
    st.stop()

# ——— Tabs —————————————————————————
tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table","📈 Plot","📥 Download"])

# ——— Table tab ——————————————————————
with tab_tbl:
    st.dataframe(df, use_container_width=True, height=420)

# ——— Plot tab ——————————————————————
with tab_plot:
    if mode=="Binary A–B":
        fig = px.scatter(df, x="stability", y="Eg", color="score",
                         color_continuous_scale="Turbo",
                         hover_data=["formula","x","Eg","stability","score"])
        fig.update_traces(marker=dict(size=11,line=dict(width=1,color="black")))
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig3d = px.scatter_3d(df, x="x",y="y",z="score", color="score",
                              color_continuous_scale="Turbo",
                              hover_data=["x","y","Eg","score"])
        fig3d.update_traces(marker=dict(size=4,line=dict(width=1,color="black")))
        st.plotly_chart(fig3d, use_container_width=True)

# ——— Download tab ————————————————————
with tab_dl:
    st.download_button("⬇ CSV", df.to_csv(index=False).encode(),
                       "EnerMat_results.csv", "text/csv")
    top = df.iloc[0]
    top_lbl = (top.formula if mode=="Binary A–B"
               else f"{A}-{B}-{C} x={top.x:.2f} y={top.y:.2f}")
    txt = (f"EnerMat report ({dt.date.today()})\n"
           f"Top candidate : {top_lbl}\n"
           f"Band-gap      : {top.Eg}\n"
           f"Stability     : {top.stability}\n"
           f"Gap factor    : {top.gap_score}\n"
           f"Composite S   : {top.score}\n")
    st.download_button("⬇ TXT", txt, "EnerMat_report.txt","text/plain")
