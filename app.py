"""EnerMat Perovskite Explorer – v9.6 – Streamlit UI (2025‑07‑13)"""
import datetime, io, os
import streamlit as st
import pandas as pd
import plotly.express as px
from docx import Document

from backend.perovskite_utils import (
    mix_abx3   as screen_binary,
    screen_ternary,
    END_MEMBERS,
)

API_KEY=os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY","")
if len(API_KEY)!=32:
    st.error("🛑  Set a valid 32‑character MP_API_KEY in the Secrets tab.")
    st.stop()

st.set_page_config(page_title="EnerMat Explorer",layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.6")

# ── sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Mode")
    mode=st.radio("Choose screening type",["Binary A–B","Ternary A–B–C"])

    st.header("End‑members")
    preset_A=st.selectbox("Preset A",END_MEMBERS,index=0)
    preset_B=st.selectbox("Preset B",END_MEMBERS,index=1)
    custom_A=st.text_input("Custom A (optional)","").strip()
    custom_B=st.text_input("Custom B (optional)","").strip()
    A=custom_A or preset_A; B=custom_B or preset_B
    if mode=="Ternary A–B–C":
        preset_C=st.selectbox("Preset C",END_MEMBERS,index=2)
        custom_C=st.text_input("Custom C (optional)","").strip()
        C=custom_C or preset_C

    st.header("Environment")
    rh  = st.slider("Humidity [%]",0,100,50)
    temp= st.slider("Temperature [°C]",-20,100,25)

    st.header("Target band‑gap [eV]")
    bg_lo,bg_hi=st.slider("Gap window [eV]",0.5,3.0,(1.0,1.4),0.01)

    st.header("Model settings")
    bow = st.number_input("Bowing [eV]",-1.0,1.0,-0.15,0.05)
    dx  = st.number_input("x‑step",0.01,0.5,0.05,0.01)
    if mode=="Ternary A–B–C":
        dy = st.number_input("y‑step",0.01,0.5,0.05,0.01)

# ── cached wrappers ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Running binary…",max_entries=25)
def _run_binary(*args,**kws):
    return screen_binary(*args,**kws)

@st.cache_data(show_spinner="⏳ Running ternary…",max_entries=10)
def _run_ternary(*args,**kws):
    return screen_ternary(*args,**kws)

# ── execute run ───────────────────────────────────────────────────────────
if st.button("▶ Run screening",type="primary"):
    if mode=="Binary A–B":
        df=_run_binary(A,B,rh,temp,(bg_lo,bg_hi),bow,dx)
    else:
        bows={"AB":bow,"AC":bow,"BC":bow}
        df=_run_ternary(A,B,C,rh,temp,(bg_lo,bg_hi),bows,dx,dy)
    if df.empty:
        st.error("❌ No data returned – check formulas and API key.")
        st.stop()

    # ─ results table ─
    st.subheader("Run parameters")
    rp=pd.DataFrame({"Parameter":["Humidity [%]","Temperature [°C]","Gap window [eV]","Bowing [eV]","x‑step"]+(["y‑step"] if mode=="Ternary A–B–C" else []),
                    "Value"     :[rh,temp,f"{bg_lo:.2f}–{bg_hi:.2f}",bow,dx]+([dy] if mode=="Ternary A–B–C" else [])})
    st.table(rp)

    st.subheader("Candidate results")
    st.dataframe(df,use_container_width=True,height=420)

    # ─ plot tab ─
    plot=st.toggle("Show plot")
    if plot:
        if mode=="Binary A–B":
            fig=px.scatter(df,x="Eox",y="Eg",color="score",hover_data=["formula","Ehull"],color_continuous_scale="Turbo")
            st.plotly_chart(fig,use_container_width=True)
        else:
            fig=px.scatter_3d(df,x="x",y="y",z="Eox",color="score",hover_data=["Eg","Ehull","formula"],color_continuous_scale="Turbo")
            st.plotly_chart(fig,use_container_width=True)

    # ─ simple report download ─
    top=df.iloc[0]
    label=(top.formula if mode=="Binary A–B" else f"{A}+{B}+{C} (x={top.x:.2f}, y={top.y:.2f})")
    txt=(f"EnerMat auto‑report  {datetime.date.today()}\n"
         f"Top candidate : {label}\n"
         f"Band‑gap      : {top.Eg} eV\n"
         f"Ehull         : {top.Ehull} eV atom⁻¹\n"
         f"ΔE_ox         : {top.Eox} eV Sn⁻¹\n"
         f"Score         : {top.score}\n")
    st.download_button("📄 TXT report",txt,"EnerMat_report.txt","text/plain")

    doc=Document(); doc.add_heading("EnerMat Report",0)
    doc.add_paragraph(f"Date: {datetime.date.today()}")
    doc.add_paragraph(f"Top candidate: {label}")
    tbl=doc.add_table(rows=1,cols=2); hdr=tbl.rows[0].cells
    hdr[0].text,hdr[1].text="Property","Value"
    for k,v in [("Band‑gap",top.Eg),("Ehull",top.Ehull),("ΔE_ox",top.Eox),("Score",top.score)]:
        r=tbl.add_row(); r.cells[0].text,r.cells[1].text=k,str(v)
    buf=io.BytesIO(); doc.save(buf); buf.seek(0)
    st.download_button("📝 DOCX report",buf,"EnerMat_report.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

else:
    st.info("⬅️  Enter parameters, then click ▶ Run screening.")
