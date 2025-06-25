import io, os, datetime
import streamlit as st
import pandas as pd
import plotly.express as px, plotly.graph_objects as go
from docx import Document

# ── EnerMat backend ───────────────────────────────────────────────────────────
from backend.perovskite_utils import (
    mix_abx3 as screen, screen_ternary,
    END_MEMBERS, fetch_mp_data as _summary
)

# ── API key sanity check ──────────────────────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    st.error("🛑  set a valid 32-character MP_API_KEY in secrets!")
    st.stop()

# ── Page set-up ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.7")

# Session storage for back/forward
if "history" not in st.session_state:
    st.session_state.history = []

# ───────────── Sidebar ─────────────
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose screening type", ["Binary A–B", "Ternary A–B–C"])

    st.header("End-members")
    preset_A = st.selectbox("Preset A", END_MEMBERS, 0)
    preset_B = st.selectbox("Preset B", END_MEMBERS, 1)
    custom_A = st.text_input("Custom A", "").strip()
    custom_B = st.text_input("Custom B", "").strip()
    A, B = custom_A or preset_A, custom_B or preset_B

    if mode == "Ternary A–B–C":
        preset_C = st.selectbox("Preset C", END_MEMBERS, 2)
        custom_C = st.text_input("Custom C", "").strip()
        C = custom_C or preset_C

    st.header("Environment")
    rh   = st.slider("Humidity [%]", 0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)

    st.header("Band-gap target")
    bg_lo, bg_hi = st.slider("Window [eV]", .5, 3.0, (1.0,1.4), .01)

    st.header("Model")
    bow = st.number_input("Bowing [eV]", .0, 1.0, .30, .05)
    dx  = st.number_input("x-step", .01, .5, .05, .01)
    if mode == "Ternary A–B–C":
        dy = st.number_input("y-step", .01, .5, .05, .01)

    if st.button("🗑 Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()

    st.caption(f"⚙ Git @ {st.secrets.get('GIT_SHA','dev')}")
    st.caption("© 2025 Dr G. T. Yusuf")

# ───────────── Cached runner ─────────────
@st.cache_data(show_spinner="⏳ Running …", max_entries=20)
def run_screen(a,b,rh,temp,win,bow,dx):
    return screen(formula_A=a, formula_B=b,
                  rh=rh,temp=temp,bg_window=win,
                  bowing=bow,dx=dx)

# ───────────── Controls ─────────────
col_run, col_prev = st.columns([3,1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_prev = col_prev.button("⏪ Previous", disabled=not st.session_state.history)

if do_prev:
    st.session_state.history.pop()
    prev = st.session_state.history[-1]
    mode = prev["mode"]; df = prev["df"]
    A,B,rh,temp,bg_lo,bg_hi,bow,dx = prev["A"],prev["B"],prev["rh"],prev["temp"],*prev["bg"],prev["bow"],prev["dx"]
    if mode=="Ternary A–B–C": C,dy = prev["C"],prev["dy"]
    st.success("Showing previous run")

elif do_run:
    for f in [A,B] + ([C] if mode=="Ternary A–B–C" else []):
        if not _summary(f, ["band_gap"]):
            st.error(f"❌ Unknown formula “{f}”")
            st.stop()

    if mode=="Binary A–B":
        df = run_screen(A,B,rh,temp,(bg_lo,bg_hi),bow,dx)
    else:
        df = screen_ternary(A,B,C,rh,temp,(bg_lo,bg_hi),
                            {"AB":bow,"AC":bow,"BC":bow},
                            dx=dx,dy=dy)

    entry = dict(mode=mode,A=A,B=B,rh=rh,temp=temp,bg=(bg_lo,bg_hi),
                 bow=bow,dx=dx,df=df)
    if mode=="Ternary A–B–C": entry.update(C=C,dy=dy)
    st.session_state.history.append(entry)

elif st.session_state.history:
    df = st.session_state.history[-1]["df"]
else:
    st.info("Press ▶ Run screening to start.")
    st.stop()

# ───────────── Tabs ─────────────
tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table","📈 Plot","📥 Download"])

# === TABLE ===
with tab_tbl:
    st.dataframe(df,use_container_width=True,height=400)

# === PLOT ===
with tab_plot:
    if mode=="Binary A–B":
        fig = px.scatter(df,x="stability",y="Eg",color="score",
                         color_continuous_scale="Turbo",
                         hover_data=["formula","x","Eg","stability","gap_score","score"],
                         height=750)
        fig.update_traces(marker=dict(size=12,line=dict(width=1,color="black")))
        st.plotly_chart(fig,use_container_width=True)

        # optional humidity-curve
        if st.button("📈 Show S vs RH curve"):
            from backend.perovskite_utils import mix_abx3
            def humidity_curve(a,b,x_fixed):
                rec=[]
                for r in range(0,101,10):
                    d=mix_abx3(a,b,r,temp,(bg_lo,bg_hi),bow,dx=0.01)
                    row=d.iloc[(d.x-x_fixed).abs().argsort().iloc[0]]
                    rec.append({"RH %":r,"S":row.score})
                return pd.DataFrame(rec)
            curve = humidity_curve(A,B,0.30)
            st.plotly_chart(px.line(curve,x="RH %",y="S",markers=True,
                                    title=f"S vs RH at x = 0.30 ({A}/{B})"),
                            use_container_width=True)

    else:   # ternary 3-D
        fig3 = px.scatter_3d(df,x="x",y="y",z="score",color="score",
                             color_continuous_scale="Turbo",
                             hover_data=["x","y","Eg","gap_score","score"],
                             height=800)
        st.plotly_chart(fig3,use_container_width=True)

# === DOWNLOADS ===
with tab_dl:
    st.download_button("📥 CSV", df.to_csv(index=False).encode(),
                       "EnerMat_results.csv","text/csv")

    top = df.iloc[0]
    top_label = (top.formula if mode=="Binary A–B"
                 else f"{A}-{B}-{C} x={top.x:.2f} y={top.y:.2f}")

    txt = f"""EnerMat report ({datetime.date.today()})
Top candidate : {top_label}
Band-gap      : {top.Eg}
Stability     : {getattr(top,'stability','N/A')}
Gap fitness   : {getattr(top,'gap_score','N/A')}
Composite S   : {top.score}
"""
    st.download_button("📄 TXT", txt, "EnerMat_report.txt","text/plain")

    doc = Document(); doc.add_heading("EnerMat Report",0)
    doc.add_paragraph(f"Date: {datetime.date.today()}")
    doc.add_paragraph(f"Top candidate: {top_label}")
    tbl=doc.add_table(rows=1,cols=2)
    hdr=tbl.rows[0].cells; hdr[0].text="Property"; hdr[1].text="Value"
    for k,v in [("Band-gap (eV)",top.Eg),
                ("Stability",getattr(top,'stability','–')),
                ("Gap fitness",getattr(top,'gap_score','–')),
                ("Composite S",top.score)]:
        r=tbl.add_row(); r.cells[0].text=k; r.cells[1].text=str(v)
    buf=io.BytesIO(); doc.save(buf); buf.seek(0)
    st.download_button("📝 DOCX",buf,"EnerMat_report.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
