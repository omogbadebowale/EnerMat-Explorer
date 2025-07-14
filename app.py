# app.py  –  EnerMat Perovskite Explorer v9.6  (2025-07-14, Ge-ready)
# ----------------------------------------------------------------------
import io, os, datetime, functools, pathlib
import streamlit as st
import pandas as pd
import plotly.express as px
from docx import Document

# ─── Materials-Project key ─────────────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    st.error("🛑  Add a valid 32-character MP_API_KEY in Streamlit Secrets")
    st.stop()

# ─── Local helpers ────────────────────────────────────────────────────
from backend.perovskite_utils import (
    mix_abx3 as screen_binary,
    screen_ternary,
)
from materials import load_end_members          # <–– single source of truth
END_MEMBERS = list(load_end_members())

# ─── Page config / banner ─────────────────────────────────────────────
st.set_page_config("EnerMat Explorer", layout="wide")
st.title("🔬  EnerMat **Perovskite** Explorer v9.6")

# ─── Session state ----------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Sidebar controls ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose screening type", ["Binary A-B", "Ternary A-B-C"])

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

    st.header("Environment")
    rh   = st.slider("Humidity [%]",      0, 100, 50)
    temp = st.slider("Temperature [°C]",-20, 100, 25)

    st.header("Target band-gap [eV]")
    bg_lo, bg_hi = st.slider("Gap window", 0.50, 3.00, (1.00, 1.40), 0.01)

    st.header("Model settings")
    bow = st.number_input("Bowing (eV, negative ⇒ gap↑)",
                          -1.0, 1.0, -0.15, 0.05)
    dx  = st.number_input("x-step", 0.01, 0.50, 0.05, 0.01)
    if mode.startswith("Ternary"):
        dy = st.number_input("y-step", 0.01, 0.50, 0.05, 0.01)

    # ── Ge slider (only if enabled in config) ─────────────────────────
    if st.session_state.get("b_site_mixing", True):
        z = st.slider("Ge fraction z", 0.00, 0.30, 0.10, 0.05,
                      help="B-site Ge²⁺ fraction in CsSn₁₋zGe_z(Br,Cl)₃")
    else:
        z = 0.0

    if st.button("🗑 Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()

    st.caption(f"⚙️ Build SHA : {st.secrets.get('GIT_SHA','dev')}  •  "
               f"🕒 {datetime.datetime.now():%Y-%m-%d %H:%M}")

# ─── Cached wrappers --------------------------------------------------
@st.cache_data(show_spinner="⏳ Screening …", max_entries=20)
def _run_binary(*a, **k):   return screen_binary(*a, **k)

@st.cache_data(show_spinner="⏳ Screening …", max_entries=10)
def _run_ternary(*a, **k):  return screen_ternary(*a, **k)

# ─── Control buttons --------------------------------------------------
col_run, col_prev = st.columns([3,1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_prev = col_prev.button("⏪ Previous", disabled=not st.session_state.history)

# ─── Retrieve previous result ----------------------------------------
if do_prev:
    st.session_state.history.pop()
    prev = st.session_state.history[-1]
    df   = prev["df"]
    mode = prev["mode"]
    st.success("Showing previous result")

# ─── New run ----------------------------------------------------------
elif do_run:
    # Quick formula validation
    for fml in (A,B,*(C,) if mode.startswith("Ternary") else ()):
        if fml not in END_MEMBERS:
            st.error(f"❌  Unknown formula: {fml}")
            st.stop()

    if mode.startswith("Binary"):
        df = _run_binary(A,B,rh,temp,(bg_lo,bg_hi),bow,dx,z=z)
    else:
        df = _run_ternary(A,B,C,rh,temp,(bg_lo,bg_hi),
                          bows={"AB":bow,"AC":bow,"BC":bow},
                          dx=dx,dy=dy,z=z)

    st.session_state.history.append({"mode":mode,"df":df})

# ─── No data yet ------------------------------------------------------
elif not st.session_state.history:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

# ─── Current DataFrame -----------------------------------------------
df   = st.session_state.history[-1]["df"]
mode = st.session_state.history[-1]["mode"]

# ─── Tabs -------------------------------------------------------------
tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table","📈 Plot","📥 Download"])

with tab_tbl:
    st.dataframe(df, use_container_width=True, height=420)

with tab_plot:
    if mode.startswith("Binary") and {"Ehull","Eg"}.issubset(df.columns):
        fig = px.scatter(df,x="Ehull",y="Eg",color="score",
                         color_continuous_scale="Turbo",
                         hover_data=df.columns, height=780)
        fig.update_traces(marker_size=9,marker_line_width=1,
                          marker_line_color="black")
        st.plotly_chart(fig,use_container_width=True)
    elif mode.startswith("Ternary") and {"x","y","score"}.issubset(df.columns):
        fig = px.scatter_3d(df,x="x",y="y",z="score",color="score",
                            color_continuous_scale="Turbo",
                            hover_data=df.columns,height=820)
        st.plotly_chart(fig,use_container_width=True)

with tab_dl:
    st.download_button("📥 Download CSV",
                       df.to_csv(index=False).encode(),
                       "EnerMat_results.csv", "text/csv")

# ─── ONE robust label builder (no duplicates!) ───────────────────────
top = df.iloc[0]                             # first (best) row – Series
formula = str(top["formula"])
coords  = [f"{c}={top[c]:.2f}" for c in ("x","y","z","ge_frac") 
           if c in top and pd.notna(top[c])]
label = f"{formula} ({', '.join(coords)})" if coords else formula

# ─── TXT + DOCX auto-report ------------------------------------------
txt = ( "EnerMat auto-report  "
        f"{datetime.date.today()}\n"
        f"Top candidate   : {label}\n"
        f"Band-gap [eV]   : {top['Eg']}\n"
        f"Ehull  [eV/at.] : {top['Ehull']}\n"
        f"Eox_e [eV/e⁻]   : {top.get('Eox_e','N/A')}\n"
        f"Score           : {top['score']}\n" )
st.download_button("📄 Download TXT", txt, "EnerMat_report.txt", "text/plain")

doc = Document()
doc.add_heading("EnerMat Report", 0)
doc.add_paragraph(f"Date : {datetime.date.today()}")
doc.add_paragraph(f"Top candidate : {label}")
tbl = doc.add_table(rows=1,cols=2)
hdr = tbl.rows[0].cells
hdr[0].text, hdr[1].text = "Property","Value"
for k in ("Eg","Ehull","Eox_e","score"):
    if k in top:
        row = tbl.add_row()
        row.cells[0].text, row.cells[1].text = k, str(top[k])
buf = io.BytesIO(); doc.save(buf); buf.seek(0)
st.download_button("📝 Download DOCX", buf, "EnerMat_report.docx",
                   "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
