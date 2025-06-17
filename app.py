import io, zipfile, uuid, datetime
import streamlit as st, plotly.express as px, pandas as pd
import nbformat as nbf
from docx import Document
from backend.perovskite_utils import screen, END_MEMBERS

st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.6")

# ── keep history in the session ---------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []        # list of (A,B,rh,…,df)

# ── sidebar inputs -----------------------------------------------------------
with st.sidebar:
    st.header("Environment")
    rh   = st.slider("Humidity [%]", 0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)
    bg_lo,bg_hi = st.slider("Target gap [eV]", 0.5, 3.0, (1.0,1.4),0.01)

    st.header("Parent formulas")
    A_pick = st.selectbox("Preset A", END_MEMBERS, index=0)
    B_pick = st.selectbox("Preset B", END_MEMBERS, index=1)
    A_custom = st.text_input("Custom A (optional)").strip()
    B_custom = st.text_input("Custom B (optional)").strip()
    A = A_custom or A_pick
    B = B_custom or B_pick

    st.header("Model")
    bow = st.number_input("Bowing [eV]",0.0,1.0,0.30,0.05)
    dx  = st.number_input("x-step",0.01,0.50,0.05,0.01)

    st.markdown("---")
    st.caption("© 2024 Gbadebo Taofeek Yusuf")

@st.cache_data(show_spinner="Monte-Carlo …")
def run_screen(**kw): return screen(**kw)

# ── run / back buttons -------------------------------------------------------
c_run, c_back = st.columns([3,1])
do_run   = c_run.button("▶ Run screening", type="primary")
do_back  = c_back.button("⏪ Previous", disabled=len(st.session_state["history"])<2)

if do_back:
    st.session_state["history"].pop()      # discard current
    _,_,_,_,prev_df = st.session_state["history"][-1]
    df = prev_df                            # show previous result
    st.success("Showing previous result")
elif do_run:
    df = run_screen(A=A,B=B,rh=rh,temp=temp,
                    bg=(bg_lo,bg_hi),bow=bow,dx=dx)
    if df.empty:
        st.error("No candidates – check formulas or widen window."); st.stop()
    st.session_state["history"].append((A,B,rh,temp,df))

elif st.session_state["history"]:
    # page reload – show most recent
    _,_,_,_,df = st.session_state["history"][-1]
else:
    st.info("Press **Run screening** to start."); st.stop()

# ── tabs: table / plot / download -------------------------------------------
tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table", "📈 Plot", "⬇ Download"])

with tab_tbl:
    st.dataframe(df, height=450, use_container_width=True)

with tab_plot:
    fig = px.scatter(
        df, x="stability", y="band_gap",
        error_x="stab_hi", error_x_minus=df["stability"]-df["stab_lo"],
        error_y="gap_high", error_y_minus=df["band_gap"]-df["gap_low"],
        size="lifetime", size_max=45,
        color="score", color_continuous_scale="viridis_r",
        hover_data=["form_score","env_pen"],
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_dl:
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("CSV", csv_bytes, "EnerMat_results.csv", "text/csv")

    top = df.iloc[0]
    report_txt = (
f" EnerMat Perovskite Explorer ({datetime.date.today()})\n"
f" Top candidate: {top['formula']}\n"
f" Gap  (eV): {top['band_gap']}\n"
f" Stability : {top['stability']}\n"
f" Env. pen. : {top['env_pen']}\n"
f" Score     : {top['score']}\n"
    )
    st.download_button("TXT report", report_txt,
                       "EnerMat_report.txt", "text/plain")

    doc = Document()
    doc.add_heading("EnerMat Perovskite Report", 0)
    doc.add_paragraph(f"Date: {datetime.date.today()}")
    doc.add_paragraph(f"Top candidate: {top['formula']}")
    table = doc.add_table(rows=1, cols=2)
    for k,v in [("Band-gap (eV)", top['band_gap']),
                ("Stability",     top['stability']),
                ("Formability",   top['form_score']),
                ("Env. penalty",  top['env_pen']),
                ("Composite",     top['score'])]:
        row=table.add_row(); row.cells[0].text=k; row.cells[1].text=str(v)
    b = io.BytesIO(); doc.save(b); b.seek(0)
    st.download_button("DOCX report", b, "EnerMat_report.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
