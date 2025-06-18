import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from docx import Document
from backend.perovskite_utils import screen, END_MEMBERS, _summary

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="EnerMat Perovskite Explorer v9.6")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("Environment")
humidity   = st.sidebar.slider("Humidity [%]",          0, 100,  50)
temperature= st.sidebar.slider("Temperature [°C]",   -20, 100,  25)

st.sidebar.markdown("---")
bg_lo, bg_hi = st.sidebar.slider(
    "Target gap [eV]", 0.0, 3.0, (0.8, 1.4), step=0.01
)

st.sidebar.markdown("---")
st.sidebar.header("Parent formulas")
# include FA/M Sn variants for lead-free screening
ALL_ENDS = ["CsPbBr3","CsSnBr3","CsSnCl3","CsPbI3","FASnBr3","MASnBr3"]
A_pick = st.sidebar.selectbox("Preset A", ALL_ENDS, index=0)
B_pick = st.sidebar.selectbox("Preset B", ALL_ENDS, index=1)
A = st.sidebar.text_input("Custom A (optional)", "").strip() or A_pick
B = st.sidebar.text_input("Custom B (optional)", "").strip() or B_pick

st.sidebar.markdown("---")
st.sidebar.header("Model knobs")
bow        = st.sidebar.number_input("Bowing [eV]",            0.0, 1.0, 0.30, 0.05)
dx         = st.sidebar.number_input("x-step",                 0.01,0.50, 0.05, 0.01)
sn_penalty = st.sidebar.slider(
    "Sn oxidation penalty (α)", 0.0, 1.0, 0.50, 0.05,
    help="Lower ⇒ less penalty on Sn²⁺ under humidity"
)

st.sidebar.caption("© 2025 Dr Gbadebo Taofeek Yusuf")
if st.sidebar.button("🗑 Clear history"):
    st.session_state.clear()
    st.experimental_rerun()

# ─── Keep history of runs ─────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Cached screening call ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Monte-Carlo sampling…")
def run_screen_cached(**kw):
    return screen(**kw)

# ─── Run / Back logic ─────────────────────────────────────────────────────────
col_run, col_back = st.columns([3,1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=len(st.session_state.history)<1)

if do_back:
    st.session_state.history.pop()
    A,B,humidity,temperature,df = st.session_state.history[-1]
    st.success("Showing previous result")
elif do_run:
    # validate MP data
    if not (_summary(A) and _summary(B)):
        st.error("Could not fetch MP data for A or B")
        st.stop()
    df = run_screen_cached(
        A=A, B=B,
        rh=humidity, temp=temperature,
        bg=(bg_lo,bg_hi),
        bow=bow, dx=dx,
        sn_penalty=sn_penalty
    )
    if df.empty:
        st.error("No candidates found — relax filters")
        st.stop()
    st.session_state.history.append((A,B,humidity,temperature,df))
else:
    if st.session_state.history:
        A,B,humidity,temperature,df = st.session_state.history[-1]
    else:
        st.info("Press ▶ Run screening to begin")
        st.stop()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🖼 Structure",
    "📊 Table",
    "📈 Plot",
    "📥 Download",
    "⚖ Benchmark"
])

# ─── Structure tab ────────────────────────────────────────────────────────────
with tab1:
    st.header("🖼 Perovskite Structure Visualization")
    choice = st.selectbox("Choose image:", [
        ("pero", "https://raw.githubusercontent.com/omogbadebowale/EnerMat-Explorer/main/images/pero.png"),
        ("CsPbBr3", "https://raw.githubusercontent.com/omogbadebowale/EnerMat-Explorer/main/images/CsPbBr3_structure.png"),
        ("CsSnBr3", "https://raw.githubusercontent.com/omogbadebowale/EnerMat-Explorer/main/images/CsSnBr3_structure.png")
    ], format_func=lambda x: x[0])[1]
    st.image(choice, use_column_width=True, caption="ABX₃ perovskite")

# ─── Table tab ─────────────────────────────────────────────────────────────────
with tab2:
    st.header("📊 Candidates Table")
    st.dataframe(df, use_container_width=True, height=500)

# ─── Plot tab ──────────────────────────────────────────────────────────────────
with tab3:
    st.header("📈 Stability vs. Band-gap")
    # highlight top 20%
    cutoff = df.score.quantile(0.80)
    df["is_top"]=df.score>=cutoff
    fig = px.scatter(
        df, x="stability", y="band_gap",
        color="score", color_continuous_scale="plasma",
        hover_data=["formula","x","band_gap","stability","score"],
        height=500
    )
    # outline top
    fig.add_trace(go.Scatter(
        x=df.loc[df.is_top,"stability"],
        y=df.loc[df.is_top,"band_gap"],
        mode="markers", marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(width=2,color="black")),
        showlegend=False, hoverinfo="skip"
    ))
    fig.update_layout(template="simple_white", margin=dict(l=50,r=50,t=50,b=50))
    st.plotly_chart(fig, use_container_width=True)
    png = fig.to_image(format="png", scale=2)
    st.download_button("📥 Download plot (PNG)", png, "stability_vs_gap.png", "image/png")

# ─── Download tab ──────────────────────────────────────────────────────────────
with tab4:
    st.header("📥 Download Results & Reports")
    st.download_button("Download CSV", df.to_csv(index=False).encode(), "EnerMat_results.csv","text/csv")
    # TXT
    top = df.iloc[0]
    txt = (
f"EnerMat report {pd.Timestamp.today().date()}\n"
f"Top: {top.formula}\n"
f"Gap: {top.band_gap:.3f} eV\n"
f"Stability: {top.stability:.3f}\n"
f"Score: {top.score:.3f}\n"
    )
    st.download_button("Download TXT report", txt, "report.txt","text/plain")
    # DOCX
    doc = Document()
    doc.add_heading("EnerMat Report",0)
    doc.add_paragraph(f"Date: {pd.Timestamp.today().date()}")
    doc.add_paragraph(f"Top candidate: {top.formula}")
    tbl=doc.add_table(rows=1,cols=2)
    for k,v in [("Gap",top.band_gap),("Stab",top.stability),("Score",top.score)]:
        r=tbl.add_row(); r.cells[0].text=k; r.cells[1].text=str(v)
    buf=io.BytesIO(); doc.save(buf); buf.seek(0)
    st.download_button("Download DOCX report",buf,"report.docx","application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# ─── Benchmark tab ────────────────────────────────────────────────────────────
with tab5:
    st.header("⚖ Benchmark: DFT vs. Experimental")
    # 1) experimental
    upload = st.file_uploader("Upload experimental CSV", type="csv")
    if upload:
        df_exp=pd.read_csv(upload); st.success("Loaded uploaded CSV")
    else:
        df_exp=pd.read_csv("exp_bandgaps.csv"); st.info("Loaded bundled CSV")
    if not {"formula","exp_gap"}.issubset(df_exp.columns):
        st.error("Need formula & exp_gap cols"); st.stop()
    # 2) DFT
    df_dft=pd.read_csv("pbe_bandgaps.csv")
    if not {"formula","pbe_gap"}.issubset(df_dft.columns):
        st.error("Need formula & pbe_gap cols"); st.stop()
    st.info(f"Loaded {len(df_dft)} DFT entries")
    # 3) merge
    dfm = pd.merge(df_exp, df_dft, on="formula").rename(
        columns={"exp_gap":"Exp Eg (eV)","pbe_gap":"DFT Eg (eV)"}
    )
    x,y = dfm["Exp Eg (eV)"].values, dfm["DFT Eg (eV)"].values
    # metrics
    mae  = np.mean(np.abs(y-x))
    rmse = np.sqrt(np.mean((y-x)**2))
    st.markdown(f"**MAE:** {mae:.3f} eV **RMSE:** {rmse:.3f} eV")
    # select labels
    fm = sorted(dfm.formula.unique())
    pick = st.multiselect("Formulas to label", fm, default=fm[:5])
    # parity
    mn = dfm[["Exp Eg (eV)","DFT Eg (eV)"]].min().min()
    mx = dfm[["Exp Eg (eV)","DFT Eg (eV)"]].max().max()
    m,b=np.polyfit(x,y,1)
    figp=px.scatter(dfm,x="Exp Eg (eV)",y="DFT Eg (eV)",hover_data=["formula"])
    figp.add_shape("line",x0=mn,y0=mn,x1=mx,y1=mx,line=dict(color="gray",dash="dash"))
    figp.add_shape("line",x0=mn,y0=m*mn+b,x1=mx,y1=m*mx+b,line=dict(color="black"))
    for _,r in dfm[dfm.formula.isin(pick)].iterrows():
        figp.add_annotation(x=r["Exp Eg (eV)"],y=r["DFT Eg (eV)"],text=r.formula,showarrow=False)
    figp.update_layout(margin=dict(l=50,r=50,t=50,b=50))
    st.plotly_chart(figp,use_container_width=True)
    imgp=figp.to_image(format="png",scale=2)
    st.download_button("Download parity PNG",imgp,"parity.png","image/png")
    # histogram
    errs=y-x
    figh=px.histogram(errs,nbins=20,labels={"value":"Δ Eg (eV)"})
    figh.update_layout(title_text="Error distribution")
    st.plotly_chart(figh,use_container_width=True)
    imgh=figh.to_image(format="png",scale=2)
    st.download_button("Download error hist PNG",imgh,"err_hist.png","image/png")
