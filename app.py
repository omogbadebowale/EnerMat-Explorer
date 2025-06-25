# app.py  ─ EnerMat Perovskite Explorer v9.7  (2025-06-25)
# -----------------------------------------------------------------------
# • works with backend/perovskite_utils.py v2025-06-25
# • shows stability & gap_score columns
# • TXT / DOCX export include gap factor
# • optional “S vs RH” curve button for binary runs
# -----------------------------------------------------------------------

import io, os, datetime as dt
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from docx import Document

# ─── Materials-Project key ──────────────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    st.error("🛑 Please set a valid 32-character MP_API_KEY in Streamlit Secrets")
    st.stop()

# ─── Backend helpers ────────────────────────────────────────────────────
from backend.perovskite_utils import (
    mix_abx3 as screen,
    screen_ternary,
    END_MEMBERS,
    fetch_mp_data as _summary,
)

# ─── Helper: S-vs-RH curve for a fixed x (Fig. 3) ───────────────────────
def humidity_curve(A, B, x_fixed=0.30, temp=25, bow=0.30):
    """Return DataFrame of S versus RH for one binary composition."""
    records = []
    for rh in range(0, 101, 10):          # 0,10,…100 %
        df = screen(
            formula_A=A, formula_B=B,
            rh=rh, temp=temp,
            bg_window=(1.0, 1.4),
            bowing=bow, dx=0.005,
        )
        row = df.iloc[(df["x"] - x_fixed).abs().argsort().iloc[0]]
        records.append({"RH %": rh, "S": row.score})
    return pd.DataFrame(records)

# ─── Streamlit-page settings ────────────────────────────────────────────
st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.7")

# ─── Session state ──────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Sidebar UI ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose screening type", ["Binary A–B", "Ternary A–B–C"])

    st.header("End-members")
    preset_A = st.selectbox("Preset A", END_MEMBERS, 0)
    preset_B = st.selectbox("Preset B", END_MEMBERS, 1)
    A = (custA := st.text_input("Custom A (optional)", "").strip()) or preset_A
    B = (custB := st.text_input("Custom B (optional)", "").strip()) or preset_B
    if mode == "Ternary A–B–C":
        preset_C = st.selectbox("Preset C", END_MEMBERS, 2)
        C = (custC := st.text_input("Custom C (optional)", "").strip()) or preset_C

    st.header("Environment")
    rh   = st.slider("Humidity [%]",      0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)

    st.header("Target band-gap window [eV]")
    bg_lo, bg_hi = st.slider("Gap window", 0.5, 3.0, (1.0, 1.4), 0.01)

    st.header("Model options")
    bow = st.number_input("Bowing [eV]", 0.0, 1.0, 0.30, 0.05)
    dx  = st.number_input("x-step",       0.01, 0.50, 0.05, 0.01)
    if mode == "Ternary A–B–C":
        dy = st.number_input("y-step",    0.01, 0.50, 0.05, 0.01)

    if st.button("🗑 Clear history"):
        st.session_state.history.clear(); st.experimental_rerun()

    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    st.caption(f"⚙️ Build: dev • ⏱ {ts}")
    st.caption("© 2025 Dr Gbadebo Taofeek Yusuf")

# ─── Cached binary runner (Streamlit native cache) ──────────────────────
@st.cache_data(show_spinner="⏳ Running screening…", max_entries=20)
def run_screen(A, B, rh, temp, bg, bow, dx):
    return screen(A, B, rh, temp, bg, bow, dx)

# ─── Control buttons ────────────────────────────────────────────────────
col_run, col_back = st.columns([3, 1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=not st.session_state.history)

# ─── Restore previous view ──────────────────────────────────────────────
if do_back:
    st.session_state.history.pop()
    prev = st.session_state.history[-1]
    mode, A, B, rh, temp = prev["mode"], prev["A"], prev["B"], prev["rh"], prev["temp"]
    bg_lo, bg_hi = prev["bg"]
    bow, dx      = prev["bow"], prev["dx"]
    if mode == "Ternary A–B–C":
        C, dy = prev["C"], prev["dy"]
    df = prev["df"]
    st.success("Showing previous result")

# ─── Fresh run ──────────────────────────────────────────────────────────
elif do_run:
    try:
        docA = _summary(A, ["band_gap", "energy_above_hull"]); docB = _summary(B, ["band_gap","energy_above_hull"])
        if mode == "Ternary A–B–C": docC = _summary(C, ["band_gap","energy_above_hull"])
    except Exception as e:
        st.error(f"❌ Error querying Materials Project: {e}"); st.stop()

    if not docA or not docB or (mode=="Ternary A–B–C" and not docC):
        st.error("❌ Invalid formula–check entries"); st.stop()

    if mode == "Binary A–B":
        df = run_screen(A, B, rh, temp, (bg_lo,bg_hi), bow, dx)
    else:
        try:
            df = screen_ternary(
                A=A, B=B, C=C, rh=rh, temp=temp,
                bg=(bg_lo,bg_hi),
                bows={"AB":bow,"AC":bow,"BC":bow},
                dx=dx, dy=dy
            )
        except Exception as e:
            st.error(f"❌ Ternary error: {e}"); st.stop()

    entry = dict(mode=mode, A=A, B=B, rh=rh, temp=temp,
                 bg=(bg_lo,bg_hi), bow=bow, dx=dx, df=df)
    if mode=="Ternary A–B–C": entry.update(C=C, dy=dy)
    st.session_state.history.append(entry)

# ─── Grab existing result if present ────────────────────────────────────
elif st.session_state.history:
    prev = st.session_state.history[-1]
    mode, A, B, rh, temp = prev["mode"], prev["A"], prev["B"], prev["rh"], prev["temp"]
    bg_lo, bg_hi = prev["bg"]; bow, dx = prev["bow"], prev["dx"]
    if mode == "Ternary A–B–C": C, dy = prev["C"], prev["dy"]
    df = prev["df"]
else:
    st.info("Press ▶ Run screening to begin."); st.stop()

# ─── Tabs ───────────────────────────────────────────────────────────────
tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table", "📈 Plot", "📥 Download"])

# ─── Table tab ─────────────────────────────────────────────────────────
with tab_tbl:
    st.markdown("**Run parameters**")
    params = {"Parameter":["Humidity [%]","Temperature [°C]","Gap window [eV]","Bowing [eV]","x-step"],
              "Value":[rh,temp,f"{bg_lo:.2f}–{bg_hi:.2f}",bow,dx]}
    if mode=="Ternary A–B–C": params["Parameter"].append("y-step"); params["Value"].append(dy)
    st.table(pd.DataFrame(params))

    st.subheader("Candidate results")
    st.dataframe(df, use_container_width=True, height=400)   # now shows gap_score

# ─── Plot tab ──────────────────────────────────────────────────────────
with tab_plot:
    if mode=="Binary A–B":
        req = [c for c in ("stability","Eg","score") if c in df.columns]
        if len(req)<3: st.error("Missing columns"); st.stop()
        plot_df = df.dropna(subset=req).copy()
        fig = px.scatter(plot_df, x="stability", y="Eg", color="score",
                         color_continuous_scale="Turbo",
                         hover_data=["formula","x","Eg","stability","score"],
                         width=1200, height=800)
        fig.update_traces(marker=dict(size=12,opacity=0.9,
                                      line=dict(width=1,color="black")))
        top20 = plot_df["score"].quantile(0.80)
        mask  = plot_df["score"]>=top20
        fig.add_trace(go.Scatter(x=plot_df.loc[mask,"stability"],
                                 y=plot_df.loc[mask,"Eg"],
                                 mode="markers",
                                 marker=dict(size=20,symbol="circle-open",
                                             line=dict(width=2,color="black")),
                                 hoverinfo="skip",showlegend=False))
        fig.update_layout(template="plotly_white",
                          xaxis_title="Stability (exp-weighted)",
                          yaxis_title="Band-gap (eV)",
                          margin=dict(l=80,r=40,t=60,b=80))
        st.plotly_chart(fig,use_container_width=True)

        # optional S-vs-RH curve
        if st.button("📈 Show S vs RH curve"):
            curve_df = humidity_curve(A, B, x_fixed=0.30, temp=temp, bow=bow)
            fig2 = px.line(curve_df, x="RH %", y="S", markers=True,
                           title=f"S vs RH (x=0.30, {A}/{B})")
            fig2.update_layout(template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)

    else:
        req = [c for c in ("x","y","score") if c in df.columns]
        if len(req)<3: st.warning("Not enough columns"); st.stop()
        plot_df = df.dropna(subset=req).copy()
        fig3d = px.scatter_3d(plot_df, x="x", y="y", z="score",
                              color="score", color_continuous_scale="Turbo",
                              hover_data={"x":True,"y":True,"Eg":True,"score":True},
                              width=1200,height=900)
        fig3d.update_traces(marker=dict(size=5,opacity=0.9,
                                        line=dict(width=1,color="black")))
        fig3d.update_layout(template="plotly_white",
                            scene=dict(xaxis_title="A fraction",
                                       yaxis_title="B fraction",
                                       zaxis_title="Score"))
        st.plotly_chart(fig3d,use_container_width=True)

# ─── Download tab ──────────────────────────────────────────────────────
with tab_dl:
    st.download_button("📥 Download CSV", df.to_csv(index=False).encode(),
                       "EnerMat_results.csv", "text/csv")

    top = df.iloc[0]
    if mode=="Binary A–B":    top_label = top.formula
    else:                     top_label = f"{A}-{B}-{C} x={top.x:.2f} y={top.y:.2f}"

    txt = f"""EnerMat report ({dt.date.today()})
Top candidate  : {top_label}
Band-gap (eV)  : {top.Eg}
Stability      : {getattr(top,'stability','N/A')}
Gap factor     : {getattr(top,'gap_score','N/A')}
Composite S    : {top.score}
"""
    st.download_button("📄 Download TXT", txt, "EnerMat_report.txt", "text/plain")

    doc = Document(); doc.add_heading("EnerMat Report", 0)
    doc.add_paragraph(f"Date: {dt.date.today()}")
    doc.add_paragraph(f"Top candidate: {top_label}")
    tbl = doc.add_table(rows=1, cols=2)
    hdr_cells = tbl.rows[0].cells; hdr_cells[0].text, hdr_cells[1].text = "Property", "Value"
    rows = [("Band-gap (eV)", top.Eg)]
    if hasattr(top,"stability"): rows.append(("Stability", top.stability))
    if hasattr(top,"gap_score"): rows.append(("Gap factor", top.gap_score))
    rows.append(("Composite S", top.score))
    for k,v in rows:
        row=tbl.add_row(); row.cells[0].text, row.cells[1].text = k, str(v)
    buf = io.BytesIO(); doc.save(buf); buf.seek(0)
    st.download_button("📝 Download DOCX", buf, "EnerMat_report.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
