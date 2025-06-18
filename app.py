# app.py  –  EnerMat Perovskite Explorer v9.6 with Publication-Ready Benchmark

import io
import os
import datetime
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from docx import Document

from backend.perovskite_utils import screen, END_MEMBERS, _summary

# ───────────────────────────────────── App config ────────────────────────────
load_dotenv()  # load MP_API_KEY etc.
st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.6")

# ───────────────────────────────── Session History ──────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []

# ───────────────────────────────────── Sidebar ───────────────────────────────
with st.sidebar:
    st.header("Environment")
    rh   = st.slider("Humidity [%]", 0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)
    bg_lo, bg_hi = st.slider("Target gap [eV]", 0.5, 3.0, (1.0, 1.4), 0.01)

    st.header("Parent formulas")
    A_pick = st.selectbox("Preset A", END_MEMBERS, 0)
    B_pick = st.selectbox("Preset B", END_MEMBERS, 1)
    A = st.text_input("Custom A (optional)", "").strip() or A_pick
    B = st.text_input("Custom B (optional)", "").strip() or B_pick

    st.header("Model knobs")
    bow = st.number_input("Bowing [eV]", 0.0, 1.0, 0.30, 0.05)
    dx  = st.number_input("x-step",   0.01, 0.50, 0.05, 0.01)

    if st.button("🗑 Clear history"):
        st.session_state["history"].clear()
        st.experimental_rerun()

    st.caption("© 2025 Dr Gbadebo Taofeek Yusuf")
    GIT_SHA = st.secrets.get("GIT_SHA", "dev")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    st.caption(f"⚙️ Version: `{GIT_SHA}` • ⏱ {ts}")

# ─────────────────────────────────── Backend call ────────────────────────────
@st.cache_data(show_spinner="Monte-Carlo sampling…")
def run_screen(**kwargs):
    return screen(**kwargs)

# ──────────────────────────────── Run / Back logic ──────────────────────────
col_run, col_back = st.columns([3,1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=(len(st.session_state["history"]) < 1))

if do_back:
    st.session_state["history"].pop()
elif do_run:
    dA, dB = _summary(A), _summary(B)
    if not dA or not dB:
        st.error("Failed to fetch Materials Project data.")
        st.stop()
    df = run_screen(A=A, B=B, rh=rh, temp=temp, bg=(bg_lo, bg_hi), bow=bow, dx=dx)
    if df.empty:
        st.error("No candidates found – try widening your window.")
        st.stop()
    st.session_state["history"].append((A, B, rh, temp, df))
elif st.session_state["history"]:
    A, B, rh, temp, df = st.session_state["history"][-1]
else:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

# ───────────────────────────────────── Tabs ─────────────────────────────────
tab_tbl, tab_plot, tab_dl, tab_bench = st.tabs(
    ["📊 Table", "📈 Plot", "⬇ Download", "⚖ Benchmark"]
)

# ─────────── Table Tab ───────────
with tab_tbl:
    params = pd.DataFrame({
        "Parameter": ["Humidity [%]", "Temperature [°C]", "Gap window [eV]", "Bowing [eV]", "x-step"],
        "Value":     [rh, temp, f"{bg_lo:.2f}–{bg_hi:.2f}", bow, dx]
    })
    st.markdown("**Run parameters**")
    st.table(params.astype(str))

    docA, docB = _summary(A), _summary(B)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**A-endmember: {A}**")
        st.write(f"Band gap: {docA.band_gap:.2f} eV")
        st.write(f"E_above_hull: {docA.energy_above_hull:.3f} eV/atom")
    with c2:
        st.markdown(f"**B-endmember: {B}**")
        st.write(f"Band gap: {docB.band_gap:.2f} eV")
        st.write(f"E_above_hull: {docB.energy_above_hull:.3f} eV/atom")

    st.dataframe(df.astype(str), height=400, use_container_width=True)

# ─────────── Plot Tab ───────────
with tab_plot:
    st.caption("ℹ️ Hover, scroll, drag…")
    top_cut = df["score"].quantile(0.80)
    df["is_top"] = df["score"] >= top_cut

    fig = px.scatter(
        df, x="stability", y="band_gap",
        color="score", color_continuous_scale="plasma",
        hover_data=["formula","x","band_gap","stability","score"], height=450
    )
    fig.update_traces(marker=dict(size=16, line_width=1), opacity=0.9)
    outline = go.Scatter(
        x=df.loc[df.is_top,"stability"], y=df.loc[df.is_top,"band_gap"],
        mode="markers", hoverinfo="skip",
        marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(width=2,color="black")),
        showlegend=False
    )
    fig.add_trace(outline)
    fig.update_layout(template="simple_white", margin=dict(l=60,r=20,t=40,b=60),
                      coloraxis_colorbar=dict(title="Score"))
    st.plotly_chart(fig, use_container_width=True)

    png = fig.to_image(format="png", scale=2)
    st.download_button("📥 Download Plot (PNG)", png, "stability_vs_gap.png", "image/png")

# ─────────── Download Tab ───────────
with tab_dl:
    st.download_button("⬇ CSV", df.to_csv(index=False).encode(), "EnerMat_results.csv", "text/csv")
    top = df.iloc[0]
    summary = (
        f"EnerMat report ({datetime.date.today()})\n"
        f"Top: {top['formula']}\n"
        f"Band-gap: {top['band_gap']}\n"
        f"Stability: {top['stability']}\n"
        f"Score: {top['score']}\n"
    )
    st.download_button("📄 TXT report", summary, "EnerMat_report.txt", "text/plain")

    doc = Document()
    doc.add_heading("EnerMat Report", 0)
    doc.add_paragraph(f"Date: {datetime.date.today()}")
    doc.add_paragraph(f"Top: {top['formula']}")
    tbl = doc.add_table(rows=1, cols=2)
    for k,v in [("Band-gap",top['band_gap']),("Stability",top['stability']),("Score",top['score'])]:
        row = tbl.add_row()
        row.cells[0].text, row.cells[1].text = k, str(v)
    buf = io.BytesIO(); doc.save(buf); buf.seek(0)
    st.download_button("📥 DOCX report", buf,
                       "EnerMat_report.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# ─────────── Benchmark Tab ──────────
with tab_bench:
    st.markdown("## ⚖ Benchmark: DFT vs. Experimental Gaps")

    # 1) Try user upload first
    uploaded = st.file_uploader(
        "Upload experimental CSV (`formula`, `exp_gap`)", type="csv"
    )

    # 2) If no upload, fall back to bundled CSV
    if uploaded is None:
        st.info("No file uploaded — using bundled experimental data…")
        exp_path = Path(__file__).parent / "exp_bandgaps.csv"
        try:
            exp_df = pd.read_csv(exp_path)
            st.success("Loaded experimental data from bundled CSV")
        except Exception:
            st.error("Failed to load bundled CSV. Please upload your own.")
            st.stop()
    else:
        exp_df = pd.read_csv(uploaded)
        st.success("Loaded experimental data from uploaded file")

    # Validate columns
    if not {"formula", "exp_gap"}.issubset(exp_df.columns):
        st.error("CSV must contain columns: `formula` and `exp_gap`.")
        st.stop()

    # 3) Load DFT band gaps from bundled CSV
    pbe_path = Path(__file__).parent / "pbe_bandgaps.csv"
    try:
        dft_df = pd.read_csv(pbe_path)
        st.info(f"Loaded {len(dft_df)} DFT band gaps from bundled CSV")
    except Exception:
        st.error("Failed to load DFT CSV. Please bundle `pbe_bandgaps.csv`.")
        st.stop()

    if not {"formula", "pbe_gap"}.issubset(dft_df.columns):
        st.error("DFT CSV must contain columns: `formula` and `pbe_gap`.")
        st.stop()

    # 4) Merge
    exp_df = exp_df.rename(columns={"formula": "Formula", "exp_gap": "Exp Eg (eV)"})
    dft_df = dft_df.rename(columns={"formula": "Formula", "pbe_gap": "DFT Eg (eV)"})
    dfm = pd.merge(dft_df, exp_df, on="Formula", how="inner")
    if dfm.empty:
        st.error("No matching formulas between DFT and experimental data.")
        st.stop()

    dfm["Δ Eg (eV)"] = dfm["DFT Eg (eV)"] - dfm["Exp Eg (eV)"]

    # 5) Show stats (fixed parentheses!)
    mae = dfm["Δ Eg (eV)"].abs().mean()
    rmse = np.sqrt((dfm["Δ Eg (eV)"] ** 2).mean())
    st.write(f"**MAE:** {mae:.3f} eV **RMSE:** {rmse:.3f} eV")

    # 6) Parity Plot with guarded trendline
    x = dfm["Exp Eg (eV)"].to_numpy()
    y = dfm["DFT Eg (eV)"].to_numpy()
    mn, mx = x.min(), x.max()

    try:
        m, b = np.polyfit(x, y, 1)
    except np.linalg.LinAlgError:
        m, b = 1.0, 0.0  # fallback to y=x

    fig1 = px.scatter(
        dfm, x="Exp Eg (eV)", y="DFT Eg (eV)", hover_name="Formula",
        title="Parity Plot: DFT vs. Experimental"
    )
    fig1.add_trace(go.Scatter(
        x=[mn, mx],
        y=[m * mn + b, m * mx + b],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Fit" if b != 0 else "y = x"
    ))
    fig1.update_layout(template="simple_white", margin=dict(l=60, r=20, t=40, b=60))
    st.plotly_chart(fig1, use_container_width=True)

    png1 = fig1.to_image(format="png", scale=2)
    st.download_button("📥 Download parity plot (PNG)",
                       png1, "parity_plot.png", "image/png")

    # 7) Error Histogram
    fig2 = px.histogram(
        dfm, x="Δ Eg (eV)", nbins=10, title="Error Distribution (DFT – Experimental)"
    )
    fig2.update_layout(template="simple_white", margin=dict(l=60, r=20, t=40, b=60))
    st.plotly_chart(fig2, use_container_width=True)

    png2 = fig2.to_image(format="png", scale=2)
    st.download_button("📥 Download error histogram (PNG)",
                       png2, "error_histogram.png", "image/png")

