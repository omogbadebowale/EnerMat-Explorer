import io
import os
import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from docx import Document

# ─── Load API Key ─────────────────────────────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    st.error("🛑 Please set a valid 32-character MP_API_KEY in Streamlit Secrets.")
    st.stop()

# ─── Backend Imports ──────────────────────────────────────────────────────────
from backend.perovskite_utils import (
    mix_abx3 as screen,
    screen_ternary,
    END_MEMBERS,
    fetch_mp_data as _summary,
)

# ─── Streamlit Config ─────────────────────────────────────────────────────────
st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.6")

# ─── Session State Init ───────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Sidebar Configuration ────────────────────────────────────────────────────
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
    if mode == "Ternary A–B–C":
        preset_C = st.selectbox("Preset C", END_MEMBERS, index=2)
        custom_C = st.text_input("Custom C (optional)", "").strip()
        C = custom_C or preset_C

    st.header("Environment")
    rh = st.slider("Humidity [%]", 0, 100, 50)
    temp = st.slider("Temperature [°C]", -20, 100, 25)

    st.header("Target Band Gap [eV]")
    bg_lo, bg_hi = st.slider("Gap window [eV]", 0.5, 3.0, (1.0, 1.4), 0.01)

    st.header("Model Settings")
    bow = st.number_input("Bowing [eV]", 0.0, 1.0, 0.30, 0.05)
    dx = st.number_input("x-step", 0.01, 0.50, 0.05, 0.01)
    if mode == "Ternary A–B–C":
        dy = st.number_input("y-step", 0.01, 0.50, 0.05, 0.01)

    if st.button("🗑 Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()

    GIT_SHA = st.secrets.get("GIT_SHA", "dev")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    st.caption(f"⚙️ Version: `{GIT_SHA}` • ⏱ {ts}")
    st.caption("© 2025 Dr Gbadebo Taofeek Yusuf")

# ─── Cached Screen Runner ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Running screening…", max_entries=20)
def run_screen(formula_A, formula_B, rh, temp, bg_window, bowing, dx):
    return screen(
        formula_A=formula_A,
        formula_B=formula_B,
        rh=rh,
        temp=temp,
        bg_window=bg_window,
        bowing=bowing,
        dx=dx
    )

# ─── Execution Control ────────────────────────────────────────────────────────
col_run, col_back = st.columns([3, 1])
do_run = col_run.button("▶ Run screening", type="primary")
do_back = col_back.button("⏪ Previous", disabled=not st.session_state.history)

if do_back:
    st.session_state.history.pop()
    prev = st.session_state.history[-1]
    mode, A, B, rh, temp = prev["mode"], prev["A"], prev["B"], prev["rh"], prev["temp"]
    bg_lo, bg_hi = prev["bg"]
    bow, dx = prev["bow"], prev["dx"]
    if mode == "Ternary A–B–C":
        C, dy = prev["C"], prev["dy"]
    df = prev["df"]
    st.success("Showing previous result")

elif do_run:
    try:
        docA = _summary(A, ["band_gap", "energy_above_hull"])
        docB = _summary(B, ["band_gap", "energy_above_hull"])
        if mode == "Ternary A–B–C":
            docC = _summary(C, ["band_gap", "energy_above_hull"])
    except Exception as e:
        st.error(f"❌ Error querying Materials Project: {e}")
        st.stop()

    if not docA or not docB or (mode == "Ternary A–B–C" and not docC):
        st.error("❌ Invalid formula(s)—check your entries.")
        st.stop()

    if mode == "Binary A–B":
        df = run_screen(A, B, rh, temp, (bg_lo, bg_hi), bow, dx)
    else:
        try:
            df = screen_ternary(A, B, C, rh, temp, (bg_lo, bg_hi), {"AB": bow, "AC": bow, "BC": bow}, dx, dy, n_mc=200)
        except Exception as e:
            st.error(f"❌ Ternary error: {e}")
            st.stop()

    df = df.rename(columns={"energy_above_hull": "stability", "band_gap": "Eg"})
    entry = {"mode": mode, "A": A, "B": B, "rh": rh, "temp": temp,
             "bg": (bg_lo, bg_hi), "bow": bow, "dx": dx, "df": df}
    if mode == "Ternary A–B–C": entry.update({"C": C, "dy": dy})
    st.session_state.history.append(entry)

elif st.session_state.history:
    prev = st.session_state.history[-1]
    mode, A, B, rh, temp = prev["mode"], prev["A"], prev["B"], prev["rh"], prev["temp"]
    bg_lo, bg_hi = prev["bg"]
    bow, dx = prev["bow"], prev["dx"]
    if mode == "Ternary A–B–C": C, dy = prev["C"], prev["dy"]
    df = prev["df"]

else:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table", "📈 Plot", "📥 Download"])

# ─── Table Tab ───────────────────────────────────────────────────────────────
with tab_tbl:
    st.markdown("**Run parameters**")
    params = {"Parameter": ["Humidity [%]", "Temperature [°C]", "Gap window [eV]", "Bowing [eV]", "x-step"],
              "Value":    [rh, temp, f"{bg_lo:.2f}–{bg_hi:.2f}", bow, dx]}
    if mode == "Ternary A–B–C": params["Parameter"].append("y-step"); params["Value"].append(dy)
    st.table(pd.DataFrame(params))
    st.subheader("Candidate Results")
    st.dataframe(df, use_container_width=True, height=400)

# ─── Plot Tab ────────────────────────────────────────────────────────────────
with tab_plot:
    if mode == "Binary A–B":
        required = [c for c in ("stability","Eg","score") if c in df.columns]
        if len(required) < 3:
            st.error("❌ Missing required columns for plotting.")
            st.stop()
        plot_df = df.dropna(subset=required).copy()

        fig = px.scatter(
            plot_df,
            x="stability",
            y="Eg",
            color="score",
            color_continuous_scale="Inferno",
            hover_data=["formula","x","Eg","stability","score"],
            width=1200,
            height=800
        )
        fig.update_traces(marker=dict(size=10, opacity=0.9, line=dict(width=1, color="black")))
        top_cut = plot_df["score"].quantile(0.80)
        mask = plot_df["score"] >= top_cut
        fig.add_trace(go.Scatter(
            x=plot_df.loc[mask,"stability"],
            y=plot_df.loc[mask,"Eg"],
            mode="markers",
            marker=dict(size=14, symbol="circle-open", line=dict(width=2, color="black")),
            hoverinfo="skip",
            showlegend=False
        ))
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=80,r=40,t=60,b=80),
            font=dict(family="Times New Roman", size=20, color="#333"),
            xaxis=dict(title="Stability", title_font_size=24, tickfont_size=16,
                       showline=True, linecolor="black", linewidth=1.5),
            yaxis=dict(title="Band Gap (eV)", title_font_size=24, tickfont_size=16,
                       showline=True, linecolor="black", linewidth=1.5),
            coloraxis_colorbar=dict(title="Score", title_font_size=18, tickfont_size=16,
                                    thickness=20, len=0.75, outlinewidth=1, outlinecolor="#666"),
            annotations=[dict(xref='paper', yref='paper', x=0, y=1.02, text='(a)', showarrow=False,
                              font=dict(size=24, family="Times New Roman"))]
        )
        st.plotly_chart(fig, use_container_width=False)
        # static export button (requires kaleido)
        buf = fig.to_image(format="png", width=1200, height=800, scale=3)
        st.download_button(
            label="📷 Download Binary (3600×2400 px PNG)",
            data=buf,
            file_name="binary_highres.png",
            mime="image/png",
        )

    else:
        required = [c for c in ("x","y","score") if c in df.columns]
        if len(required) < 3:
            st.warning("❗ Not enough columns for ternary 3D plot.")
            st.stop()
        plot_df = df.dropna(subset=required).copy()

        fig3d = px.scatter_3d(
            plot_df,
            x="x",
            y="y",
            z="score",
            color="score",
            color_continuous_scale="Cividis",
            hover_data={k:True for k in ("x","y","Eg","score") if k in plot_df},
            width=1200,
            height=900
        )
        fig3d.update_traces(marker=dict(size=8, opacity=0.9, line=dict(width=1,color="black")))
        fig3d.update_layout(
            template="plotly_white",
            margin=dict(l=80,r=80,t=60,b=60),
            font=dict(family="Times New Roman", size=24, color="#222"),
            scene=dict(
                aspectmode='cube',
                camera=dict(projection=dict(type='orthographic'), eye=dict(x=1.2,y=1.2,z=0.8)),
                xaxis=dict(title="A fraction", title_font_size=28, tickfont_size=18,
                           showgrid=False, showline=False, zeroline=False),
                yaxis=dict(title="B fraction", title_font_size=28, tickfont_size=18,
                           showgrid=False, showline=False, zeroline=False),
                zaxis=dict(title="Score", title_font_size=28, tickfont_size=18,
                           showgrid=False, showline=False, zeroline=False)
            ),
            coloraxis=dict(cmin=0, cmax=1),
            coloraxis_colorbar=dict(title="Score", title_font_size=22, tickfont_size=18,
                                    thickness=20, len=0.6, outlinewidth=1, outlinecolor="#444"),
            annotations=[dict(xref='paper', yref='paper', x=0, y=1.02, text='(b)', showarrow=False,
                              font=dict(size=26, family="Times New Roman"))]
        )
        st.plotly_chart(fig3d, use_container_width=False)
        # static export button (requires kaleido)
        buf3 = fig3d.to_image(format="png", width=1200, height=900, scale=3)
        st.download_button(
            label="📷 Download Ternary (3600×2700 px PNG)",
            data=buf3,
            file_name="ternary_highres.png",
            mime="image/png",
        )

# ─── Download Tab ────────────────────────────────────────────────────────────
with tab_dl:
    csv = df.to_csv(index=False).encode()
    st.download_button("📥 Download CSV", csv, "EnerMat_results.csv", "text/csv")

    top = df.iloc[0]
    top_label = top.formula if mode=="Binary A–B" else f"{A}-{B}-{C} x={top.x:.2f} y={top.y:.2f}"

    txt = (
        f"EnerMat report ({datetime.date.today()})\n"
        f"Top candidate : {top_label}\n"
        f"Band-gap     : {top.Eg}\n"
        f"Stability    : {getattr(top,'stability','N/A')}\n"
        f"Score        : {top.score}\n"
    )
    st.download_button("📄 Download TXT", txt, "EnerMat_report.txt", "text/plain")

    doc = Document()
    doc.add_heading("EnerMat Report", 0)
    doc.add_paragraph(f"Date: {datetime.date.today()}")
    doc.add_paragraph(f"Top candidate: {top_label}")
    tbl = doc.add_table(rows=1, cols=2)
    hdr = tbl.rows[0].cells
    hdr[0].text, hdr[1].text = "Property", "Value"
    rows = [("Band-gap",top.Eg), ("Score",top.score)]
    if hasattr(top,'stability'):
        rows.insert(1,("Stability",top.stability))
    for prop,val in rows:
        r = tbl.add_row().cells
        r[0].text, r[1].text = prop, str(val)
    buf = io.BytesIO(); doc.save(buf); buf.seek(0)
    st.download_button(
        "📝 Download DOCX", buf, "EnerMat_report.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
