# ─────────────────────────────────────────────────────────────────────────────
#  app.py – EnerMat Perovskite Explorer v9.6   (2025-07-15, Ge-ready frontend)
# ─────────────────────────────────────────────────────────────────────────────
import datetime
import io

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from docx import Document
from backend.perovskite_utils import (
    screen_binary,
    screen_ternary,
    END_MEMBERS,
)
# ╭─────────────────────  PAGE CONFIG / TITLE  ─────────────────────╮
st.set_page_config("EnerMat Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.6")

# ╭─────────────────────  SESSION STATE  ───────────────────────────╮
if "history" not in st.session_state:
    st.session_state.history = []

# ╭─────────────────────  SIDEBAR  ─────────────────────────────────╮
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose screening type", ["Binary A–B", "Ternary A–B–C"])

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

    # optional B-site Ge fraction
    z = st.slider("Ge fraction z", 0.00, 0.30, 0.10, 0.05,
                  help="B-site Ge²⁺ in CsSn₁₋zGe_zX₃")

    if st.button("🗑 Clear history"):
        st.session_state.history.clear()
        st.experimental_rerun()

    st.caption(f"⚙️ Build SHA : dev • 🕒 {datetime.datetime.now():%Y-%m-%d %H:%M}")

# ╭─────────────────────  CACHE WRAPPERS  ──────────────────────────╮
@st.cache_data(show_spinner="⏳ Screening …", max_entries=20)
def _run_binary(*a, **k):
    return screen_binary(*a, **k)

@st.cache_data(show_spinner="⏳ Screening …", max_entries=10)
def _run_ternary(*a, **k):
    return screen_ternary(*a, **k)

# ╭─────────────────────  ACTION BUTTONS  ──────────────────────────╮
col_run, col_prev = st.columns([3, 1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_prev = col_prev.button("⏪ Previous", disabled=not st.session_state.history)

# ╭─────────────────────  HANDLE PREVIOUS  ─────────────────────────╮
if do_prev:
    st.session_state.history.pop()
    prev = st.session_state.history[-1]
    df   = prev["df"]; mode = prev["mode"]
    st.success("Showing previous result")

# ────────────── RUN NEW SCREEN ──────────────
elif do_run:
    # sanity-check end-member formulas
    for fml in ([A, B] if mode.startswith("Binary") else [A, B, C]):
        if fml not in END_MEMBERS:
            st.error(f"❌ Unknown end-member: {fml}")
            st.stop()

    if mode.startswith("Binary"):
        df = _run_binary(
            A, B, rh, temp,
            (bg_lo, bg_hi), bow, dx, z=z
        )

    else:  # ───────── TERNARY branch ─────────
        df = _run_ternary(
            A, B, C,                       # 1-3
            rh, temp,                      # 4-5
            (bg_lo, bg_hi),                # 6
            {"AB": bow, "AC": bow, "BC": bow},  # 7
            dx=dx, dy=dy,                  # 8-9
            z=z                            # 10 ← pass slider
        )

    st.session_state.history.append({"mode": mode, "df": df})

# nothing yet?
elif not st.session_state.history:
    st.info("Press ▶ Run screening to begin.")
    st.stop()

# ╭─────────────────────  MAIN TABS  ────────────────────────────────╮
df   = st.session_state.history[-1]["df"]
mode = st.session_state.history[-1]["mode"]

tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table", "📈 Plot", "📥 Download"])

with tab_tbl:
    st.dataframe(df, use_container_width=True, height=440)

with tab_plot:
    if mode.startswith("Binary") and {"Ehull", "Eg"}.issubset(df.columns):
        # ────────── Publication-grade Ehull vs Eg scatter ──────────
        fig = go.Figure()

        # main scatter
        fig.add_trace(go.Scatter(
            x=df["Ehull"], y=df["Eg"],
            mode="markers",
            marker=dict(
                size=8 + 12 * df["score"],
                color=df["score"],
                colorscale="Viridis",
                cmin=0, cmax=1,
                colorbar=dict(
                    title="Normalized<br>Score",
                    title_side="right",     # <-- changed here
                    tickmode="array",
                    tickvals=[0, 0.5, 1.0]
                ),
                line=dict(width=0.5, color="black"),
            ),
            hovertemplate=(
                "<b>%{customdata[6]}</b><br>"
                "Eg=%{y:.3f} eV<br>"
                "Ehull=%{x:.4f} eV/at<br>"
                "Score=%{marker.color:.3f}<extra></extra>"
            ),
            customdata=df.to_numpy(),
        ))

        # add target-window rectangle
        lo, hi = bg_lo, bg_hi
        fig.add_shape(type="rect",
            x0=0, x1=0.05,
            y0=lo, y1=hi,
            line=dict(color="LightSeaGreen", width=2, dash="dash"),
            fillcolor="LightSeaGreen", opacity=0.1,
        )

       -   fig.update_layout(
-       title="EnerMat Binary Screen: Stability vs Band-Gap",
-       xaxis=dict(
-           title="Ehull (eV/atom)",
-           zeroline=True, zerolinewidth=1, zerolinecolor="lightgrey",
-           showgrid=True, gridwidth=0.5, gridcolor="lightgrey",
-       ),
-       yaxis=dict(
-           title="Band-Gap Eg (eV)",
-           showgrid=True, gridwidth=0.5, gridcolor="lightgrey",
-       ),
-       template="simple_white",
-       font=dict(size=14),
-       margin=dict(l=60, r=20, t=60, b=60),
-       height=700,
-   )
+   fig.update_layout(
+       title="EnerMat Binary Screen",
+       title_font_size=18,
+       xaxis=dict(
+           title="Ehull (eV/atom)",
+           title_font_size=16,
+           tickfont_size=14,
+           zeroline=True, zerolinewidth=0.8, zerolinecolor="lightgrey",
+           showgrid=True, gridwidth=0.3, gridcolor="lightgrey",
+           mirror=True,  # draw axis lines on all sides
+           linecolor="black", linewidth=1
+       ),
+       yaxis=dict(
+           title="Band-Gap Eg (eV)",
+           title_font_size=16,
+           tickfont_size=14,
+           showgrid=True, gridwidth=0.3, gridcolor="lightgrey",
+           mirror=True,
+           linecolor="black", linewidth=1
+       ),
+       coloraxis_colorbar=dict(
+           title="Score",
+           title_font_size=14,
+           tickfont_size=12,
+           len=0.5,
+           y=0.5
+       ),
+       template="simple_white",
+       font=dict(family="Arial, sans-serif"),
+       margin=dict(l=50, r=30, t=50, b=50),
+       height=500,  # much more compact
+       width=650
+   )
 

        st.plotly_chart(fig, use_container_width=True)



    elif mode.startswith("Ternary") and {"x", "y", "score"}.issubset(df.columns):
        fig = px.scatter_3d(df, x="x", y="y", z="score", color="score",
                            color_continuous_scale="Turbo",
                            hover_data=df.columns, height=820)
        st.plotly_chart(fig, use_container_width=True)

with tab_dl:
    st.download_button("📥 Download CSV",
                       df.to_csv(index=False).encode(),
                       "EnerMat_results.csv", "text/csv")

# ╭─────────────────────  AUTO-REPORT  (TXT / DOCX)  ────────────────╮
_top = df.iloc[0]
formula = str(_top["formula"])
coords  = ", ".join(
    f"{c}={_top[c]:.2f}"
    for c in ("x", "y", "z", "ge_frac") if c in _top and pd.notna(_top[c])
)
label = formula if len(df) == 1 else f"{formula} ({coords})"

_txt = (
    "EnerMat auto-report  "
    f"{datetime.date.today()}\n"
    f"Top candidate   : {label}\n"
    f"Band-gap [eV]   : {_top['Eg']}\n"
    f"Ehull [eV/at.]  : {_top['Ehull']}\n"
    f"Eox_e [eV/e⁻]   : {_top.get('Eox_e', 'N/A')}\n"
    f"Score           : {_top['score']}\n"
)

st.download_button("📄 Download TXT", _txt,
                   "EnerMat_report.txt", mime="text/plain")

_doc = Document()
_doc.add_heading("EnerMat Report", level=0)
_doc.add_paragraph(f"Date : {datetime.date.today()}")
_doc.add_paragraph(f"Top candidate : {label}")

table = _doc.add_table(rows=1, cols=2)
table.style = "LightShading-Accent1"
hdr = table.rows[0].cells
hdr[0].text, hdr[1].text = "Property", "Value"
for k in ("Eg", "Ehull", "Eox_e", "score"):
    if k in _top:
        row = table.add_row().cells
        row[0].text, row[1].text = k, str(_top[k])

buf = io.BytesIO()
_doc.save(buf)
buf.seek(0)
st.download_button("📝 Download DOCX", buf,
                   "EnerMat_report.docx",
                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
# ─────────────────────────────────────────────────────────────────────────────
