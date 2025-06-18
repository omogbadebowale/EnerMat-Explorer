import io
import itertools
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from backend.perovskite_utils import screen

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="EnerMat Discovery Engine")

# ─── Sidebar: Environment + knobs ─────────────────────────────────────────────
st.sidebar.header("Environment")
rh   = st.sidebar.slider("Humidity [%]",        0, 100, 50)
temp = st.sidebar.slider("Temperature [°C]", -20, 100, 25)

st.sidebar.markdown("---")
st.sidebar.header("Screening knobs")
bg_lo, bg_hi = st.sidebar.slider("Target gap [eV]", 0.0, 3.0, (0.8, 1.4), 0.01)
bowing       = st.sidebar.number_input("Bowing [eV]", 0.0, 1.0, 0.30, 0.05)
dx_mix       = st.sidebar.select_slider(
    "Mix Δx step", options=[0.1, 0.25, 0.5, 1.0], value=0.25
)

st.sidebar.markdown("---")
st.sidebar.caption("© 2025 Dr. Gbadebo Taofeek Yusuf")

# ─── Enumerate ABX₃ endpoints ──────────────────────────────────────────────────
A_POOL = ["Cs", "Rb", "MA", "FA"]
B_POOL = ["Pb", "Sn", "Ge"]
X_POOL = ["I", "Br", "Cl"]
END_MEMBERS = [f"{A}{B}{X}3" for A in A_POOL for B in B_POOL for X in X_POOL]

# ─── Run discovery when button clicked ────────────────────────────────────────
if st.sidebar.button("▶ Run full discovery"):
    records = []

    # 1) Pure end-members
    for M in END_MEMBERS:
        df1 = screen(
            A=M,
            B=M,
            rh=rh,
            temp=temp,
            bg=(bg_lo, bg_hi),
            bow=bowing,
            dx=1.0
        )
        records.append(df1)

    # 2) Binary mixes at dx_mix
    for a, b in itertools.combinations(END_MEMBERS, 2):
        df2 = screen(
            A=a,
            B=b,
            rh=rh,
            temp=temp,
            bg=(bg_lo, bg_hi),
            bow=bowing,
            dx=dx_mix
        )
        records.append(df2)

    # concatenate & sort by composite_score
    results = pd.concat(records, ignore_index=True)
    results = results.sort_values("score", ascending=False).reset_index(drop=True)

    st.session_state["results"] = results

# If never run, prompt and exit
if "results" not in st.session_state:
    st.info("Click ▶ Run full discovery to screen all ABX₃ chemistries.")
    st.stop()

results = st.session_state["results"]

# ─── Tabs (Table / Pareto Plot / Download) ────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Table", "📈 Pareto Plot", "📥 Download"])

with tab1:
    st.header("Top Candidates (first 100 rows)")
    st.dataframe(results.head(100), use_container_width=True, height=600)

with tab2:
    st.header("Pareto Front: Stability vs. Band-gap")
    df = results.copy()
    arr = df[["stability", "band_gap"]].values
    mask = np.ones(len(arr), dtype=bool)
    for i, row in enumerate(arr):
        if mask[i]:
            mask &= np.any(arr >= row, axis=1)
            mask[i] = True
    df["pareto"] = mask

    fig = px.scatter(
        df,
        x="stability", y="band_gap",
        color="score", size="score",
        color_continuous_scale="plasma",
        hover_data=["formula", "x", "score"],
        title="Discovery Pareto Front"
    )
    fig.add_trace(go.Scatter(
        x=df.loc[df.pareto, "stability"],
        y=df.loc[df.pareto, "band_gap"],
        mode="markers",
        marker=dict(symbol="circle-open-dot", size=18, line=dict(width=2)),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.update_layout(template="simple_white", margin=dict(l=50, r=50, t=50, b=50))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Download Full Discovery Results")
    csv = results.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download CSV",
        data=csv,
        file_name="EnerMat_full_discovery.csv",
        mime="text/csv",
        use_container_width=True
    )
