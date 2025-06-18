import io, itertools
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from docx import Document
from backend.perovskite_utils import screen

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="EnerMat Discovery Engine")

# ─── Sidebar: Environment + Model knobs ───────────────────────────────────────
st.sidebar.header("Environment")
rh   = st.sidebar.slider("Humidity [%]",       0, 100, 50)
temp = st.sidebar.slider("Temperature [°C]", -20, 100, 25)

st.sidebar.markdown("---")
st.sidebar.header("Screening knobs")
bg_lo, bg_hi = st.sidebar.slider("Target gap [eV]", 0.0, 3.0, (0.8,1.4), 0.01)
bow         = st.sidebar.number_input("Bowing [eV]", 0.0,1.0,0.30,0.05)
dx_mix      = st.sidebar.select_slider("Mix Δx step", options=[0.1,0.25,0.5,1.0], value=0.25)
sn_penalty  = st.sidebar.slider("Sn oxidation penalty", 0.0,1.0,0.50,0.05)

st.sidebar.markdown("---")
st.sidebar.caption("© 2025 Dr. Gbadebo Yusuf")

# ─── Define chemical pools & enumerate endpoints ──────────────────────────────
A_POOL = ["Cs","Rb","MA","FA"]
B_POOL = ["Pb","Sn","Ge"]
X_POOL = ["I","Br","Cl"]

END_MEMBERS = [f"{A}{B}{X}3"
               for A in A_POOL for B in B_POOL for X in X_POOL]

# ─── Run logic ────────────────────────────────────────────────────────────────
if st.sidebar.button("▶ Run full discovery"):
    all_records = []

    # 1) Pure end-members
    for M in END_MEMBERS:
        df_p = screen(
            formula_A=M, formula_B=M,
            rh=rh, temp=temp,
            bg=(bg_lo,bg_hi),
            bow=bow, dx=1.0,
            sn_penalty=sn_penalty
        )
        all_records.append(df_p)

    # 2) Binary mixes at fixed dx
    for a,b in itertools.combinations(END_MEMBERS, 2):
        df_m = screen(
            formula_A=a, formula_B=b,
            rh=rh, temp=temp,
            bg=(bg_lo,bg_hi),
            bow=bow, dx=dx_mix,
            sn_penalty=sn_penalty
        )
        all_records.append(df_m)

    # concatenate and sort
    results = pd.concat(all_records, ignore_index=True)
    results = results.sort_values("composite_score", ascending=False).reset_index(drop=True)
    st.session_state["results"] = results
else:
    results = st.session_state.get("results")
    if results is None:
        st.info("Click ▶ Run full discovery to begin screening hundreds of perovskites.")
        st.stop()

# ─── Tabs: Table / Plot / Download ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Table","📈 Plot","📥 Download"])

with tab1:
    st.header("Top candidates (all ABX₃ & binary mixes)")
    st.dataframe(results.head(100), use_container_width=True, height=600)

with tab2:
    st.header("Pareto front: Stability vs. Band-gap")
    df = results.copy()
    # Pareto mask
    mask = np.ones(len(df), dtype=bool)
    arr = df[["stability","band_gap"]].values
    for i,r in enumerate(arr):
        if mask[i]:
            mask &= np.any(arr >= r, axis=1)
            mask[i] = True
    df["pareto"] = mask

    fig = px.scatter(
        df, x="stability", y="band_gap",
        color="composite_score", size="composite_score",
        color_continuous_scale="plasma",
        hover_data=["formula_pretty","x","composite_score"],
        title="Full‐space Pareto front"
    )
    # outline Pareto
    fig.add_trace(go.Scatter(
        x=df.loc[df.pareto,"stability"], y=df.loc[df.pareto,"band_gap"],
        mode="markers", marker=dict(symbol="circle-open-dot", size=18, line=dict(width=2)),
        showlegend=False, hoverinfo="skip"
    ))
    fig.update_layout(template="simple_white", margin=dict(l=50,r=50,t=50,b=50))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Download all results")
    csv = results.to_csv(index=False).encode()
    st.download_button("Download FULL CSV", csv, "EnerMat_full_discovery.csv", "text/csv")
