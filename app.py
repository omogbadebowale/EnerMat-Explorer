import io
import os
import datetime
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from docx import Document
from mp_api.client import MPRester

from backend.perovskite_utils import (
    mix_abx3 as screen,
    screen_ternary,
    END_MEMBERS,
    fetch_mp_data as _summary
)

# ───────────────────────────────────── App Config ─────────────────────────────
st.set_page_config(page_title="EnerMat Perovskite Explorer", layout="wide")
st.title("🔬 EnerMat **Perovskite** Explorer v9.6")

# ─────────────────────────────────── Session History ──────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# … (Sidebar + run/back logic, and the cached run_screen & run_ternary functions) …

# ─────────────────────────────────── Tabs ─────────────────────────────────────
tab_tbl, tab_plot, tab_dl, tab_bench, tab_results = st.tabs([
    "📊 Table", "📈 Plot", "📥 Download", "⚖ Benchmark", "📑 Results Summary"
])

# ─────────────────────────────── Table Tab ──────────────────────────────────
with tab_tbl:
    # … your existing code to show run parameters and df …

# ─────────────────────────────── Plot Tab ───────────────────────────────────
with tab_plot:
    if mode == "Binary A–B":
        st.caption("ℹ️ Hover for details; zoom/drag to explore")
        top_cut = df.score.quantile(0.80)
        df["is_top"] = df.score >= top_cut

        fig = px.scatter(
            df,
            x="stability",
            y="band_gap",
            color="score",
            color_continuous_scale="plasma",
            hover_data=["formula","x","band_gap","stability","score"],
            height=450
        )
        fig.update_traces(marker=dict(size=16, line_width=1), opacity=0.9)
        fig.add_trace(go.Scatter(
            x=df.loc[df.is_top, "stability"],
            y=df.loc[df.is_top, "band_gap"],
            mode="markers",
            marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(width=2,color="black")),
            hoverinfo="skip", showlegend=False
        ))
        fig.update_xaxes(title="<b>Stability</b>", range=[0.75,1.0], dtick=0.05)
        fig.update_yaxes(title_
