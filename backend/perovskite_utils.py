import io, os, datetime, streamlit as st, pandas as pd
import plotly.express as px, plotly.graph_objects as go
from docx import Document

# ── API key check ──────────────────────────────────────────────────
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    st.error("🛑 Please set MP_API_KEY (32 chars) in Secrets.")
    st.stop()

# ── backend helpers ────────────────────────────────────────────────
from backend.perovskite_utils import (
    mix_abx3      as screen_binary,
    screen_ternary,
    END_MEMBERS,
    fetch_mp_data as _summary,
)

# (everything else in app.py stays exactly as you already had it)
