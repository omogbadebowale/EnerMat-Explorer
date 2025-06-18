```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page config
st.set_page_config(layout="wide", page_title="EnerMat Perovskite Explorer v9.6")

# --- Sidebar ---
st.sidebar.header("Environment")
humidity = st.sidebar.slider("Humidity [%]", 0, 100, 50)
temperature = st.sidebar.slider("Temperature [°C]", -20, 100, 25)
st.sidebar.markdown("---")
target_low, target_high = st.sidebar.slider(
    "Target gap [eV]", 0.5, 3.0, (1.0, 1.4), step=0.01
)
st.sidebar.markdown("---")

st.sidebar.header("Parent formulas")
preset_a = st.sidebar.selectbox("Preset A", ["CsPbBr3", "CsSnBr3"], index=0)
preset_b = st.sidebar.selectbox("Preset B", ["CsSnBr3", "CsPbBr3"], index=1)
custom_a = st.sidebar.text_input("Custom A (optional)")
custom_b = st.sidebar.text_input("Custom B (optional)")

if st.sidebar.button("▶ Run screening"):
    st.experimental_rerun()

# --- Main tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🖼 Structure",
    "📊 Table",
    "🖼 Plot",
    "📥 Download",
    "⚖ Benchmark"
])([
    "📊 Table",
    "🖼 Plot",
    "📥 Download",
    "⚖ Benchmark"
])

# --- Structure tab ---
with tab1:
    st.header("🖼 Perovskite Structure Visualization")
    # Display a generic perovskite crystal structure image
    # You can replace the URL with your own generated structure images
    structure_url = st.selectbox(
        "Choose structure image:",
        options=[
            ("CsPbBr3", "https://raw.githubusercontent.com/omogbadebowale/EnerMat-Explorer/main/images/CsPbBr3_structure.png"),
            ("CsSnBr3", "https://raw.githubusercontent.com/omogbadebowale/EnerMat-Explorer/main/images/CsSnBr3_structure.png"),
            ("Custom A-B", "https://raw.githubusercontent.com/omogbadebowale/EnerMat-Explorer/main/images/perovskite_generic.png")
        ], format_func=lambda x: x[0]
    )[1]
    st.image(structure_url, use_column_width=True, caption="Perovskite ABX3 Structure")

# --- Benchmark tab ---
with tab4:
    st.header("⚖ Benchmark: DFT vs. Experimental Gaps")

    # 1) Upload experimental data
    uploaded = st.file_uploader(
        "Upload experimental CSV (formula, exp_gap)", type="csv"
    )
    if uploaded:
        df_exp = pd.read_csv(uploaded)
        st.success("Loaded experimental data from uploaded file")
    else:
        df_exp = pd.read_csv("exp_bandgaps.csv")
        st.info("Loaded experimental data from bundled CSV")

    # Validate exp
    if not {"formula", "exp_gap"}.issubset(df_exp.columns):
        st.error("CSV must contain 'formula' and 'exp_gap' columns.")
        st.stop()

    # 2) Load DFT data
    df_dft = pd.read_csv("pbe_bandgaps.csv")
    if not {"formula", "pbe_gap"}.issubset(df_dft.columns):
        st.error("DFT CSV needs columns 'formula' and 'pbe_gap'.")
        st.stop()
    st.info(f"Loaded {len(df_dft)} DFT band gaps from bundled CSV")

    # 3) Merge and rename
    dfm = pd.merge(df_exp, df_dft, on="formula").rename(
        columns={"exp_gap": "Exp Eg (eV)", "pbe_gap": "DFT Eg (eV)"}
    )
    x = dfm["Exp Eg (eV)"].values
    y = dfm["DFT Eg (eV)"].values

    # 4) Metrics
    mae = np.mean(np.abs(y - x))
    rmse = np.sqrt(np.mean((y - x) ** 2))
    st.markdown(f"**MAE:** {mae:.3f} eV  **RMSE:** {rmse:.3f} eV")

    # 5) Label selection
    formulas = sorted(dfm["formula"].unique())
    to_label = st.multiselect(
        "Formulas to draw labels for", options=formulas, default=formulas[:5]
    )

    # 6) Build parity plot
    mn = dfm[["Exp Eg (eV)", "DFT Eg (eV)"]].min().min()
    mx = dfm[["Exp Eg (eV)", "DFT Eg (eV)"]].max().max()
    m, b = np.polyfit(x, y, 1)

    fig = px.scatter(
        dfm,
        x="Exp Eg (eV)",
        y="DFT Eg (eV)",
        hover_data=["formula"],
        labels={"Exp Eg (eV)": "Experimental Eg (eV)", "DFT Eg (eV)": "DFT Eg (eV)"},
    )
    # 1:1 line
    fig.add_shape(
        type="line", x0=mn, y0=mn, x1=mx, y1=mx,
        line=dict(color="lightgray", dash="dash"),
    )
    # best-fit line
    fig.add_shape(
        type="line", x0=mn, y0=m*mn + b, x1=mx, y1=m*mx + b,
        line=dict(color="gray", dash="dash"),
    )
    # annotate
    for _, row in dfm[dfm.formula.isin(to_label)].iterrows():
        fig.add_annotation(
            x=row["Exp Eg (eV)"], y=row["DFT Eg (eV)"],
            text=row["formula"], showarrow=False,
            font=dict(size=12)
        )
    fig.update_layout(margin=dict(l=50, r=50, t=50, b=50))
    st.plotly_chart(fig, use_container_width=True)

    # 7) Download parity
    img1 = fig.to_image(format="png", width=800, height=500, scale=2)
    st.download_button(
        "📥 Download parity plot (PNG)", data=img1, file_name="parity.png", mime="image/png"
    )

    # 8) Error histogram
    errors = y - x
    fig2 = px.histogram(
        errors, nbins=20,
        labels={"value": "Δ Eg (eV)"},
    )
    fig2.update_layout(title_text="Error Distribution (DFT – Experimental)")
    st.plotly_chart(fig2, use_container_width=True)

    img2 = fig2.to_image(format="png", width=800, height=400, scale=2)
    st.download_button(
        "📥 Download error histogram (PNG)", data=img2, file_name="error_hist.png", mime="image/png"
    )
```
