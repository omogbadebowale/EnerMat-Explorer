# ─────────── Benchmark Tab ───────────
with tab_bench:
    st.markdown("## ⚖ Benchmark: DFT vs. Experimental Gaps")

    # ————————————————————————————
    # 1) Load bundled experimental & DFT CSVs
    # ————————————————————————————
    exp_path = Path(__file__).parent / "exp_bandgaps.csv"
    dft_path = Path(__file__).parent / "pbe_bandgaps.csv"

    try:
        exp_df = pd.read_csv(exp_path)
        st.success(f"Loaded {len(exp_df)} experimental entries from bundled CSV")
    except Exception:
        st.error("Could not load experimental CSV — please upload your own.")
        st.stop()

    try:
        dft_df = pd.read_csv(dft_path)
        st.info(f"Loaded {len(dft_df)} DFT band gaps from bundled CSV")
    except Exception:
        st.error("Could not load DFT CSV.")
        st.stop()

    # ————————————————————————————
    # 2) Validate
    # ————————————————————————————
    if not {"formula","exp_gap"}.issubset(exp_df.columns):
        st.error("Experimental CSV must have columns `formula` and `exp_gap`.")
        st.stop()
    if not {"formula","pbe_gap"}.issubset(dft_df.columns):
        st.error("DFT CSV must have columns `formula` and `pbe_gap`.")
        st.stop()

    # ————————————————————————————
    # 3) Merge & compute errors
    # ————————————————————————————
    exp_df = exp_df.rename(columns={"formula":"Formula","exp_gap":"Exp Eg (eV)"})
    dft_df = dft_df.rename(columns={"formula":"Formula","pbe_gap":"DFT Eg (eV)"})
    dfm = pd.merge(dft_df, exp_df, on="Formula", how="inner")
    dfm["ΔEg (eV)"] = dfm["DFT Eg (eV)"] - dfm["Exp Eg (eV)"]

    # ————————————————————————————
    # 4) Metrics
    # ————————————————————————————
    mae  = dfm["ΔEg (eV)"].abs().mean()
    rmse = np.sqrt((dfm["ΔEg (eV)"]**2).mean())
    st.markdown(f"**MAE:** {mae:.3f} eV  **RMSE:** {rmse:.3f} eV")

    # ————————————————————————————
    # 5) Publication-ready Parity Plot
    # ————————————————————————————
    fig1 = px.scatter(
        dfm,
        x="Exp Eg (eV)",
        y="DFT Eg (eV)",
        text="Formula",
        title="Parity Plot: DFT vs. Experimental",
        trendline="ols",                     # OLS line
        trendline_color_override="gray",
        height=500,
        width=800,
    )
    # 1:1 guide-line
    mn, mx = dfm[["Exp Eg (eV)","DFT Eg (eV)"]].min().min(), dfm[["Exp Eg (eV)","DFT Eg (eV)"]].max().max()
    fig1.add_shape(
        type="line",
        x0=mn, y0=mn, x1=mx, y1=mx,
        line=dict(dash="dash", color="lightgrey")
    )

    fig1.update_traces(
        marker=dict(size=8),
        textposition="top center",
        selector=dict(mode="markers+text")
    )
    fig1.update_layout(
        template="simple_white",
        margin=dict(l=80, r=40, t=60, b=60),
        xaxis=dict(title="Experimental Eg (eV)", tickfont_size=12, title_font_size=14),
        yaxis=dict(title="DFT Eg (eV)",        tickfont_size=12, title_font_size=14),
        showlegend=False,
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Download button
    png1 = fig1.to_image(format="png", scale=2)
    st.download_button(
        "📥 Download Parity Plot (PNG)",
        data=png1,
        file_name="parity_plot.png",
        mime="image/png"
    )

    # ————————————————————————————
    # 6) Publication-ready Error Histogram
    # ————————————————————————————
    fig2 = px.histogram(
        dfm,
        x="ΔEg (eV)",
        nbins=10,
        title="Error Distribution (DFT – Experimental)",
        height=400,
        width=800,
    )
    fig2.update_traces(marker_line_width=0)
    fig2.update_layout(
        template="simple_white",
        margin=dict(l=80, r=40, t=60, b=60),
        xaxis=dict(title="ΔEg (eV)", tickfont_size=12, title_font_size=14),
        yaxis=dict(title="Count",    tickfont_size=12, title_font_size=14),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Download button
    png2 = fig2.to_image(format="png", scale=2)
    st.download_button(
        "📥 Download Error Histogram (PNG)",
        data=png2,
        file_name="error_histogram.png",
        mime="image/png"
    )
