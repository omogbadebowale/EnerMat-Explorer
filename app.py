# … everything above stays the same, up through the Download tab …

# ─────────── Benchmark Tab ───────────
with tab_bench:
    st.markdown("## ⚖ Benchmark: DFT vs. Experimental Gaps")
    st.write("Upload a CSV file with columns `formula` and `exp_gap` for your experimental data.")

    uploaded = st.file_uploader("Experimental band-gap CSV", type="csv")
    if not uploaded:
        st.info("Please upload `exp_bandgaps.csv` to enable benchmarking.")
    else:
        # 1) Load DFT gaps from MP API
        load_dotenv()
        mpr = MPRester(os.getenv("MP_API_KEY", ""))
        bench = []
        for f in END_MEMBERS:
            docs = mpr.summary.search(formula=f)
            entry = next(iter(docs), None)
            if entry:
                bench.append({"Formula": f, "DFT Eg (eV)": entry.band_gap})
        dfb = pd.DataFrame(bench)

        # 2) Load experimental data you just uploaded
        exp = pd.read_csv(uploaded)
        if "formula" not in exp.columns or "exp_gap" not in exp.columns:
            st.error("CSV must contain `formula` and `exp_gap` columns.")
        else:
            exp = exp.rename(columns={"formula": "Formula", "exp_gap": "Exp Eg (eV)"})

            # 3) Merge and compute deviations
            dfm = pd.merge(dfb, exp, on="Formula", how="inner")
            dfm["Δ Eg (eV)"] = dfm["DFT Eg (eV)"] - dfm["Exp Eg (eV)"]
            mae  = dfm["Δ Eg (eV)"].abs().mean()
            rmse = (dfm["Δ Eg (eV)"]**2).mean()**0.5
            st.write(f"**MAE:** {mae:.3f} eV **RMSE:** {rmse:.3f} eV")

            # 4) Parity plot
            fig1 = px.scatter(
                dfm, x="Exp Eg (eV)", y="DFT Eg (eV)",
                text="Formula",
                labels={"Exp Eg (eV)":"Experimental Eg","DFT Eg (eV)":"DFT Eg"},
                title="Parity Plot: DFT vs. Experimental"
            )
            fig1.add_shape(
                type="line", x0=dfm["Exp Eg (eV)"].min(), y0=dfm["Exp Eg (eV)"].min(),
                x1=dfm["Exp Eg (eV)"].max(), y1=dfm["Exp Eg (eV)"].max(),
                line=dict(dash="dash", color="gray")
            )
            st.plotly_chart(fig1, use_container_width=True)

            # Download parity plot
            png1 = fig1.to_image(format="png", scale=2)
            st.download_button(
                "📥 Download parity plot (PNG)",
                png1, "parity_plot.png", "image/png",
                use_container_width=True
            )

            # 5) Error histogram
            fig2 = px.histogram(
                dfm, x="Δ Eg (eV)", nbins=10,
                labels={"Δ Eg (eV)":"DFT − Exp (eV)"},
                title="Distribution of Band-Gap Errors"
            )
            st.plotly_chart(fig2, use_container_width=True)
