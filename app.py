# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_tbl, tab_plot, tab_dl = st.tabs(["📊 Table", "📈 Plot", "📥 Download"])

# ─── Table Tab ───────────────────────────────────────────────────────────────
with tab_tbl:
    st.markdown("**Run parameters**")
    param_data = {
        "Parameter": ["Humidity [%]", "Temperature [°C]", "Gap window [eV]", "Bowing [eV]", "x-step"],
        "Value": [rh, temp, f"{bg_lo:.2f}–{bg_hi:.2f}", bow, dx]
    }
    if mode == "Ternary A–B–C":
        param_data["Parameter"].append("y-step")
        param_data["Value"].append(dy)

    st.table(pd.DataFrame(param_data))

    st.subheader("Candidate Results")
    st.dataframe(df, use_container_width=True, height=400)

# ─── Plot Tab ────────────────────────────────────────────────────────────────
with tab_plot:
    # Ensure necessary columns are numeric and non-null
    if mode == "Binary A–B":
        required = ["stability", "Eg", "score"]
        plot_df = df.dropna(subset=required).copy()
        for col in required:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        # Filter out any remaining invalid rows
        plot_df = plot_df.dropna(subset=required)

        # Compute top candidates for highlight
        top_cut = plot_df["score"].quantile(0.80)
        plot_df["is_top"] = plot_df["score"] >= top_cut

        try:
            fig = px.scatter(
                plot_df,
                x="stability",
                y="Eg",
                color="score",
                color_continuous_scale="plasma",
                hover_data=["formula", "x", "Eg", "stability", "score"]
            )
            fig.update_traces(marker=dict(size=14, line_width=1), opacity=0.85)
            fig.add_trace(
                go.Scatter(
                    x=plot_df.loc[plot_df["is_top"], "stability"],
                    y=plot_df.loc[plot_df["is_top"], "Eg"],
                    mode="markers",
                    marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(width=2, color="black")),
                    hoverinfo="skip", showlegend=False
                )
            )
            fig.update_layout(template="simple_white", margin=dict(l=60, r=30, t=30, b=60))
            fig.update_xaxes(title="Stability")
            fig.update_yaxes(title="Band Gap (eV)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Plot error: {e}")

    else:  # Ternary A–B–C
        required = [col for col in ["x", "y", "score"] if col in df.columns]
        if not required:
            st.warning("❗ No required ternary plot columns found. Cannot generate 3D plot.")
            st.stop()
        plot_df = df.dropna(subset=required).copy()
        for col in required:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df = plot_df.dropna(subset=required)

        try:
            fig3d = px.scatter_3d(
                plot_df,
                x="x",
                y="y",
                z="score",
                color="score",
                hover_data={k: True for k in ["x", "y", "Eg", "score"] if k in plot_df.columns},
                height=600
            )
            fig3d.update_layout(template="simple_white")
            st.plotly_chart(fig3d, use_container_width=True)
        except Exception as e:
            st.error(f"3D plot error: {e}")

# ─── Download Tab ──────────────────────────────────────────────────────────── ──────────────────────────────────────────────────────────── ────────────────────────────────────────────────────────────
with tab_dl:
    csv = df.to_csv(index=False).encode()
    st.download_button("📥 Download CSV", csv, "EnerMat_results.csv", "text/csv")

    # Determine top candidate representation
    top = df.iloc[0]
    if mode == "Binary A–B":
        top_label = top.formula
    else:
        # build formula string for ternary
        top_label = f"{A}-{B}-{C} x={top.x:.2f} y={top.y:.2f}"

    # Compose report text
    txt = f"""
EnerMat report ({datetime.date.today()})
Top candidate : {top_label}
Band-gap     : {top.Eg}
Stability    : {getattr(top, 'stability', 'N/A')}
Score        : {top.score}
"""
    st.download_button("📄 Download TXT", txt, "EnerMat_report.txt", "text/plain")

    # DOCX report
    doc = Document()
    doc.add_heading("EnerMat Report", 0)
    doc.add_paragraph(f"Date: {datetime.date.today()}")
    doc.add_paragraph(f"Top candidate: {top_label}")
    tbl = doc.add_table(rows=1, cols=2)
    # Add rows: Band-gap, Stability (if exists), and Score
    rows = [("Band-gap", top.Eg), ("Score", top.score)]
    if hasattr(top, 'stability'):
        rows.insert(1, ("Stability", top.stability))
    for k, v in rows:
        row = tbl.add_row()
        row.cells[0].text = k
        row.cells[1].text = str(v)
    buf = io.BytesIO()
    doc.save(buf); buf.seek(0)
    st.download_button("📝 Download DOCX", buf, "EnerMat_report.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
