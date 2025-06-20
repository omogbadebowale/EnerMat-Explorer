# ─── Plot Tab ─────────────────────────────────────────────────────────────────
with tab_plot:
    if mode == "Binary A–B":
        top_cut = df["score"].quantile(0.80)
        df["is_top"] = df["score"] >= top_cut

        fig = px.scatter(
            df,
            x="stability",
            y="Eg",                    # ← was "band_gap"
            color="score",
            color_continuous_scale="plasma",
            hover_data=["formula", "x", "Eg", "stability", "score"]
        )
        fig.update_traces(marker=dict(size=14, line_width=1), opacity=0.85)
        fig.add_trace(
            go.Scatter(
                x=df[df["is_top"]]["stability"],
                y=df[df["is_top"]]["Eg"],  # ← was "band_gap"
                mode="markers",
                marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(width=2, color="black")),
                hoverinfo="skip", showlegend=False
            )
        )
        fig.update_layout(template="simple_white", margin=dict(l=60, r=30, t=30, b=60))
        fig.update_xaxes(title="Stability")
        fig.update_yaxes(title="Band Gap (eV)")
        st.plotly_chart(fig, use_container_width=True)

    else:  # Ternary
        fig3d = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="score",
            color="score",
            hover_data=["Eg", "score"],  # only existing columns
            height=600
        )
        fig3d.update_layout(template="simple_white")
        st.plotly_chart(fig3d, use_container_width=True)
