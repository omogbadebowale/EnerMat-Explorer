fig3d.update_layout(
    template="plotly_white",
    margin=dict(l=80, r=80, t=60, b=60),
    font=dict(family="Arial", size=14, color="#222"),

    # enforce equal physical scaling
    scene_aspectmode='cube',

    # orthographic camera
    scene_camera=dict(projection_type='orthographic',
                      eye=dict(x=1.2, y=1.2, z=0.8)),

    scene=dict(
        xaxis=dict(
            title="A fraction", title_font_size=16, tickfont_size=12,
            gridcolor="lightgrey", zerolinecolor="lightgrey",
            showbackground=False
        ),
        yaxis=dict(
            title="B fraction", title_font_size=16, tickfont_size=12,
            gridcolor="lightgrey", zerolinecolor="lightgrey",
            showbackground=False
        ),
        zaxis=dict(
            title="Score", title_font_size=16, tickfont_size=12,
            gridcolor="lightgrey", zerolinecolor="lightgrey",
            showbackground=False
        )
    ),

    # lock color range to [0,1]
    coloraxis_colorbar=dict(
        title="Score", title_font_size=14, tickfont_size=12,
        thickness=20, len=0.6, outlinewidth=1, outlinecolor="#444",
        # these two ensure consistent mapping
        cmin=0, cmax=1
    )
)
