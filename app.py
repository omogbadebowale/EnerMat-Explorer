# ─────────── SESSION STATE INIT ───────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────── RUNNING SCREEN ───────────
col_run, col_prev = st.columns([3, 1])
do_run  = col_run.button("▶ Run screening", type="primary")
do_prev = col_prev.button(
    "⏪ Previous",
    key="prev_button",
    disabled=(len(st.session_state.history) == 0)
)

if do_prev:
    # pop last result and show the one before it
    st.session_state.history.pop()
    entry = st.session_state.history[-1]
    df, mode = entry["df"], entry["mode"]
    st.success("Showing previous result")

elif do_run:
    # sanity check for custom end‐members
    members = [A, B] if mode.startswith("Binary") else [A, B, C]
    for f in members:
        if f not in END_MEMBERS:
            st.error(f"❌ Unknown end-member: {f}")
            st.stop()

    if mode.startswith("Binary"):
        df = _run_binary(
            A, B,
            rh, temp,
            (bg_lo, bg_hi),
            bow, dx,
            z=z,
            application=application,
        )
    else:
        df = _run_ternary(
            A, B, C,
            rh, temp,
            (bg_lo, bg_hi),
            {"AB": bow, "AC": bow, "BC": bow},
            dx=dx, dy=dy,
            z=z,
            application=application,
        )

    # save for “Previous”
    st.session_state.history.append({"mode": mode, "df": df})

# nothing run yet? show placeholder
elif not st.session_state.history:
    st.info("▶ Run screening to begin.")
    st.stop()

# finally, display the *current* results
current = st.session_state.history[-1]
df = current["df"]
mode = current["mode"]
