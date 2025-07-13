# ── Download view ───────────────────────────────────────────────────────────
with tab_dl:
    st.download_button(
        "📥 Download CSV",
        df.to_csv(index=False).encode(),
        "EnerMat_results.csv", "text/csv"
    )

    top = df.iloc[0]
    label = (top.formula if mode.startswith("Binary")
             else f"{A}+{B}+{C} (x={top.x:.2f}, y={top.y:.2f})")

    txt = (f"EnerMat auto-report  {datetime.date.today()}\n"
           f"Top candidate   : {label}\n"
           f"Band-gap [eV]   : {top.Eg}\n"
           f"Ehull  [eV/atom]: {top.Ehull}\n"
           f"Eox   [eV/Sn]   : {getattr(top,'Eox','N/A')}\n"
           f"Score           : {top.score}\n")
    st.download_button("📄 Download TXT", txt, "EnerMat_report.txt", "text/plain")

    # DOCX
    doc = Document()
    doc.add_heading("EnerMat Report", 0)
    doc.add_paragraph(f"Date: {datetime.date.today()}")
    doc.add_paragraph(f"Top candidate: {label}")
    tbl = doc.add_table(rows=1, cols=2)
    hdr = tbl.rows[0].cells
    hdr[0].text, hdr[1].text = "Property", "Value"
    for k in ("Eg", "Ehull", "Eox", "score"):
        if k in top:
            row = tbl.add_row()
            row.cells[0].text, row.cells[1].text = k, str(top[k])
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    st.download_button(
        "📝 Download DOCX", buf, "EnerMat_report.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
