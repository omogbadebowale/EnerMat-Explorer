# EnerMat Explorer

Open-source tools for photovoltaic materials screening, with an interactive Streamlit dashboard and a manuscript-specific reproducibility package for lead-free Sn–Ge perovskite screening.

## v2.0.0 manuscript reproducibility package

The paper-specific workflow is located in [`paper_reproducibility/`](paper_reproducibility/). It contains the deterministic code, complete 231-composition dataset, 10,000-draw single-junction and tandem sensitivity analyses, 5,000-draw descriptor-calibration stress test, and the main/supplementary figures associated with the manuscript:

**“Physics-informed multi-objective screening reveals robust band-gap–stability windows in lead-free tin–germanium perovskites.”**

To reproduce the paper analysis:

```bash
python paper_reproducibility/Supplementary_Code_1.py
```

The paper workflow uses literature-informed band-gap anchors together with interpolation-based stability and oxidation proxies. These descriptors are intended for transparent prioritisation rather than composition-specific thermodynamic prediction. Composition-specific DFT calculations and experimental validation were not performed in the study.

See [`paper_reproducibility/README.md`](paper_reproducibility/README.md) for the full data and figure inventory.

## Interactive EnerMat Explorer dashboard

The original Streamlit application is retained for provenance and exploratory use.

### Quick start

1. Activate your Python environment.
2. Add your Materials Project API key to `.env` if using Materials Project functionality.
3. Install dependencies with `pip install -r requirements.txt`.
4. Run `streamlit run app.py`.
5. Select end members or enter a custom ABX3 composition and explore the available screening controls.

## Repository structure

- `paper_reproducibility/` — manuscript-specific v2.0.0 code, datasets and figures.
- `app.py`, `pages/`, `backend/`, `data/` — interactive EnerMat Explorer dashboard and supporting modules.
- `analysis/` — earlier analysis utilities retained for provenance.
- `CITATION.cff` — citation metadata for the current software release.
- `LICENSE` — repository license.

## Version history

- **v1.0.0** — first archival snapshot of the exploratory EnerMat Explorer workflow.
- **v2.0.0** — manuscript reproducibility release with explicit composite-score mathematics, 231-composition screening, scoring-model sensitivity analysis and descriptor-calibration stress testing.

## Author

Gbadebo Taofeek Yusuf
