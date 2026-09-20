# EnerMat Explorer

Open-source tools for photovoltaic materials screening, with an interactive Streamlit dashboard and a manuscript-specific reproducibility package for lead-free Sn–Ge halide-perovskite screening.

## v3.0.0 manuscript reproducibility package

The paper-specific workflow is located in [`paper_reproducibility/`](paper_reproducibility/). Version 3.0.0 supports the manuscript:

**“Uncertainty-aware multi-objective screening defines validation windows for Sn–Ge halide perovskites.”**

The release contains the complete 231-composition screening dataset and extends the v2.0.0 workflow with:

- 10,000-draw single-junction and tandem scoring-model sensitivity analyses;
- a 5,000-draw coherent descriptor-calibration stress test;
- a weight-free four-objective Pareto diagnostic;
- Latin-hypercube sample-size convergence analysis;
- an independent synthesis-resolved optical challenge against Hooper et al. (2026); and
- a 10,000-draw, 19-dimensional nonlinear model-form stress test that perturbs calibration inputs and introduces anchor-preserving curvature in the phase-stability and oxidation-screening proxy surfaces.

To reproduce the paper analysis:

```bash
python paper_reproducibility/Supplementary_Code_1.py
python paper_reproducibility/Supplementary_Code_2_revision_checks.py
python paper_reproducibility/Supplementary_Code_3_Model_Form_Stress.py
```

The stability and oxidation quantities are screening proxies. The workflow is intended for transparent prioritisation and uncertainty analysis, not as a substitute for composition-specific alloy thermodynamics, defect calculations, oxidation kinetics, or experimental validation. No new composition-specific DFT calculations were performed for the inherited stability/oxidation corner inputs.

See [`paper_reproducibility/README.md`](paper_reproducibility/README.md) for the full code, dataset and generated-figure inventory.

### ACS submission snapshot

The final flat ACS supplementary-package text files are preserved under [`paper_reproducibility/acs_submission_exact/`](paper_reproducibility/acs_submission_exact/). The committed S1–S7b CSV files under `paper_reproducibility/data/` have been verified byte-for-byte against the final supplementary ZIP, and the original SHA-256 manifest is retained in the snapshot folder. Repository-native scripts remain in `paper_reproducibility/` because they write outputs into the repository's structured `data/` and `figures/` directories.

## Interactive EnerMat Explorer dashboard

The original Streamlit application is retained for provenance and exploratory use.

### Quick start

1. Activate your Python environment.
2. Add your Materials Project API key to `.env` if using Materials Project functionality.
3. Install dependencies with `pip install -r requirements.txt`.
4. Run `streamlit run app.py`.
5. Select end members or enter a custom ABX3 composition and explore the available screening controls.

## Repository structure

- `paper_reproducibility/` — manuscript-specific v3.0.0 code, datasets and figure-generation routines.
- `app.py`, `pages/`, `backend/`, `data/` — interactive EnerMat Explorer dashboard and supporting modules.
- `analysis/` — earlier analysis utilities retained for provenance.
- `CITATION.cff` — citation metadata for the current software release.
- `LICENSE` — repository license.

## Version history

- **v1.0.0** — first archival snapshot of the exploratory EnerMat Explorer workflow.
- **v2.0.0** — manuscript reproducibility release with explicit score mathematics, 231-composition screening, scoring-model sensitivity analysis and descriptor-calibration stress testing.
- **v3.0.0** — final ACS Applied Energy Materials reproducibility release adding Pareto, convergence, external optical-challenge and nonlinear model-form robustness analyses.

## Author

Gbadebo Taofeek Yusuf
