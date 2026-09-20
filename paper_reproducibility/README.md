# Manuscript reproducibility package — v3.0.0

This directory contains the code, data and figure-generation routines used for the manuscript **“Uncertainty-aware multi-objective screening defines validation windows for Sn–Ge halide perovskites.”**

## Scope

The workflow screens 231 compositions in the Cs(Sn1−zGez)(I1−xBrx)3 design space, with x = 0–1 and z = 0–0.50 at 0.05 increments. It combines literature-informed band-gap anchors, interpolation-based phase-stability and oxidation-screening proxies, a tolerance-factor prior and an explicit weighted geometric composite score.

Uncertainty and robustness are examined at four complementary levels:

1. **Scoring-model sensitivity:** 10,000 Latin-hypercube draws for single-junction and tandem targets.
2. **Descriptor-calibration stress:** 5,000 coherent perturbations of optical and stability/oxidation anchor inputs.
3. **Weight-free and numerical checks:** four-objective Pareto analysis plus sample-size convergence diagnostics.
4. **Model-form stress:** 10,000 Latin-hypercube draws in a 19-dimensional space combining the declared calibration perturbations with six anchor-preserving nonlinear curvature coefficients for the phase-stability and oxidation-screening proxy surfaces.

The model-form stress test preserves every perturbed composition-space corner exactly. Each curvature coefficient spans ±25% of the corresponding full anchor range; aligned terms can produce an interior deformation up to 75% of that anchor span at the grid centre. These ranges are deliberate stress envelopes, not statistical confidence intervals.

The interpolated stability descriptors are prioritisation tools, not composition-specific free energies. No new composition-specific DFT calculations were performed for the inherited phase-stability or oxidation corner inputs in this study.

## Reproduce the analysis

From the repository root:

```bash
python paper_reproducibility/Supplementary_Code_1.py
python paper_reproducibility/Supplementary_Code_2_revision_checks.py
python paper_reproducibility/Supplementary_Code_3_Model_Form_Stress.py
```

All routines use fixed random seeds. Code 1 regenerates the base grid, S1–S3 datasets and original data-derived figures. Code 2 regenerates S4–S6 and the Pareto, convergence, external optical-challenge and TOC graphics. Code 3 regenerates S7a–S7b and the nonlinear model-form stress figure.

## Files

### Code
- `Supplementary_Code_1.py` — base grid, score mathematics, 10,000-draw scoring sensitivity, 5,000-draw calibration stress and original data-derived figures.
- `Supplementary_Code_2_revision_checks.py` — weight-free Pareto analysis, LHS convergence and independent Hooper et al. optical challenge.
- `Supplementary_Code_3_Model_Form_Stress.py` — 10,000-draw, 19-dimensional anchor-preserving nonlinear model-form stress test.

### Data
- `data/Supplementary_Data_S1_231_compositions.csv` — complete 231-composition descriptor, reference-score and robustness table.
- `data/Supplementary_Data_S2a_single_sensitivity.csv` — 10,000 single-junction scoring-model draws and representative ranks.
- `data/Supplementary_Data_S2b_tandem_sensitivity.csv` — 10,000 tandem-top-cell scoring-model draws and representative ranks.
- `data/Supplementary_Data_S3a_calibration_inputs.csv` — 5,000 coherent descriptor-calibration perturbations.
- `data/Supplementary_Data_S3b_calibration_summary.csv` — composition-level calibration-stress robustness summary.
- `data/Supplementary_Data_S4_Pareto_Grid.csv` — four-objective Pareto classification for the full grid.
- `data/Supplementary_Data_S5_LHS_Convergence.csv` — sample-size convergence diagnostic.
- `data/Supplementary_Data_S6_Hooper_Optical_Challenge.csv` — synthesis-resolved bromide optical challenge.
- `data/Supplementary_Data_S7a_Model_Form_Stress.csv` — composition-level nonlinear model-form stress summary.
- `data/Supplementary_Data_S7b_Model_Form_Draws.csv` — complete 10,000-draw, 19-dimensional model-form stress parameter set.
- `data/model_form_summary.txt` — compact cross-check values produced by Code 3.

### Figures
Code 1 generates Figures 1–3 and Supplementary Figures S1–S4. Code 2 generates the Pareto, convergence and independent optical-challenge figures and a TOC graphic. Code 3 generates the model-form stress figure. Publication graphics can be exported as high-resolution PNG and SVG from the supplied scripts.

### Exact ACS submission snapshot

The exact final ACS text files are archived in [`acs_submission_exact/`](acs_submission_exact/), including the three submitted Python scripts, `README.txt`, and the original `SHA256SUMS.txt`. To avoid duplicating the large datasets, the CSVs themselves remain in `data/`; S1–S7b have been verified byte-for-byte against the final ACS supplementary ZIP. The scripts in the parent directory are repository-native equivalents adapted to write outputs into `data/` and `figures/`.

## Key v3.0.0 robustness checks

- Single-junction x=0, z=0.40: top-decile occupancy **99.99%** under the combined calibration/model-form stress.
- Single-junction x=0, z=0.45: **99.78%**.
- Single-junction x=0, z=0.50: **98.56%**.
- Tandem x=0.35, z=0.50: **99.84%**.
- Tandem x=0.40, z=0.45: **100.00%**.
- Tandem x=0.45, z=0.45: **99.71%**.

These are occupancy fractions conditional on the declared stress-test family; they are not probabilities that a composition is physically optimal.

## Software requirements

Python 3 with NumPy, pandas, SciPy and Matplotlib. For the manuscript-only environment, install `paper_reproducibility/requirements.txt`; the repository-level `requirements.txt` also contains dependencies for the legacy interactive EnerMat Explorer dashboard.

Tested final-analysis environment:
- Python 3.13.5
- NumPy 2.3.5
- pandas 2.2.3
- SciPy 1.17.0
- Matplotlib 3.10.8

## Provenance

The v1.0.0 exploratory release and v2.0.0 manuscript release are retained in the Git/Zenodo version history. The inherited EnerMat source-workflow record for the stability/oxidation anchors is DOI **10.5281/zenodo.15756624**.

## License

Code is distributed under the repository MIT license. Dataset and figure reuse should cite the associated software release and manuscript.
