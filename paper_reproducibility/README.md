# Manuscript reproducibility package

This directory contains the code, data and figures used for the manuscript **“Physics-informed multi-objective screening reveals robust band-gap–stability windows in lead-free tin–germanium perovskites.”**

## Scope

The paper-specific workflow screens 231 compositions in the Cs(Sn1−zGez)(I1−xBrx)3 design space, with x = 0–1 and z = 0–0.50 at 0.05 increments. It combines literature-informed band-gap anchors, interpolation-based stability/oxidation proxies, a tolerance-factor prior, an explicit weighted geometric composite score, 10,000-draw score-sensitivity analyses, and a 5,000-draw coherent descriptor-calibration stress test.

The interpolated stability descriptors are prioritisation tools, not composition-specific free energies. Composition-specific DFT calculations and experimental validation were **not** performed in this study; the workflow identifies candidates for such future validation.

## Reproduce the analysis

From the repository root:

```bash
python paper_reproducibility/Supplementary_Code_1.py
```

The script recreates the manuscript datasets under `paper_reproducibility/data/` and the data-derived figures under `paper_reproducibility/figures/` using fixed random seeds.

## Files

### Code
- `Supplementary_Code_1.py` — deterministic screening, scoring, sensitivity analysis, calibration stress testing and figure generation.

### Data
- `data/Supplementary_Data_S1_231_compositions.csv` — complete 231-composition grid, descriptors, reference scores, ranks and robustness statistics.
- `data/Supplementary_Data_S2a_single_sensitivity.csv` — 10,000 single-junction scoring-model draws and representative ranks.
- `data/Supplementary_Data_S2b_tandem_sensitivity.csv` — 10,000 tandem-top-cell scoring-model draws and representative ranks.
- `data/Supplementary_Data_S3a_calibration_inputs.csv` — 5,000 coherent descriptor-calibration perturbations.
- `data/Supplementary_Data_S3b_calibration_summary.csv` — composition-level robustness summary from the calibration stress test.

### Figures
- `figures/Figure_1_workflow.svg` — study workflow and scope.
- `figures/Figure_2_tradeoff.svg` — band-gap–oxidation trade-off for representative compositions.
- `figures/Figure_3_landscape_robustness.png` — 231-composition single-junction score and robustness landscape.
- `figures/Figure_S1_tandem_robustness.png` — tandem score and robustness landscape.
- `figures/Figure_S2_rank_distributions.png` — representative rank distributions.
- `figures/Figure_S3_parameter_influence.png` — sensitivity of representative ranking to model parameters.
- `figures/Figure_S4_descriptor_stress.png` — robustness under coherent descriptor-calibration perturbations.

## Software requirements

The paper workflow requires Python 3 with NumPy, pandas, SciPy and Matplotlib. The repository-level `requirements.txt` also contains dependencies for the legacy interactive EnerMat Explorer dashboard.

## Provenance

The earlier EnerMat Explorer dashboard and the archived v1.0.0 release are retained for provenance. The `paper_reproducibility/` directory is the manuscript-specific v2.0.0 reproducibility package.

## License

Code is distributed under the repository MIT license. Dataset and figure reuse should cite the associated software release and manuscript.
