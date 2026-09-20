ACS Applied Energy Materials — Supplementary Data and Code (Final)

Manuscript: Uncertainty-aware multi-objective screening defines validation windows for Sn–Ge halide perovskites

CONTENTS
- Supplementary Data S1: complete 231-composition descriptor, score and robustness table.
- Supplementary Data S2a/S2b: 10,000-draw single-junction/tandem score-sensitivity samples.
- Supplementary Data S3a/S3b: 5,000-draw coherent descriptor-calibration stress inputs and summary.
- Supplementary Data S4: Pareto-classified composition grid.
- Supplementary Data S5: LHS sample-size convergence diagnostic.
- Supplementary Data S6: Hooper et al. synthesis-resolved optical challenge table.
- Supplementary Data S7a: composition-level combined calibration/nonlinear model-form stress summary.
- Supplementary Data S7b: complete 10,000-draw 19-dimensional model-form stress parameter set.
- Supplementary Code 1: base grid, scoring, LHS sensitivity, calibration stress and original figures.
- Supplementary Code 2: Pareto, convergence and external optical-challenge diagnostics.
- Supplementary Code 3: 10,000-draw anchor-preserving nonlinear model-form stress analysis and Fig. 7.

MODEL-FORM STRESS TEST
The final analysis uses seed 20260919 and 10,000 Latin-hypercube draws. It retains the 13 declared optical/anchor calibration perturbations and adds six nonlinear curvature coefficients to the E_hull and oxidation-screening proxy surfaces. The curvature basis vanishes at all four composition-space corners, so every perturbed corner is preserved exactly. Each E_hull curvature coefficient spans ±25% of the full E_hull corner range (±0.00152775 eV atom^-1) and each oxidation coefficient spans ±25% of the full oxidation corner range (±0.185425 eV/Sn). Aligned terms can sum to a 75% anchor-span interior deformation at the grid center. These are stress-test envelopes, not empirical confidence intervals.

KEY FINAL CHECKS
- Single-junction x=0, z=0.40: top-decile occupancy 99.99%; mean rank 3.22.
- Single-junction x=0, z=0.45: top-decile occupancy 99.78%; mean rank 2.86.
- Single-junction x=0, z=0.50: top-decile occupancy 98.56%; mean rank 3.71.
- Tandem x=0.35, z=0.50: top-decile occupancy 99.84%; mean rank 3.12.
- Tandem x=0.40, z=0.45: top-decile occupancy 100.00%; mean rank 8.26.
- Tandem x=0.45, z=0.45: top-decile occupancy 99.71%; mean rank 9.16.

PROVENANCE STATUS
The E_hull and oxidation corner values are inherited screening inputs from the archived EnerMat source workflow (DOI 10.5281/zenodo.15756624). No new composition-specific DFT calculations were performed for these corner values in the present study. Mixed-composition proxy values are not claimed as external database entries unless explicitly present in the archived workflow. The base revised release cited in the manuscript is EnerMat Explorer v2.0.0 (DOI 10.5281/zenodo.16883095).

TESTED ENVIRONMENT
Python 3.13.5
NumPy 2.3.5
pandas 2.2.3
SciPy 1.17.0
Matplotlib 3.10.8

REPRODUCTION
1. Place all code and data files in a writable directory.
2. Install numpy, pandas, scipy and matplotlib.
3. Run: python Supplementary_Code_1_EnerMat_robustness.py
4. Run: python Supplementary_Code_2_revision_checks.py
5. Run: python Supplementary_Code_3_Model_Form_Stress.py
6. Compare regenerated CSVs/figures with the packaged copies.

The random seeds, parameter ranges and ranking definitions are stated in the manuscript and Supporting Information.
