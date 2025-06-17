# EnerMat Explorer v9.3

Interactive, climate-aware screening of halide (and analogue) perovskites.

## Quick-start
1. Activate your venv → `.\.venv\Scripts\Activate`
2. Add your Materials-Project API key to `.env`
3. `streamlit run app.py`
4. **Pick two end-members** from the dropdowns *or* type any custom ABX₃
5. Adjust humidity, temperature, band-gap window → **Run**

## Interpreting the table
| Column | Meaning |
| ------ | ------- |
| `band_gap` | Mean DFT gap (GGA) with ±0.15 eV 95 % CI |
| `stability` | exp(−ΔE<sub>hull</sub>/0.10 eV), MC-averaged (σ = 0.02 eV) |
| `form_score` | Continuous Goldschmidt & octahedral factor (σ = 0.10) |
| `env_pen` | Arrhenius × humidity × Sn oxidation penalty |
| `lifetime` | 5 yr / `env_pen` (relative) |
| `score` | form × stability × gap-score ÷ env |

Bubble area ≈ lifetime, colour ≈ composite score.

**Author:** Gbadebo Taofeek Yusuf · taofeek.yusuf@example.com
