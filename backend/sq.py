# backend/sq.py
import math

def sq_efficiency(Eg: float) -> float:
    """
    Returns the approximate Shockley–Queisser maximum PCE (as a fraction, e.g. 0.33 for 33%)
    for a single‐junction solar cell with bandgap Eg (in eV).
    This is a very rough Gaussian fit peaking at ~1.34 eV, η_max≈33%.
    """

    # peak efficiency ~33% at Eg = 1.34 eV, width ≈0.3 eV
    η_max = 0.33
    μ = 1.34
    σ = 0.30

    # if you're outside a reasonable range, return 0
    if Eg < 0.5 or Eg > 3.0:
        return 0.0

    # Gaussian approximation
    return η_max * math.exp(-((Eg - μ) ** 2) / (2 * σ * σ))
