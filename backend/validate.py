"""
backend.validate – benchmark Sn–Pb narrow-band-gap perovskites
--------------------------------------------------------------

`validate(b, df)`  ➔  (metrics_dict, residual_df, skipped_df)
`load_default_dataset()`            ➔  27-point dataframe
"""

from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────
# 1.  Data paths
# ──────────────────────────────────────────────────────────────────
DATA_CSV = Path(__file__).parent / "data" / "perovskite_bandgap_merged.csv"

# ──────────────────────────────────────────────────────────────────
# 2.  Simple helpers
# ──────────────────────────────────────────────────────────────────
_RE_SN = re.compile(r"Sn(?P<frac>[0-9.]+)?")
_RE_PB = re.compile(r"Pb(?P<frac>[0-9.]+)?")

def _frac(regex: re.Pattern[str], text: str) -> float:
    """Return the stoichiometric coefficient after an element symbol."""
    m = regex.search(text)
    if m is None:
        return 0.0
    raw = m.group("frac")
    return 1.0 if raw in (None, "") else float(raw)

# ─ Materials-Project lookup – falls back to constants if no API key ─
try:
    from .perovskite_utils import fetch_mp_data  # already in the repo
except ImportError:         # unit-tests or offline use
    fetch_mp_data = None

_FALLBACK_GAPS = {"CsSnI3": 1.30, "CsPbI3": 1.73}
_CACHE: dict[str, float] = {}

def _mp_gap(formula: str) -> float:
    if formula not in _CACHE:
        if fetch_mp_data is not None:
            try:
                doc = fetch_mp_data(formula, ["band_gap"])
                if doc and doc["band_gap"] is not None:
                    _CACHE[formula] = float(doc["band_gap"])
                    return _CACHE[formula]
            except Exception:      # network error, bad key, …
                pass
        _CACHE[formula] = _FALLBACK_GAPS.get(formula, np.nan)
    return _CACHE[formula]

E_PB = _mp_gap("CsPbI3")    # end-member gaps, cached
E_SN = _mp_gap("CsSnI3")

# ──────────────────────────────────────────────────────────────────
# 3.  Gap predictor (Vegard + bowing)
# ──────────────────────────────────────────────────────────────────
def _predict_one(formula: str, b: float = 0.30) -> float:
    """Return predicted Eg or NaN if Sn/Pb not found."""
    x_sn = _frac(_RE_SN, formula)
    x_pb = _frac(_RE_PB, formula)
    tot = x_sn + x_pb
    if tot == 0:
        return np.nan                         # nothing to interpolate
    x = x_sn / tot                            # Sn fraction
    return (1.0 - x) * E_PB + x * E_SN - b * x * (1.0 - x)

# ──────────────────────────────────────────────────────────────────
# 4.  Public helpers
# ──────────────────────────────────────────────────────────────────
def load_default_dataset() -> pd.DataFrame:
    """27-point experimental benchmark shipped with the repo."""
    return pd.read_csv(DATA_CSV)

def validate(
    b: float = 0.30,
    df: pd.DataFrame | None = None,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Run the validation for bowing parameter *b*.

    Returns
    -------
    metrics : dict   – N, MAE, RMSE, R²
    residuals : DataFrame[Composition, Eg_eV, Eg_pred, abs_err]
    skipped   : DataFrame[Composition, Eg_eV] (rows we could not parse)
    """
    if df is None:
        df = load_default_dataset()

    df = df.copy()
    df["Eg_pred"] = df["Composition"].apply(lambda f: _predict_one(f, b))

    skipped = df[df.Eg_pred.isna()][["Composition", "Eg_eV"]]
    good = df.dropna(subset=["Eg_pred"]).copy()
    good["abs_err"] = (good["Eg_pred"] - good["Eg_eV"].astype(float)).abs()

    if good.empty:
        raise ValueError("No parsable compositions in the supplied file.")

    metrics = dict(
        N=int(good.shape[0]),
        MAE=good.abs_err.mean(),
        RMSE=np.sqrt((good.abs_err ** 2).mean()),
        R2=np.corrcoef(good.Eg_eV.astype(float), good.Eg_pred)[0, 1] ** 2,
    )
    return metrics, good[["Composition", "Eg_eV", "Eg_pred", "abs_err"]], skipped
