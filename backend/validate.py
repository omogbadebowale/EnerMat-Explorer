# backend/validate.py

from __future__ import annotations
import numpy as np
import pandas as pd
import re
from pathlib import Path

# ─── point to the built-in 27-point CSV ────────────────────────────────
DATA_CSV = Path(__file__).parent / "data" / "perovskite_bandgap_merged.csv"

# ─── simple parser for CsSnxPbxI3 formulas ─────────────────────────────
_RE_SN = re.compile(r"Sn(?P<frac>[0-9.]+)?")
_RE_PB = re.compile(r"Pb(?P<frac>[0-9.]+)?")
def _frac(regex: re.Pattern[str], text: str) -> float:
    m = regex.search(text)
    if not m:
        return 0.0
    raw = m.group("frac")
    return 1.0 if raw in (None, "") else float(raw)

# ─── load experimental dataset ───────────────────────────────────────────
def load_default_dataset() -> pd.DataFrame:
    """27-point experimental benchmark shipped with the repo."""
    return pd.read_csv(DATA_CSV)

# ─── fetch end-member gaps (falls back if no API) ────────────────────────
try:
    from .perovskite_utils import fetch_mp_data
except ImportError:
    fetch_mp_data = None

_FALLBACK = {"CsSnI3": 1.30, "CsPbI3": 1.73}
_CACHE: dict[str, float] = {}
def _mp_gap(formula: str) -> float:
    if formula not in _CACHE:
        if fetch_mp_data:
            try:
                doc = fetch_mp_data(formula, ["band_gap"])
                _CACHE[formula] = float(doc["band_gap"])
            except Exception:
                _CACHE[formula] = _FALLBACK.get(formula, np.nan)
        else:
            _CACHE[formula] = _FALLBACK.get(formula, np.nan)
    return _CACHE[formula]

E_SN = _mp_gap("CsSnI3")
E_PB = _mp_gap("CsPbI3")

# ─── predict via Vegard + bowing ─────────────────────────────────────────
def _predict_one(formula: str, b: float) -> float:
    x_sn = _frac(_RE_SN, formula)
    x_pb = _frac(_RE_PB, formula)
    tot = x_sn + x_pb
    if tot == 0:
        return np.nan
    x = x_sn / tot
    return (1 - x) * E_PB + x * E_SN - b * x * (1 - x)

# ─── public validate() ───────────────────────────────────────────────────
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
    skipped   : DataFrame[Composition, Eg_eV]
    """
    # 1) Load default if none provided
    if df is None:
        df = load_default_dataset()

    # 2) Normalize column names so we always have Composition and Eg_eV
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )

    # 3) Coerce Eg_eV to numeric, drop rows where it fails or Composition missing
    df["Eg_eV"] = pd.to_numeric(df.get("Eg_eV", pd.Series()), errors="coerce")
    df = df.dropna(subset=["Composition", "Eg_eV"])

    # 4) Predict using Vegard + bowing
    df["Eg_pred"] = df["Composition"].apply(lambda f: _predict_one(f, b))

    # 5) Split into skipped (no Eg_pred) vs valid
    skipped = df[df.Eg_pred.isna()][["Composition", "Eg_eV"]]
    good = df.dropna(subset=["Eg_pred"]).copy()
    good["abs_err"] = (good["Eg_pred"] - good["Eg_eV"]).abs()

    if good.empty:
        raise ValueError("No valid compositions remain after parsing and type coercion.")

    # 6) Compute metrics
    metrics = dict(
        N=int(good.shape[0]),
        MAE=good.abs_err.mean(),
        RMSE=np.sqrt((good.abs_err ** 2).mean()),
        R2=np.corrcoef(good.Eg_eV.to_numpy(), good.Eg_pred.to_numpy())[0, 1] ** 2,
    )

    # 7) Return metrics, residuals, and any skipped rows
    residuals = good[["Composition", "Eg_eV", "Eg_pred", "abs_err"]]
    return metrics, residuals, skipped

