# backend/validate.py

from __future__ import annotations
import numpy as np
import pandas as pd
import re
from pathlib import Path

# ─── point to the *built-in* 27-point CSV ────────────────────────────────
DATA_CSV = Path(__file__).parent / "data" / "perovskite_bandgap_merged.csv"

# ─── simple parser for CsSnxPbxI3 formulas ───────────────────────────────
_RE_SN = re.compile(r"Sn(?P<frac>[0-9.]+)?")
_RE_PB = re.compile(r"Pb(?P<frac>[0-9.]+)?")
def _frac(regex, text):
    m = regex.search(text)
    if not m or m.group("frac") in (None, ""):
        return 0.0 if m is None else 1.0
    return float(m.group("frac"))

# ─── load experimental dataset ───────────────────────────────────────────
def load_default_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_CSV)

# ─── fetch end-member gaps (falls back if no API) ─────────────────────────
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

# ─── predict via Vegard + bowing ──────────────────────────────────────────
def _predict_one(formula: str, b: float) -> float:
    x_sn = _frac(_RE_SN, formula)
    x_pb = _frac(_RE_PB, formula)
    tot = x_sn + x_pb
    if tot == 0:
        return np.nan
    x = x_sn / tot
    return (1 - x) * E_PB + x * E_SN - b * x * (1 - x)

# ─── public validate() ────────────────────────────────────────────────────
def validate(
    b: float = 0.30,
    df: pd.DataFrame | None = None,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    if df is None:
        df = load_default_dataset()
    df = df.copy()
    df["Eg_pred"] = df["Composition"].apply(lambda f: _predict_one(f, b))

    skipped = df[df.Eg_pred.isna()][["Composition", "Eg_eV"]]
    good = df.dropna(subset=["Eg_pred"]).copy()
    good["abs_err"] = (good.Eg_pred - good.Eg_eV.astype(float)).abs()

    metrics = dict(
        N=int(good.shape[0]),
        MAE=good.abs_err.mean(),
        RMSE=np.sqrt((good.abs_err ** 2).mean()),
        R2=np.corrcoef(good.Eg_eV.astype(float), good.Eg_pred)[0, 1] ** 2,
    )
    return metrics, good[["Composition", "Eg_eV", "Eg_pred", "abs_err"]], skipped
