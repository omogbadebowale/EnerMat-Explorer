import os, math
import numpy as np, pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dataclasses import dataclass

# ─── Materials Project setup ──────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY")
if not API_KEY or len(API_KEY) < 22:
    raise RuntimeError("MP_API_KEY missing – set it in .env")
mpr = MPRester(API_KEY)

# ─── End-members and ionic radii ──────────────────────────────────────────────
IONIC_RADII = {"Cs":1.88,"Pb":1.19,"Sn":1.18,"Ge":0.73,"I":2.20,"Br":1.96,"Cl":1.81}

# ─── Retry wrapper for MP queries ──────────────────────────────────────────────
class MPUp(Exception): pass

@retry(stop=stop_after_attempt(3), wait=wait_exponential(), retry=retry_if_exception_type(MPUp))
def _summary(formula: str):
    try:
        docs = mpr.summary.search(formula=formula,
                                  fields=["band_gap","energy_above_hull","is_stable"])
    except Exception as e:
        raise MPUp(e)
    if not docs:
        return None
    for d in docs:
        if getattr(d, "is_stable", True):
            return d
    return docs[0]

# ─── Scoring helpers ─────────────────────────────────────────────────────────
def _gap_score(g, lo, hi):
    if lo <= g <= hi:
        return 1.0
    return max(0.0, 1 - min(abs(g - lo), abs(g - hi)) / (hi - lo))

def _env_penalty(rh: float, T: float, sn_frac: float) -> float:
    k_B, T_ref = 8.617e-5, 298.15
    arr = math.exp(0.60/k_B * (1/(T+273.15) - 1/T_ref))
    hum = math.exp(1.0/85 * rh)
    oxi = 1 + 0.50 * sn_frac * rh/100
    return arr * hum * oxi

def _form_score(t, mu):
    g1 = math.exp(-0.5*((t - 0.90)/0.07)**2)
    g2 = math.exp(-0.5*((mu - 0.50)/0.07)**2)
    return g1 * g2

# ─── Monte Carlo candidate container ─────────────────────────────────────────
@dataclass
class Candidate:
    x: float
    Eg: float
    Eg_err: float
    Eh: float
    Eh_err: float
    t: float
    mu: float

    def stability(self) -> float:
        Eh_sample = np.random.normal(self.Eh, self.Eh_err)
        return math.exp(-max(Eh_sample, 0) / 0.10)

# ─── Main screening routine ───────────────────────────────────────────────────
def screen(
    A: str, B: str,
    rh: float, temp: float,
    bg: tuple[float, float],
    bow: float = 0.30, dx: float = 0.05,
    n_mc: int = 1000
) -> pd.DataFrame:
    lo, hi = bg
    dA, dB = _summary(A), _summary(B)
    if not dA or not dB:
        return pd.DataFrame()

    # ionic radii
    def _spec(form, pool): return next(s for s in pool if s in form)
    rA = IONIC_RADII["Cs"]
    rX = IONIC_RADII[_spec(A, {"I","Br","Cl"})]
    rB0 = IONIC_RADII[_spec(A, {"Pb","Sn","Ge"})]
    rB1 = IONIC_RADII[_spec(B, {"Pb","Sn","Ge"})]

    rows = []
    for x in np.around(np.arange(0, 1+1e-6, dx), 3):
        # band‐gap and hull
        Eg = (1 - x)*dA.band_gap + x*dB.band_gap - bow*x*(1 - x)
        Eh = ((1 - x)*dA.energy_above_hull + x*dB.energy_above_hull +
              0.07*x*(1 - x))

        # tolerance & octahedral factors
        rB = (1 - x)*rB0 + x*rB1
        t = (rA + rX)/(math.sqrt(2)*(rB + rX))
        mu = rB/rX

        # Monte Carlo stability
        cand = Candidate(x, Eg, 0.15, Eh, 0.02, t, mu)
        stab_draws = np.array([cand.stability() for _ in range(n_mc)])
        stab_mean = stab_draws.mean()
        stab_ci   = np.percentile(stab_draws, [2.5, 97.5])

        # scoring
        gap_score = _gap_score(Eg, lo, hi)
        form      = _form_score(t, mu)
        env_pen   = _env_penalty(rh, temp, x)
        composite = form * stab_mean * gap_score / env_pen
        lifetime  = 5 / env_pen

        rows.append(dict(
            x=x,
            band_gap=round(Eg,3),
            gap_low=round(Eg - 1.96*0.15,3),
            gap_high=round(Eg + 1.96*0.15,3),
            stability=round(stab_mean,3),
            stab_lo=round(stab_ci[0],3),
            stab_hi=round(stab_ci[1],3),
            form_score=round(form,3),
            env_pen=round(env_pen,2),
            lifetime=round(lifetime,2),
            score=round(composite,3),
            formula=f"{A}-{B} x={x:.2f}"
        ))

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
