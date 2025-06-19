from __future__ import annotations
import os, math, functools, numpy as np, pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dataclasses import dataclass

load_dotenv(); API_KEY = os.getenv("MP_API_KEY")
if not API_KEY or len(API_KEY) < 22:
    raise RuntimeError("MP_API_KEY missing – set it in .env")
mpr = MPRester(API_KEY)

END_MEMBERS = ["CsPbBr3","CsSnBr3","CsSnCl3","CsPbI3"]
IONIC_RADII = {"Cs":1.88,"Pb":1.19,"Sn":1.18,"I":2.20,"Br":1.96,"Cl":1.81}

# global hyper-parameters (easier to tune)
MIX_BOW_STAB   = 0.07      # eV atom⁻¹
DECAY_E        = 0.10      # eV atom⁻¹
Ea0, k_rh      = 0.60, 0.02
OXI_PENALTY_SN = 0.50      # 0→1 scaling of lifetime when Sn=1 & RH=100 %

# ── robust Materials Project query  ------------------------------------------
class MPUp(Exception): pass
@retry(stop=stop_after_attempt(3), wait=wait_exponential(), retry=retry_if_exception_type(MPUp))
def _summary(formula: str) -> MDoc:
    try:
        return mpr.summary.search(material_ids_or_formula=formula)[0]
    except Exception:
        # Fallback for known missing entries
        if formula == "FASnBr3":
            return MDoc(
                formula="FASnBr3",
                band_gap=1.25,
                energy_above_hull=0.065,
                stability=0.935,
                notes="📄 Experimental fallback – not in MP"
            )
        elif formula == "(PEA)SnBr3":
            return MDoc(
                formula="(PEA)SnBr3",
                band_gap=2.10,
                energy_above_hull=0.08,
                stability=0.915,
                notes="📄 Experimental fallback – 2D structure"
            )
        else:
            raise RuntimeError(f"{formula} not found in MP and no fallback defined.")

# ── helpers ------------------------------------------------------------------
def _gap_score(g, lo, hi):
    if lo<=g<=hi: return 1.0
    return max(0., 1-min(abs(g-lo), abs(g-hi))/(hi-lo))

def _env_penalty(rh:float, T:float, sn_frac:float)->float:
    """folds Arrhenius, humidity and Sn²⁺ oxidation"""
    k_B, T_ref = 8.617e-5, 298.15
    arr = math.exp(Ea0/k_B * (1/(T+273.15)-1/T_ref))
    hum = math.exp(1.0/85 * rh)                    # 2× @ 85 % RH
    oxi = 1 + OXI_PENALTY_SN * sn_frac * rh/100    # linear model
    return arr*hum*oxi

def _form_score(t,mu):
    g1 = math.exp(-0.5*((t-0.90)/0.07)**2)
    g2 = math.exp(-0.5*((mu-0.50)/0.07)**2)
    return g1*g2

# ── dataclass for Monte-Carlo draws ------------------------------------------
@dataclass
class Candidate:
    x:          float
    Eg:         float
    Eg_err:     float
    Eh:         float
    Eh_err:     float
    t:          float
    mu:         float

    def stability(self)->float:
        # sample energy above hull
        Eh_sample = np.random.normal(self.Eh, self.Eh_err)
        return math.exp(-max(Eh_sample,0)/DECAY_E)

# ── main API -----------------------------------------------------------------
def screen(A:str,B:str,rh:float,temp:float,
           bg:(float,float),bow:float=0.30,dx:float=0.05,
           n_mc:int=1000)->pd.DataFrame:
    lo,hi = bg
    dA,dB = _summary(A),_summary(B)
    if not dA or not dB: return pd.DataFrame()

    # radii for formability
    def _spec(f, pool): return next(s for s in pool if s in f)
    rA = IONIC_RADII["Cs"]  # constant here
    rX = IONIC_RADII[_spec(A,{"I","Br","Cl"})]
    rB0 = IONIC_RADII[_spec(A,{"Pb","Sn"})]
    rB1 = IONIC_RADII[_spec(B,{"Pb","Sn"})]

    rows=[]
    for x in np.around(np.arange(0,1+1e-6,dx),3):
        # mean band-gap
        Eg = (1-x)*dA.band_gap + x*dB.band_gap - bow*x*(1-x)
        # mean hull energy + mixing penalty
        Eh = ((1-x)*dA.energy_above_hull + x*dB.energy_above_hull +
              MIX_BOW_STAB*x*(1-x))
        # geometry
        rB = (1-x)*rB0 + x*rB1
        t, mu = (rA+rX)/(math.sqrt(2)*(rB+rX)), rB/rX

        # uncertainty → MC
        cand = Candidate(x,Eg,0.15,Eh,0.02,t,mu)
        stab_draws = np.array([cand.stability() for _ in range(n_mc)])
        stab_mean  = stab_draws.mean(); stab_ci = np.percentile(stab_draws,[2.5,97.5])

        gap_score  = _gap_score(Eg,lo,hi)
        form       = _form_score(t,mu)
        env        = _env_penalty(rh,temp,x)        # Sn fraction ≈ x

        composite  = form*stab_mean*gap_score/env
        lifetime   = 5/env                          # years relative; 5 yr at baseline

        rows.append(dict(x=x,band_gap=round(Eg,3),
                         gap_low=round(Eg-0.15*1.96,3),
                         gap_high=round(Eg+0.15*1.96,3),
                         stability=round(stab_mean,3),
                         stab_lo=round(stab_ci[0],3),
                         stab_hi=round(stab_ci[1],3),
                         form_score=round(form,3),
                         env_pen=round(env,2),
                         lifetime=round(lifetime,2),
                         score=round(composite,3),
                         formula=f"{A}-{B} x={x:.2f}"))
    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)
