# ─── backend/perovskite_utils.py  (EnerMat v9.6, 2025-07-15) ───────────────────
from __future__ import annotations
import math, os
from functools import lru_cache

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from mp_api.client import MPRester
from pymatgen.core import Composition

# ───────────────────────────────────────── API key ────────────────────────────
load_dotenv()
API_KEY = os.getenv("MP_API_KEY") or st.secrets.get("MP_API_KEY")
if not API_KEY or len(API_KEY) != 32:
    raise RuntimeError("🛑 32-character MP_API_KEY missing")

mpr = MPRester(API_KEY)

# ───────────────────────── calibration + constants ────────────────────────────
END_MEMBERS = ["CsSnI3", "CsSnBr3", "CsSnCl3", "CsGeBr3", "CsGeCl3"]

CALIBRATED_GAPS = dict(CzSnBr3=1.79, CsSnCl3=2.83, CsSnI3=1.00,
                       CsGeBr3=2.20, CsGeCl3=3.30)
GAP_OFFSET = {"I": +0.52, "Br": +0.88, "Cl": +1.10}

IONIC_RADII = {"Cs":1.88,"Sn":1.18,"Ge":0.73,"I":2.20,"Br":1.96,"Cl":1.81}
K_T_EFF = 0.20        # soft-penalty “kT” (eV)

score_band_gap = lambda Eg, lo, hi: 1.0 if lo <= Eg <= hi else 0.0

# ────────────────────────────── helpers ───────────────────────────────────────
def fetch_mp_data(fml:str, fields:list[str]):
    docs = mpr.summary.search(formula=fml, fields=tuple(fields))
    if not docs:  return None
    ent = docs[0]
    out = {f:getattr(ent,f,None) for f in fields}
    if "band_gap" in fields:
        if fml in CALIBRATED_GAPS:
            out["band_gap"] = CALIBRATED_GAPS[fml]
        else:
            h = next(h for h in ("I","Br","Cl") if h in fml)
            out["band_gap"] = (out["band_gap"] or 0.0) + GAP_OFFSET[h]
    return out

@lru_cache(maxsize=64)
-    def ΔH(formula: str):
-        d = fetch_mp_data(formula, ["formation_energy_per_atom"])
-        if not d or d["formation_energy_per_atom"] is None:
-            raise ValueError(f"Missing formation-energy for {formula}")
-        comp = Composition(formula)
-        return d["formation_energy_per_atom"] * comp.num_atoms
+    def ΔH(formula: str):
+        doc = fetch_mp_data(formula, ["formation_energy_per_atom"])
+        if not doc or doc["formation_energy_per_atom"] is None:
+            raise ValueError(f"Missing formation-energy for {formula}")
+        comp = Composition(formula)
+        return doc["formation_energy_per_atom"] * comp.num_atoms

# ───────────────────────────── binary screen ──────────────────────────────────
def screen_binary(A,B,rh,temp,bg,bow,dx,*,z:float=0.0):
    return mix_abx3(A,B,rh,temp,bg,bow,dx,z=z)

def mix_abx3(A,B,rh,temp,bg,bow,dx,z=0.0,alpha=1,beta=1)->pd.DataFrame:
    lo,hi = bg
    dA,dB = fetch_mp_data(A,["band_gap","energy_above_hull"]), \
            fetch_mp_data(B,["band_gap","energy_above_hull"])
    if not(dA and dB): return pd.DataFrame()

    hal = next(h for h in ("I","Br","Cl") if h in A)
    rA,rB,rX = (IONIC_RADII[k] for k in ("Cs","Sn",hal))
    oxA,oxB = oxidation_energy(A),oxidation_energy(B)

    rows=[]
    for x in np.arange(0,1+1e-9,dx):
        Eg  = (1-x)*dA["band_gap"]+x*dB["band_gap"]-bow*x*(1-x)
        Eh  = (1-x)*dA["energy_above_hull"]+x*dB["energy_above_hull"]
        dEox= (1-x)*oxA + x*oxB
        raw = score_band_gap(Eg,lo,hi)*math.exp(-Eh/(alpha*0.0259))* \
              math.exp(dEox/K_T_EFF)*math.exp(-beta*abs((rA+rX)/(math.sqrt(2)*(rB+rX))-0.95))
        rows.append({"x":round(x,3),"z":round(z,2),"Eg":round(Eg,3),
                     "Ehull":round(Eh,4),"Eox":round(dEox,3),"raw":raw,
                     "formula":f"{A}-{B} x={x:.2f} z={z:.2f}"})
    if not rows: return pd.DataFrame()
    m=max(r["raw"] for r in rows) or 1
    for r in rows: r["score"]=round(r["raw"]/m,3); del r["raw"]
    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)

# ───────────────────────────── ternary screen ────────────────────────────────
def screen_ternary(A,B,C,rh,temp,bg,bows,*,dx=0.10,dy=0.10)->pd.DataFrame:
    dA,dB,dC=[fetch_mp_data(f,["band_gap","energy_above_hull"])
              for f in (A,B,C)]
    if not(dA and dB and dC): return pd.DataFrame()
    oxA,oxB,oxC=[oxidation_energy(f) for f in (A,B,C)]
    lo,hi=bg; rows=[]
    for x in np.arange(0,1+1e-9,dx):
        for y in np.arange(0,1-x+1e-9,dy):
            z=1-x-y
            Eg=(z*dA["band_gap"]+x*dB["band_gap"]+y*dC["band_gap"]
                 -bows["AB"]*x*z - bows["AC"]*y*z - bows["BC"]*x*y)
            Eh=z*dA["energy_above_hull"]+x*dB["energy_above_hull"]+y*dC["energy_above_hull"]
            dEox=z*oxA + x*oxB + y*oxC
            raw=score_band_gap(Eg,lo,hi)*math.exp(-Eh/0.0518)*math.exp(dEox/K_T_EFF)
            rows.append({"x":round(x,3),"y":round(y,3),"Eg":round(Eg,3),
                         "Ehull":round(Eh,4),"Eox":round(dEox,3),"raw":raw,
                         "formula":f"{A}-{B}-{C} x={x:.2f} y={y:.2f}"})
    if not rows: return pd.DataFrame()
    m=max(r["raw"] for r in rows) or 1
    for r in rows: r["score"]=round(r["raw"]/m,3); del r["raw"]
    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------
_summary = fetch_mp_data
