#!/usr/bin/env python3
"""Reproducible screening and robustness analysis for the submission-ready manuscript.

The model is intentionally a low-fidelity prioritisation framework. Band-gap anchors are
literature-informed; E_hull and oxidation values are interpolation-based proxies inherited
from the archived EnerMat source workflow. The script regenerates the 231-composition
reference grid, sensitivity analyses, supplementary data and all data-derived figures.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import qmc, spearmanr

OUT = Path(__file__).resolve().parent

# 1. Composition space and literature-informed band-gap model
X_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
Z_GRID = np.round(np.arange(0.0, 0.5001, 0.05), 2)

# Experimental/literature anchors (eV)
EG_I_SN = 1.31        # CsSnI3, room-temperature optical gap (Chen et al., 2019)
EG_BR_SN = 1.77       # CsSnBr3, direct optical gap (Almalawi et al., 2025)
EG_I_EQ = 1.50        # CsSn0.5Ge0.5I3 optical gap (Chen et al., 2019)
EG_BR_GE = 2.50       # CsGeBr3 end of the ~1.8-2.5 eV Sn-Ge bromide range (Kama et al., 2022)
BOWING = 0.13         # CsSn(I1-xBrx)3 band-gap bowing from Zhao & Xu (2026)
ALPHA_I = (EG_I_EQ - EG_I_SN) / 0.50
ALPHA_BR = EG_BR_GE - EG_BR_SN

# Bilinear proxy surfaces reconstructed from archived EnerMat source-workflow anchors.
# E_hull(x,z) = a + b x + c z + d xz [eV atom^-1]
EH_COEF = np.array([0.0118, -0.0005, -0.01177777777777778, 0.000555555555555556])
# Delta E_ox(x,z) = a + b x + c z + d xz [eV per Sn]
EOX_COEF = np.array([-1.374, -0.055, 1.3733333333333333, 0.05555555555555556])
EH_ANCHORS = np.array([0.01180, 0.01130, 0.005911, 0.005689])
EOX_ANCHORS = np.array([-1.374, -1.429, -0.6873, -0.7146])

# Shannon radii, Angstrom; Cs+ XII, Sn2+/Ge2+ VI, I-/Br- VI.
R_CS, R_SN, R_GE, R_I, R_BR = 1.88, 1.18, 0.73, 2.20, 1.96


def bilinear(coef, x, z):
    a, b, c, d = coef
    return a + b*x + c*z + d*x*z


def bilinear_anchors(a00, a10, a05, a105, x, z):
    v = z / 0.50
    return a00*(1-x)*(1-v) + a10*x*(1-v) + a05*(1-x)*v + a105*x*v


def band_gap(x, z, eg_i_sn=EG_I_SN, eg_br_sn=EG_BR_SN,
             eg_i_eq=EG_I_EQ, eg_br_ge=EG_BR_GE, bowing=BOWING):
    alpha_i = (eg_i_eq - eg_i_sn) / 0.50
    alpha_br = eg_br_ge - eg_br_sn
    eg_i = eg_i_sn + alpha_i*z
    eg_br = eg_br_sn + alpha_br*z
    return (1-x)*eg_i + x*eg_br - bowing*x*(1-x)


def tolerance_factor(x, z):
    r_b = (1-z)*R_SN + z*R_GE
    r_x = (1-x)*R_I + x*R_BR
    return (R_CS + r_x) / (math.sqrt(2.0)*(r_b + r_x))


def composition_label(x, z):
    sn = 1-z
    i = 1-x
    if abs(x) < 1e-12:
        return f"CsSn{sn:.2f}Ge{z:.2f}I3" if z > 0 else "CsSnI3"
    if abs(z) < 1e-12:
        return f"CsSn(I{i:.2f}Br{x:.2f})3"
    return f"CsSn{sn:.2f}Ge{z:.2f}(I{i:.2f}Br{x:.2f})3"


def build_grid(eg_i_sn=EG_I_SN, eg_br_sn=EG_BR_SN, eg_i_eq=EG_I_EQ,
               eg_br_ge=EG_BR_GE, bowing=BOWING,
               eh_anchors=EH_ANCHORS, eox_anchors=EOX_ANCHORS):
    rows = []
    for z in Z_GRID:
        for x in X_GRID:
            rows.append({
                'x_Br': x,
                'z_Ge': z,
                'composition': composition_label(x,z),
                'Eg_eV': band_gap(x,z,eg_i_sn,eg_br_sn,eg_i_eq,eg_br_ge,bowing),
                'Ehull_proxy_eV_atom': bilinear_anchors(*eh_anchors,x,z),
                'Eox_proxy_eV_per_Sn': bilinear_anchors(*eox_anchors,x,z),
                'tolerance_factor': tolerance_factor(x,z)
            })
    return pd.DataFrame(rows)


def score_components(df, eg_target, sigma_g=0.15, tau_h=0.03, lambda_ox=0.50,
                     t0=0.95, beta=1.0, weights=(1.0,1.0,1.0,1.0)):
    eg = df['Eg_eV'].to_numpy()
    eh = df['Ehull_proxy_eV_atom'].to_numpy()
    ox = df['Eox_proxy_eV_per_Sn'].to_numpy()
    tt = df['tolerance_factor'].to_numpy()
    s_g = np.exp(-0.5*((eg-eg_target)/sigma_g)**2)
    s_h = np.exp(-np.maximum(eh,0.0)/tau_h)
    ox_max = np.max(ox)
    s_o = np.exp((ox-ox_max)/lambda_ox)
    s_t = np.exp(-beta*(tt-t0)**2)
    wg, wh, wo, wt = weights
    W = wg + wh + wo + wt
    score = np.exp((wg*np.log(s_g) + wh*np.log(s_h) + wo*np.log(s_o) + wt*np.log(s_t))/W)
    return s_g, s_h, s_o, s_t, score


def add_reference_scores(df):
    out = df.copy()
    sg, sh, so, st, s = score_components(out, 1.34)
    out['S_gap_single'] = sg; out['S_hull'] = sh; out['S_ox'] = so; out['S_strain'] = st
    out['Score_single'] = s
    _, _, _, _, stnd = score_components(out, 1.70)
    out['Score_tandem'] = stnd
    out['Rank_single'] = out['Score_single'].rank(ascending=False,method='min').astype(int)
    out['Rank_tandem'] = out['Score_tandem'].rank(ascending=False,method='min').astype(int)
    return out


def scoring_sensitivity(df, target_range, seed, tag, n=10000):
    sampler = qmc.LatinHypercube(d=10, seed=seed)
    u = sampler.random(n)
    lo = np.array([target_range[0],0.10,0.015,0.25,0.90,0.0,0.5,0.5,0.5,0.5])
    hi = np.array([target_range[1],0.25,0.060,1.00,0.97,50.0,2.0,2.0,2.0,2.0])
    p = lo + u*(hi-lo)
    ncomp = len(df)
    top1 = np.zeros(ncomp,int); top5=np.zeros(ncomp,int); top10=np.zeros(ncomp,int); top20=np.zeros(ncomp,int)
    rank_sum = np.zeros(ncomp,float)
    rep_indices = {
        'z040_x000': df.index[(df.z_Ge==0.40)&(df.x_Br==0.00)][0],
        'z045_x000': df.index[(df.z_Ge==0.45)&(df.x_Br==0.00)][0],
        'z050_x000': df.index[(df.z_Ge==0.50)&(df.x_Br==0.00)][0],
        'z050_x035': df.index[(df.z_Ge==0.50)&(df.x_Br==0.35)][0],
    }
    rep_ranks = {k:np.zeros(n,int) for k in rep_indices}
    for j,row in enumerate(p):
        s = score_components(df,row[0],row[1],row[2],row[3],row[4],row[5],row[6:10])[-1]
        order = np.argsort(-s)
        ranks = np.empty(ncomp,int); ranks[order] = np.arange(1,ncomp+1)
        top1[order[0]] += 1; top5[order[:12]] += 1; top10[order[:24]] += 1; top20[order[:47]] += 1
        rank_sum += ranks
        for k,idx in rep_indices.items(): rep_ranks[k][j] = ranks[idx]
    summary = df[['x_Br','z_Ge','composition']].copy()
    summary[f'P_top1_{tag}'] = top1/n; summary[f'P_top5_{tag}'] = top5/n
    summary[f'P_top10_{tag}'] = top10/n; summary[f'P_top20_{tag}'] = top20/n
    summary[f'mean_rank_{tag}'] = rank_sum/n
    draws = pd.DataFrame(p,columns=['Eg_target_eV','sigma_g_eV','tau_h_eV_atom','lambda_ox_eV','t0','beta','w_g','w_h','w_ox','w_t'])
    for k,v in rep_ranks.items(): draws[f'rank_{k}'] = v
    return summary, draws


def descriptor_stress(df, n=5000, seed=20260827):
    # 13-dimensional coherent calibration perturbation:
    # EgI(Sn), EgBr(Sn), bowing, EgI(z=.5), EgBr(Ge), four Ehull multipliers, four Eox offsets.
    sampler = qmc.LatinHypercube(d=13, seed=seed)
    u = sampler.random(n)
    lo = np.array([1.28,1.70,0.05,1.45,2.35] + [0.80]*4 + [-0.10]*4)
    hi = np.array([1.34,1.90,0.30,1.55,2.65] + [1.20]*4 + [ 0.10]*4)
    p = lo + u*(hi-lo)
    ncomp=len(df); top1=np.zeros(ncomp,int); top10=np.zeros(ncomp,int); rank_sum=np.zeros(ncomp,float)
    for row in p:
        eg_i,eg_br,b,eg_i_eq,eg_br_ge = row[:5]
        eh = EH_ANCHORS*row[5:9]
        eox = EOX_ANCHORS+row[9:13]
        dd = build_grid(eg_i,eg_br,eg_i_eq,eg_br_ge,b,eh,eox)
        s = score_components(dd,1.34)[-1]
        order=np.argsort(-s); ranks=np.empty(ncomp,int); ranks[order]=np.arange(1,ncomp+1)
        top1[order[0]] += 1; top10[order[:24]] += 1; rank_sum += ranks
    summary=df[['x_Br','z_Ge','composition']].copy()
    summary['P_top1_descriptor_stress']=top1/n; summary['P_top10_descriptor_stress']=top10/n
    summary['mean_rank_descriptor_stress']=rank_sum/n
    draws=pd.DataFrame(p,columns=['Eg_I_Sn_eV','Eg_Br_Sn_eV','bowing_eV','Eg_I_z050_eV','Eg_Br_Ge_eV',
                                  'Eh00_multiplier','Eh10_multiplier','Eh05_multiplier','Eh105_multiplier',
                                  'Eox00_offset_eV','Eox10_offset_eV','Eox05_offset_eV','Eox105_offset_eV'])
    return summary,draws

# --- run analyses ---
grid = add_reference_scores(build_grid())
sj_summary,sj_draws = scoring_sensitivity(grid,(1.30,1.40),20260825,'single')
td_summary,td_draws = scoring_sensitivity(grid,(1.60,1.80),20260826,'tandem')
stress_summary,stress_draws = descriptor_stress(grid)

full = grid.merge(sj_summary,on=['x_Br','z_Ge','composition']).merge(td_summary,on=['x_Br','z_Ge','composition']).merge(stress_summary,on=['x_Br','z_Ge','composition'])
full.to_csv(OUT/'Supplementary_Data_S1_231_compositions.csv',index=False)
sj_draws.to_csv(OUT/'Supplementary_Data_S2a_single_sensitivity.csv',index=False)
td_draws.to_csv(OUT/'Supplementary_Data_S2b_tandem_sensitivity.csv',index=False)
stress_draws.to_csv(OUT/'Supplementary_Data_S3a_calibration_inputs.csv',index=False)
stress_summary.to_csv(OUT/'Supplementary_Data_S3b_calibration_summary.csv',index=False)

# --- figures ---
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans'})

def savefig(name):
    plt.tight_layout(); plt.savefig(OUT/name,dpi=360,bbox_inches='tight'); plt.close()

# Figure 1: workflow including robustness stage
fig,ax=plt.subplots(figsize=(10,3.2)); ax.axis('off')
labels=[('Reference data\n& literature anchors',0.03),('Explicit descriptor\nmodels',0.19),('PV target +\ncomposite score',0.35),('10,000-model\nscore sensitivity',0.51),('5,000-draw\ncalibration stress',0.67),('DFT / experiment\nvalidation',0.83)]
for txt,x0 in labels:
    box=FancyBboxPatch((x0,0.48),0.13,0.28,boxstyle='round,pad=0.015,rounding_size=0.015',fc='white',ec='black',lw=1.2)
    ax.add_patch(box); ax.text(x0+0.065,0.62,txt,ha='center',va='center',fontsize=9)
for i in range(len(labels)-1):
    x=labels[i][1]+0.13; xn=labels[i+1][1]
    ax.add_patch(FancyArrowPatch((x,0.62),(xn,0.62),arrowstyle='-|>',mutation_scale=12,lw=1.1))
ax.text(0.03,0.28,'Design space: Cs(Sn$_{1-z}$Ge$_z$)(I$_{1-x}$Br$_x$)$_3$;  x = 0–1, z = 0–0.50 (231 compositions)',fontsize=9)
ax.text(0.03,0.15,'Interpretation: transparent prioritisation. Interpolated stability descriptors are not composition-specific free energies.',fontsize=9)
savefig('Figure_1_workflow.png')

# Figure 2 representative trade-off
fig,ax=plt.subplots(figsize=(7.2,4.8))
reps=[(0,0,'CsSnI$_3$','o'),(.4,0,'CsSn(I$_{0.60}$Br$_{0.40}$)$_3$','s'),(0,.45,'CsSn$_{0.55}$Ge$_{0.45}$I$_3$','^'),(.35,.5,'CsSn$_{0.50}$Ge$_{0.50}$(I$_{0.65}$Br$_{0.35}$)$_3$','D')]
ax.axvspan(1.15,1.50,alpha=.10); ax.axvspan(1.60,1.80,alpha=.10)
for x,z,label,m in reps:
    r=grid[(grid.x_Br==x)&(grid.z_Ge==z)].iloc[0]
    size=85+110*(grid.Ehull_proxy_eV_atom.max()-r.Ehull_proxy_eV_atom)/(grid.Ehull_proxy_eV_atom.max()-grid.Ehull_proxy_eV_atom.min())
    ax.scatter(r.Eg_eV,r.Eox_proxy_eV_per_Sn,s=size,marker=m,edgecolor='black',linewidth=.7,zorder=3)
    ax.annotate(label,(r.Eg_eV,r.Eox_proxy_eV_per_Sn),xytext=(5,5),textcoords='offset points',fontsize=8)
ax.text(1.17,-.88,'single-junction\nband-gap window',fontsize=8)
ax.text(1.61,-1.24,'tandem-top-cell\nband-gap window',fontsize=8)
ax.set_xlabel('Calibrated band gap, $E_g$ (eV)'); ax.set_ylabel('Oxidation-driving-force proxy, $\\Delta E_{ox}^{proxy}$ (eV per Sn)')
ax.set_title('Less negative $\\Delta E_{ox}^{proxy}$ = weaker modelled oxidation driving force',fontsize=9)
ax.grid(alpha=.2)
savefig('Figure_2_tradeoff.png')

# Figure 3 full single-junction landscape
pivot_score=grid.pivot(index='z_Ge',columns='x_Br',values='Score_single')
pivot_prob=full.pivot(index='z_Ge',columns='x_Br',values='P_top10_single')
pivot_eg=grid.pivot(index='z_Ge',columns='x_Br',values='Eg_eV')
XX,ZZ=np.meshgrid(pivot_score.columns.values,pivot_score.index.values)
fig,axes=plt.subplots(1,2,figsize=(9.2,3.8),sharey=True)
for ax,mat,title,cblab in [(axes[0],pivot_score,'(a) Reference single-junction score','Composite score'),(axes[1],pivot_prob,'(b) Robustness across 10,000 scoring models','Probability of top-10% rank')]:
    im=ax.pcolormesh(XX,ZZ,mat.values,shading='nearest',vmin=0,vmax=1,cmap='viridis')
    cs=ax.contour(XX,ZZ,pivot_eg.values,levels=[1.4,1.6,1.8],colors='white',linewidths=.8,alpha=.85)
    ax.clabel(cs,inline=True,fontsize=7,fmt='%.1f')
    ax.set_xlabel('Br fraction, x'); ax.set_title(title,fontsize=9)
    fig.colorbar(im,ax=ax,label=cblab)
axes[0].set_ylabel('Ge fraction, z')
savefig('Figure_3_landscape_robustness.png')

# Supplementary Fig S1 tandem landscape
pvt=grid.pivot(index='z_Ge',columns='x_Br',values='Score_tandem'); pvp=full.pivot(index='z_Ge',columns='x_Br',values='P_top10_tandem')
fig,axes=plt.subplots(1,2,figsize=(9.2,3.8),sharey=True)
for ax,mat,title,cblab in [(axes[0],pvt,'(a) Reference tandem score','Composite score'),(axes[1],pvp,'(b) Tandem robustness','Probability of top-10% rank')]:
    im=ax.pcolormesh(XX,ZZ,mat.values,shading='nearest',vmin=0,vmax=1,cmap='viridis')
    cs=ax.contour(XX,ZZ,pivot_eg.values,levels=[1.6,1.7,1.8],colors='white',linewidths=.8)
    ax.clabel(cs,inline=True,fontsize=7,fmt='%.1f')
    ax.set_xlabel('Br fraction, x'); ax.set_title(title,fontsize=9); fig.colorbar(im,ax=ax,label=cblab)
axes[0].set_ylabel('Ge fraction, z')
savefig('Figure_S1_tandem_robustness.png')

# Supplementary Fig S2 rank distributions: representative single-junction and tandem cases
fig,axes=plt.subplots(1,2,figsize=(9.2,3.8),sharey=False)
xs=np.arange(1,41)
for col,label in [('rank_z040_x000','z=0.40, x=0'),('rank_z045_x000','z=0.45, x=0'),('rank_z050_x000','z=0.50, x=0')]:
    vals=sj_draws[col].values
    hist=np.array([(vals==k).mean() for k in xs])
    axes[0].plot(xs,hist,label=label)
axes[0].set_title('(a) Single-junction representatives',fontsize=9)
axes[0].set_xlabel('Rank'); axes[0].set_ylabel('Probability'); axes[0].set_xlim(1,40); axes[0].legend(frameon=False,fontsize=7); axes[0].grid(alpha=.2)
# Tandem representatives chosen around the robust z=0.50 moderate-Br band.
def tandem_rank_series(x,z):
    idx=grid.index[(grid.z_Ge==z)&(grid.x_Br==x)][0]
    # Recompute ranks from each stored draw because only selected rank columns are persisted.
    out=np.zeros(len(td_draws),dtype=int)
    for j,row in td_draws.iterrows():
        sc=score_components(grid,row.Eg_target_eV,row.sigma_g_eV,row.tau_h_eV_atom,row.lambda_ox_eV,row.t0,row.beta,(row.w_g,row.w_h,row.w_ox,row.w_t))[-1]
        order=np.argsort(-sc); ranks=np.empty(len(grid),int); ranks[order]=np.arange(1,len(grid)+1)
        out[j]=ranks[idx]
    return out
for x,label in [(0.30,'z=0.50, x=0.30'),(0.35,'z=0.50, x=0.35'),(0.40,'z=0.50, x=0.40')]:
    vals=tandem_rank_series(x,0.50)
    hist=np.array([(vals==k).mean() for k in xs])
    axes[1].plot(xs,hist,label=label)
axes[1].set_title('(b) Tandem-top-cell representatives',fontsize=9)
axes[1].set_xlabel('Rank'); axes[1].set_ylabel('Probability'); axes[1].set_xlim(1,40); axes[1].legend(frameon=False,fontsize=7); axes[1].grid(alpha=.2)
savefig('Figure_S2_rank_distributions.png')

# Supplementary Fig S3 parameter influence for z=.45,x=0
params=['Eg_target_eV','sigma_g_eV','tau_h_eV_atom','lambda_ox_eV','t0','beta','w_g','w_h','w_ox','w_t']
cor=[]
for pnm in params:
    rho,_=spearmanr(sj_draws[pnm],sj_draws['rank_z045_x000']); cor.append(rho)
fig,ax=plt.subplots(figsize=(7.4,4.4)); order=np.argsort(np.abs(cor))[::-1]
labels=[params[i].replace('_eV_atom','').replace('_eV','') for i in order]; vals=[cor[i] for i in order]
ax.barh(np.arange(len(vals)),vals); ax.set_yticks(np.arange(len(vals)),labels=labels); ax.invert_yaxis(); ax.axvline(0,color='black',lw=.7)
ax.set_xlabel('Spearman $\\rho$ with rank of z=0.45, x=0'); ax.grid(axis='x',alpha=.2)
savefig('Figure_S3_parameter_influence.png')

# Supplementary Fig S4 descriptor stress
pvs=stress_summary.pivot(index='z_Ge',columns='x_Br',values='P_top10_descriptor_stress')
fig,ax=plt.subplots(figsize=(5.4,4.2)); im=ax.pcolormesh(XX,ZZ,pvs.values,shading='nearest',vmin=0,vmax=1,cmap='viridis')
ax.set_xlabel('Br fraction, x'); ax.set_ylabel('Ge fraction, z'); ax.set_title('Top-10% robustness under 5,000 calibration stress tests',fontsize=9); fig.colorbar(im,ax=ax,label='Probability')
savefig('Figure_S4_descriptor_stress.png')

# A compact summary text for programmatic checking
best_sj=grid.loc[grid.Score_single.idxmax()]; best_td=grid.loc[grid.Score_tandem.idxmax()]
with open(OUT/'analysis_summary.txt','w') as f:
    f.write(f"Single reference maximum: x={best_sj.x_Br:.2f}, z={best_sj.z_Ge:.2f}, Eg={best_sj.Eg_eV:.6f}, score={best_sj.Score_single:.6f}\n")
    f.write(f"Tandem reference maximum: x={best_td.x_Br:.2f}, z={best_td.z_Ge:.2f}, Eg={best_td.Eg_eV:.6f}, score={best_td.Score_tandem:.6f}\n")
    for z in (0.40,0.45,0.50):
        r=full[(full.x_Br==0)&(full.z_Ge==z)].iloc[0]
        f.write(f"SJ x=0 z={z:.2f}: Ptop10={r.P_top10_single:.4f}, Ptop1={r.P_top1_single:.4f}, meanrank={r.mean_rank_single:.4f}, stressPtop10={r.P_top10_descriptor_stress:.4f}\n")
