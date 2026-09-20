import numpy as np
import pandas as pd
from scipy.stats import qmc
import matplotlib.pyplot as plt
from pathlib import Path
OUT = Path(__file__).resolve().parent

# Reproduction of additional robustness checks added during journal revision.
# Composition notation: x = Br fraction, z = Ge fraction.

def grid_model(EgI0=1.31, EgBr0=1.77, b=0.13, EgI_z05=1.50, EgGeBr=2.50):
    xs=np.round(np.arange(0,1.0001,0.05),2)
    zs=np.round(np.arange(0,0.5001,0.05),2)
    alphaI=(EgI_z05-EgI0)/0.50
    alphaBr=EgGeBr-EgBr0
    rows=[]
    for x in xs:
        for z in zs:
            EgI=EgI0+alphaI*z
            EgBr=EgBr0+alphaBr*z
            Eg=(1-x)*EgI+x*EgBr-b*x*(1-x)
            Eh=0.0118-5.00e-4*x-1.17778e-2*z+5.5556e-4*x*z
            Eox=-1.374-0.0550*x+1.37333*z+0.055556*x*z
            rB=(1-z)*1.18+z*0.73
            rX=(1-x)*2.20+x*1.96
            t=(1.88+rX)/(np.sqrt(2)*(rB+rX))
            rows.append((x,z,Eg,Eh,Eox,t))
    return pd.DataFrame(rows, columns=['x_Br','z_Ge','Eg_eV','Ehull_proxy_eV_atom','Eox_proxy_eV_Sn','t'])

def score_arrays(df, Egstar, sig, tau, lam, t0, beta, weights):
    Eg=df.Eg_eV.to_numpy(); Eh=df.Ehull_proxy_eV_atom.to_numpy(); Eox=df.Eox_proxy_eV_Sn.to_numpy(); t=df.t.to_numpy()
    Sg=np.exp(-((Eg-Egstar)**2)/(2*sig**2))
    Sh=np.exp(-np.maximum(Eh,0)/tau)
    Sox=np.exp((Eox-Eox.max())/lam)
    St=np.exp(-beta*(t-t0)**2)
    C=np.vstack([Sg,Sh,Sox,St])
    w=np.asarray(weights)[:,None]
    return np.exp((w*np.log(C)).sum(axis=0)/w.sum())

def pareto_mask(obj):
    n=len(obj); out=np.ones(n,dtype=bool)
    for i in range(n):
        dom=np.all(obj <= obj[i], axis=1) & np.any(obj < obj[i], axis=1)
        dom[i]=False
        if dom.any(): out[i]=False
    return out

def lhs_metrics(df, n, seed=20260825, app='single'):
    sampler=qmc.LatinHypercube(d=10, seed=seed)
    U=sampler.random(n)
    egr=(1.30,1.40) if app=='single' else (1.60,1.80)
    lo=np.array([egr[0],0.10,0.015,0.25,0.90,0.0,0.5,0.5,0.5,0.5])
    hi=np.array([egr[1],0.25,0.060,1.00,0.97,50.0,2.0,2.0,2.0,2.0])
    P=lo+U*(hi-lo)
    top10=np.zeros(len(df),int); top1=np.zeros(len(df),int); ranksum=np.zeros(len(df),float)
    for p in P:
        scores=score_arrays(df,*p[:6],p[6:])
        order=np.argsort(-scores, kind='mergesort')
        ranks=np.empty(len(df),int); ranks[order]=np.arange(1,len(df)+1)
        top10[order[:24]]+=1; top1[order[0]]+=1; ranksum+=ranks
    return top10/n, top1/n, ranksum/n

def main():
    df=grid_model()
    # Reference scores
    df['SJ_score']=score_arrays(df,1.34,0.15,0.03,0.50,0.95,1.0,[1,1,1,1])
    df['Tandem_score']=score_arrays(df,1.70,0.15,0.03,0.50,0.95,1.0,[1,1,1,1])
    eoxmax=df.Eox_proxy_eV_Sn.max()
    for label,target in [('single',1.34),('tandem',1.70)]:
        obj=np.column_stack([np.abs(df.Eg_eV-target), df.Ehull_proxy_eV_atom, eoxmax-df.Eox_proxy_eV_Sn, np.abs(df.t-0.95)])
        df[f'pareto_{label}']=pareto_mask(obj)
    df.to_csv(OUT/'Supplementary_Data_S4_Pareto_Grid.csv',index=False)

    # Convergence diagnostic for the three leading iodides
    rows=[]
    for n in [500,1000,2500,5000,10000]:
        t10,t1,mr=lhs_metrics(df,n)
        for z in [0.40,0.45,0.50]:
            idx=df.index[(df.x_Br==0)&(df.z_Ge==z)][0]
            rows.append((n,z,t10[idx],t1[idx],mr[idx]))
    conv=pd.DataFrame(rows,columns=['draws','z_Ge','top10_fraction','top1_fraction','mean_rank'])
    conv.to_csv(OUT/'Supplementary_Data_S5_LHS_Convergence.csv',index=False)

    # Independent optical challenge using Hooper et al. (2026), DOI 10.1039/D6QI00612D.
    pred_sn=1.77
    pred_eq=1.77+(2.50-1.77)*0.5
    pred_ge=2.50
    challenge=pd.DataFrame([
        ('CsSnBr3','SS',0.0,pred_sn,'1.8',-0.03,'single phase'),
        ('CsSnBr3','MCS',0.0,pred_sn,'1.8',-0.03,'single phase'),
        ('CsSnBr3','HT',0.0,pred_sn,'1.8',-0.03,'single phase'),
        ('CsSn0.5Ge0.5Br3','SS',0.5,pred_eq,'1.8 and 2.2',np.nan,'two optical edges / phase segregation'),
        ('CsSn0.5Ge0.5Br3','MCS',0.5,pred_eq,'2.0',pred_eq-2.0,'nominal mixed composition'),
        ('CsSn0.5Ge0.5Br3','HT',0.5,pred_eq,'1.9',pred_eq-1.9,'nominal mixed composition'),
        ('CsGeBr3','SS',1.0,pred_ge,'2.5',0.0,'diagnostic end line, outside formal z<=0.5 grid'),
        ('CsGeBr3','MCS',1.0,pred_ge,'2.5',0.0,'diagnostic end line, outside formal z<=0.5 grid'),
        ('CsGeBr3','HT',1.0,pred_ge,'2.5',0.0,'diagnostic end line, outside formal z<=0.5 grid'),
    ],columns=['composition','route','z_Ge','model_Eg_eV','Hooper_Eg_eV','model_minus_observed_eV','note'])
    challenge.to_csv(OUT/'Supplementary_Data_S6_Hooper_Optical_Challenge.csv',index=False)

    # Figure: Pareto diagnostic (single-junction)
    fig,ax=plt.subplots(figsize=(6.6,4.6))
    y=eoxmax-df.Eox_proxy_eV_Sn
    ax.scatter(np.abs(df.Eg_eV-1.34), y, s=18, alpha=0.45, label='All 231 compositions')
    pm=df.pareto_single.to_numpy()
    ax.scatter(np.abs(df.Eg_eV[pm]-1.34), y[pm], s=32, label='4-objective Pareto set')
    ax.set_xlabel(r'$|E_g-1.34|$ (eV)')
    ax.set_ylabel(r'Oxidation penalty, $\Delta E_{ox,max}-\Delta E_{ox}$ (eV/Sn)')
    ax.legend(frameon=False)
    ax.set_title('Weight-free Pareto check: single-junction target')
    fig.tight_layout(); fig.savefig(OUT/'Revision_Figure_Pareto.png',dpi=350); plt.close(fig)

    # Figure: convergence
    fig,ax=plt.subplots(figsize=(6.6,4.6))
    for z,g in conv.groupby('z_Ge'):
        ax.plot(g.draws,100*g.top10_fraction,marker='o',label=f'z = {z:.2f}, x = 0')
    ax.set_xlabel('Latin-hypercube draws')
    ax.set_ylabel('Top-decile occupancy (%)')
    ax.set_title('Convergence of single-junction robustness metric')
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT/'Revision_Figure_Convergence.png',dpi=350); plt.close(fig)

    # Figure: external challenge
    fig,ax=plt.subplots(figsize=(6.6,4.6))
    zz=np.linspace(0,1,101); pred=1.77+(2.50-1.77)*zz
    ax.plot(zz,pred,label='Screening-model bromide end line')
    for route,vals in {'SS':[(0,1.8),(0.5,1.8),(0.5,2.2),(1,2.5)],'MCS':[(0,1.8),(0.5,2.0),(1,2.5)],'HT':[(0,1.8),(0.5,1.9),(1,2.5)]}.items():
        a=np.array(vals,float); ax.scatter(a[:,0],a[:,1],s=40,label=f'Hooper 2026: {route}')
    ax.axvline(0.5,linestyle='--',linewidth=1)
    ax.set_xlabel('Ge fraction, z (bromide series)')
    ax.set_ylabel(r'Optical band gap $E_g$ (eV)')
    ax.set_title('Independent synthesis-resolved optical challenge')
    ax.legend(frameon=False,fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/'Revision_Figure_External_Challenge.png',dpi=350); plt.close(fig)

    # TOC graphic - conceptual, no extra claims
    fig,ax=plt.subplots(figsize=(7.2,3.6))
    ax.axis('off')
    ax.text(0.08,0.78,'CsSn$_{1-z}$Ge$_z$(I$_{1-x}$Br$_x$)$_3$',ha='center',va='center',fontsize=15)
    ax.text(0.08,0.58,'231 compositions',ha='center',va='center',fontsize=11)
    ax.annotate('',xy=(0.36,0.68),xytext=(0.19,0.68),arrowprops=dict(arrowstyle='->',lw=1.8))
    ax.text(0.49,0.80,'Transparent multi-objective screen',ha='center',fontsize=12)
    ax.text(0.49,0.63,'band gap  |  hull proxy\noxidation proxy  |  strain prior',ha='center',fontsize=10)
    ax.text(0.49,0.40,'10,000 score perturbations\n+ calibration stress tests\n+ Pareto / convergence checks',ha='center',fontsize=10)
    ax.annotate('',xy=(0.78,0.68),xytext=(0.63,0.68),arrowprops=dict(arrowstyle='->',lw=1.8))
    ax.text(0.89,0.80,'Robust windows',ha='center',fontsize=12)
    ax.text(0.89,0.61,'Single junction:\nGe-rich / I-rich',ha='center',fontsize=10)
    ax.text(0.89,0.36,'Tandem top cell:\nGe-rich / moderate Br',ha='center',fontsize=10)
    ax.text(0.50,0.10,'Ge mainly shifts modeled stability; Br mainly tunes optical placement',ha='center',fontsize=10)
    fig.tight_layout(); fig.savefig(OUT/'ACS_AEM_TOC_graphic_regenerated.png',dpi=600,bbox_inches='tight'); plt.close(fig)

    print('Single-junction Pareto count:', int(df.pareto_single.sum()))
    print('Tandem Pareto count:', int(df.pareto_tandem.sum()))
    print(conv.to_string(index=False))

if __name__=='__main__': main()
