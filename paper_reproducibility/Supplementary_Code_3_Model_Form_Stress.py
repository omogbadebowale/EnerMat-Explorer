#!/usr/bin/env python3
"""Nonlinear model-form stress test for the Sn-Ge halide perovskite screening study."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import qmc
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT/"data"; FIG_DIR=ROOT/"figures"
DATA_DIR.mkdir(parents=True,exist_ok=True); FIG_DIR.mkdir(parents=True,exist_ok=True)

X_GRID=np.round(np.arange(0.0,1.0001,0.05),2); Z_GRID=np.round(np.arange(0.0,0.5001,0.05),2)
EH_ANCHORS=np.array([0.01180,0.01130,0.005911,0.005689],float)
EOX_ANCHORS=np.array([-1.374,-1.429,-0.6873,-0.7146],float)
EH_SPAN=EH_ANCHORS.max()-EH_ANCHORS.min(); EOX_SPAN=EOX_ANCHORS.max()-EOX_ANCHORS.min()
CURVATURE_FRACTION=0.25; SEED=20260919; N_DRAWS=10000

def bilinear_anchors(a,x,z):
    q=z/0.50
    return a[0]*(1-x)*(1-q)+a[1]*x*(1-q)+a[2]*(1-x)*q+a[3]*x*q

def build_grid(params,curvature):
    eg_i_sn,eg_br_sn,bowing,eg_i_z05,eg_br_ge=params[:5]
    eh=EH_ANCHORS*params[5:9]; eox=EOX_ANCHORS+params[9:13]
    a_x,a_z,a_xz,b_x,b_z,b_xz=curvature
    alpha_i=(eg_i_z05-eg_i_sn)/0.50; alpha_br=eg_br_ge-eg_br_sn; rows=[]
    for z in Z_GRID:
        q=z/0.50; phi_z=4*q*(1-q)
        for x in X_GRID:
            phi_x=4*x*(1-x); phi_xz=phi_x*phi_z
            eg_i=eg_i_sn+alpha_i*z; eg_br=eg_br_sn+alpha_br*z
            eg=(1-x)*eg_i+x*eg_br-bowing*x*(1-x)
            eh_val=bilinear_anchors(eh,x,z)+a_x*phi_x+a_z*phi_z+a_xz*phi_xz
            eox_val=bilinear_anchors(eox,x,z)+b_x*phi_x+b_z*phi_z+b_xz*phi_xz
            r_b=(1-z)*1.18+z*0.73; r_x=(1-x)*2.20+x*1.96
            t=(1.88+r_x)/(np.sqrt(2.0)*(r_b+r_x))
            rows.append((x,z,eg,eh_val,eox_val,t))
    return np.asarray(rows,float)

def score(grid,eg_target):
    eg,eh,eox,t=grid[:,2],grid[:,3],grid[:,4],grid[:,5]
    s_g=np.exp(-0.5*((eg-eg_target)/0.15)**2); s_h=np.exp(-np.maximum(eh,0.0)/0.03)
    s_ox=np.exp((eox-eox.max())/0.50); s_t=np.exp(-(t-0.95)**2)
    return (s_g*s_h*s_ox*s_t)**0.25

def run():
    sampler=qmc.LatinHypercube(d=19,seed=SEED); u=sampler.random(N_DRAWS)
    lo=np.array([1.28,1.70,0.05,1.45,2.35]+[0.80]*4+[-0.10]*4+[-CURVATURE_FRACTION*EH_SPAN]*3+[-CURVATURE_FRACTION*EOX_SPAN]*3)
    hi=np.array([1.34,1.90,0.30,1.55,2.65]+[1.20]*4+[0.10]*4+[CURVATURE_FRACTION*EH_SPAN]*3+[CURVATURE_FRACTION*EOX_SPAN]*3)
    draws=lo+u*(hi-lo); labels=np.array([(x,z) for z in Z_GRID for x in X_GRID],float); ncomp=len(labels); rows=[]
    for target,app in [(1.34,"single-junction"),(1.70,"tandem-top-cell")]:
        top10=np.zeros(ncomp,int); top1=np.zeros(ncomp,int); rank_sum=np.zeros(ncomp,float)
        for draw in draws:
            grid=build_grid(draw[:13],draw[13:]); scores=score(grid,target); order=np.argsort(-scores,kind="mergesort")
            ranks=np.empty(ncomp,int); ranks[order]=np.arange(1,ncomp+1)
            top10[order[:24]]+=1; top1[order[0]]+=1; rank_sum+=ranks
        for i,(x,z) in enumerate(labels): rows.append((app,x,z,top10[i]/N_DRAWS,top1[i]/N_DRAWS,rank_sum[i]/N_DRAWS))
    out=pd.DataFrame(rows,columns=["application","x_Br","z_Ge","top10_fraction","top1_fraction","mean_rank"])
    out.to_csv(DATA_DIR/"Supplementary_Data_S7a_Model_Form_Stress.csv",index=False)
    cols=["Eg_I_Sn_eV","Eg_Br_Sn_eV","bowing_eV","Eg_I_z050_eV","Eg_Br_Ge_eV","Eh00_multiplier","Eh10_multiplier","Eh05_multiplier","Eh105_multiplier","Eox00_offset_eV","Eox10_offset_eV","Eox05_offset_eV","Eox105_offset_eV","Eh_curv_x_eV_atom","Eh_curv_z_eV_atom","Eh_curv_xz_eV_atom","Eox_curv_x_eV_Sn","Eox_curv_z_eV_Sn","Eox_curv_xz_eV_Sn"]
    pd.DataFrame(draws,columns=cols).to_csv(DATA_DIR/"Supplementary_Data_S7b_Model_Form_Draws.csv",index=False)

    plt.rcParams.update({"font.size":10,"axes.titlesize":11,"axes.labelsize":11,"xtick.labelsize":9,"ytick.labelsize":9})
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.1),sharey=True)
    for ax,app,title in [(axes[0],"single-junction","(a) Single-junction target"),(axes[1],"tandem-top-cell","(b) Tandem-top-cell target")]:
        d=out[out.application==app]; p=d.pivot(index="z_Ge",columns="x_Br",values="top10_fraction")
        xx,zz=np.meshgrid(p.columns.to_numpy(),p.index.to_numpy()); im=ax.pcolormesh(xx,zz,p.to_numpy(),shading="nearest",vmin=0,vmax=1)
        ax.set_xlabel("Br fraction, x"); ax.set_title(title); cb=fig.colorbar(im,ax=ax); cb.set_label("Top-decile occupancy fraction")
    axes[0].set_ylabel("Ge fraction, z"); fig.suptitle("Combined calibration and nonlinear model-form stress test",fontsize=12)
    fig.tight_layout(); fig.savefig(FIG_DIR/"Figure_7_Model_Form_Stress.png",dpi=600,bbox_inches="tight"); fig.savefig(FIG_DIR/"Figure_7_Model_Form_Stress.svg",bbox_inches="tight"); plt.close(fig)

    def pick(app,x,z): return out[(out.application==app)&(out.x_Br==x)&(out.z_Ge==z)].iloc[0]
    checks=[("single-junction",0.00,0.40),("single-junction",0.00,0.45),("single-junction",0.00,0.50),("tandem-top-cell",0.35,0.50),("tandem-top-cell",0.40,0.45),("tandem-top-cell",0.45,0.45)]
    with open(DATA_DIR/"model_form_summary.txt","w") as f:
        f.write(f"N_DRAWS={N_DRAWS}; seed={SEED}\\n")
        f.write(f"Ehull anchor span={EH_SPAN:.6f} eV/atom; each curvature coefficient range=±{CURVATURE_FRACTION*EH_SPAN:.6f} eV/atom\\n")
        f.write(f"Eox anchor span={EOX_SPAN:.6f} eV/Sn; each curvature coefficient range=±{CURVATURE_FRACTION*EOX_SPAN:.6f} eV/Sn\\n")
        for app,x,z in checks:
            r=pick(app,x,z); f.write(f"{app}: x={x:.2f}, z={z:.2f}, top10={r.top10_fraction:.4f}, top1={r.top1_fraction:.4f}, mean_rank={r.mean_rank:.4f}\\n")
    print(out.sort_values(["application","top10_fraction","mean_rank"],ascending=[True,False,True]).groupby("application").head(10).to_string(index=False))

if __name__=="__main__":
    run()
