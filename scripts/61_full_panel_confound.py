"""Full-panel graph-free confound test: env ~ geo + different_basin, with same-region (B)
and tight-geo (C) controls, across ALL usable crayfish species. Flags native-compact vs
invasive-spread species (where 'different basin' may mean different continent).
Goal: is the species-varying dendritic niche signal systematic across many species?
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

MASTER=os.path.expanduser("~/Desktop/Papers/sdm-robustness/data/raw/combined_data_true_master.csv")
N_PCA=10; NPERM=1000; NMAX=1600; MIN_N=150; MIN_BASINS=3

def haversine_km(lon,lat):
    lon=np.radians(lon); lat=np.radians(lat)
    dlon=lon[:,None]-lon[None,:]; dlat=lat[:,None]-lat[None,:]
    a=np.sin(dlat/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    return 2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))

def z(v): v=np.asarray(v,float); return (v-v.mean())/(v.std()+1e-12)

def fit_b(D_env,iu,mask,X,n,rng,nperm):
    y=z(D_env[iu][mask]); coef,_,_,_=np.linalg.lstsq(X,y,rcond=None); b=coef[-1]
    cnt=0
    for _ in range(nperm):
        p=rng.permutation(n); yp=z(D_env[np.ix_(p,p)][iu][mask])
        cp,_,_,_=np.linalg.lstsq(X,yp,rcond=None)
        if cp[-1]>=b: cnt+=1
    return b,(cnt+1)/(nperm+1)

def main():
    cm=pd.read_csv(MASTER,encoding="latin-1",low_memory=False)
    cm=cm[cm["Accuracy"].astype(str).str.strip().str.lower().eq("high")]
    env_cols=[c for c in cm.columns if c.startswith(("l_","u_"))]
    has_reg="reg_id" in cm.columns
    need=["long_snap","lat_snap","subc_id","basin_id"]+(["reg_id"] if has_reg else [])
    cm=cm.dropna(subset=need); cm=cm[cm["long_snap"].abs()>0.01].drop_duplicates("subc_id")
    rng=np.random.default_rng(0)

    # candidate species
    cnt=cm.groupby("Crayfish_scientific_name").agg(n=("subc_id","nunique"),
          nb=("basin_id","nunique"),
          span=("long_snap", lambda s:(s.max()-s.min())*111)).reset_index()
    cand=cnt[(cnt.n>=MIN_N)&(cnt.nb>=MIN_BASINS)].sort_values("n",ascending=False)["Crayfish_scientific_name"].tolist()
    print(f"testing {len(cand)} species (graph-free; env~geo+diff_basin; +same-region +tight-geo)\n")
    print(f"{'species':30s} {'N':>5} {'span':>6} | {'(A)bBas':>8} {'pA':>5} | {'(B)sReg':>8} {'pB':>5} | {'(C)<25':>7} {'pC':>5}")
    print("-"*94)
    rows=[]
    for sp in cand:
        s=cm[cm["Crayfish_scientific_name"]==sp]; s=s[s[env_cols].notna().all(axis=1)]
        span=(s.long_snap.max()-s.long_snap.min())*111
        if len(s)>NMAX: s=s.sample(NMAX,random_state=0)
        s=s.reset_index(drop=True)
        if len(s)<MIN_N: continue
        er=PCA(n_components=N_PCA,random_state=0).fit_transform(StandardScaler().fit_transform(s[env_cols].to_numpy(float)))
        D_env=squareform(pdist(er)); D_geo=haversine_km(s.long_snap.to_numpy(),s.lat_snap.to_numpy())
        bas=s["basin_id"].to_numpy(); reg=s["reg_id"].to_numpy() if has_reg else None
        n=len(s); iu=np.triu_indices(n,k=1); gp=D_geo[iu]; diffb=(bas[iu[0]]!=bas[iu[1]]).astype(float)
        mA=gp<=50
        if diffb[mA].sum()<10: continue
        XA=np.c_[np.ones(mA.sum()),z(gp[mA]),diffb[mA]]; bA,pA=fit_b(D_env,iu,mA,XA,n,rng,NPERM)
        if has_reg:
            mB=(gp<=50)&(reg[iu[0]]==reg[iu[1]])
            if mB.sum()>50 and diffb[mB].sum()>10:
                XB=np.c_[np.ones(mB.sum()),z(gp[mB]),diffb[mB]]; bB,pB=fit_b(D_env,iu,mB,XB,n,rng,NPERM)
            else: bB,pB=np.nan,np.nan
        else: bB,pB=np.nan,np.nan
        mC=gp<=25
        if mC.sum()>50 and diffb[mC].sum()>10:
            XC=np.c_[np.ones(mC.sum()),z(gp[mC]),diffb[mC]]; bC,pC=fit_b(D_env,iu,mC,XC,n,rng,NPERM)
        else: bC,pC=np.nan,np.nan
        def f(b,p):
            if np.isnan(b): return f"{'--':>8} {'--':>5}"
            fl="*" if (b>0.05 and p<0.05) else ""
            return f"{b:+7.3f}{fl} {p:5.3f}"
        tag="~spread" if span>8000 else ""
        print(f"{sp:30s} {n:5d} {span:6.0f} | {f(bA,pA)} | {f(bB,pB)} | {f(bC,pC)} {tag}")
        rows.append((sp,span,bA,pA,bB,pB,bC,pC))
    # summary
    sig=[r for r in rows if r[2]>0.05 and r[3]<0.05 and (np.isnan(r[4]) or (r[4]>0.03))]
    print(f"\n{len(rows)} species tested. Positive & sig in (A): "
          f"{sum(1 for r in rows if r[2]>0.05 and r[3]<0.05)}. "
          f"Null (b~0): {sum(1 for r in rows if abs(r[2])<0.05)}.")
    print("Look for: signal in NATIVE-COMPACT species (span<8000) surviving (B)&(C); '~spread' = interpret cautiously.")

if __name__ == "__main__":
    main()
