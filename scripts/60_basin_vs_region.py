"""Is b_basin real network branching, or crypto-region/climate? basin_id is a coarse
aggregate (1200km spans), so 'different basin' may mean 'different climate zone' not
'different drainage branch'. Test: does basin add beyond geography WITHIN a narrow region?

Per species:
  (A) full model  env ~ geo + diff_basin           (as before)
  (B) restrict to pairs where BOTH points share reg_id (same region) -> re-test diff_basin
  (C) restrict to geographically TIGHT pairs (<=25km) where 'different basin' must mean a
      genuinely nearby different branch, not a far climate zone -> re-test
If diff_basin stays positive & significant in (B) and (C) -> real branch structure.
If it collapses -> it was regional/climatic aggregation, not network branching.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

MASTER=os.path.expanduser("~/Desktop/Papers/sdm-robustness/data/raw/combined_data_true_master.csv")
N_PCA=10; NPERM=1500; NMAX=1800
SPECIES=["Pontastacus leptodactylus","Austropotamobius torrentium",
         "Astacus astacus","Austropotamobius pallipes"]

def haversine_km(lon,lat):
    lon=np.radians(lon); lat=np.radians(lat)
    dlon=lon[:,None]-lon[None,:]; dlat=lat[:,None]-lat[None,:]
    a=np.sin(dlat/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    return 2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))

def z(v): v=np.asarray(v,float); return (v-v.mean())/(v.std()+1e-12)

def fit_bbasin(D_env,iu,y_mask,Xcols,n,rng,nperm):
    y=z(D_env[iu][y_mask]); X=Xcols
    coef,_,_,_=np.linalg.lstsq(X,y,rcond=None); b=coef[-1]
    cnt=0
    for _ in range(nperm):
        p=rng.permutation(n)
        yp=z(D_env[np.ix_(p,p)][iu][y_mask])
        cp,_,_,_=np.linalg.lstsq(X,yp,rcond=None)
        if cp[-1]>=b: cnt+=1
    return b,(cnt+1)/(nperm+1)

def main():
    cm=pd.read_csv(MASTER,encoding="latin-1",low_memory=False)
    cm=cm[cm["Accuracy"].astype(str).str.strip().str.lower().eq("high")]
    env_cols=[c for c in cm.columns if c.startswith(("l_","u_"))]
    has_reg = "reg_id" in cm.columns
    need=["long_snap","lat_snap","subc_id","basin_id"]+(["reg_id"] if has_reg else [])
    cm=cm.dropna(subset=need); cm=cm[cm["long_snap"].abs()>0.01].drop_duplicates("subc_id")
    rng=np.random.default_rng(0)
    print(f"reg_id present: {has_reg}\n")
    print(f"{'species':28s} | {'(A) all: b_basin':>16} {'p':>6} | {'(B) same-reg':>13} {'p':>6} | {'(C) <=25km':>11} {'p':>6}")
    print("-"*92)
    for sp in SPECIES:
        s=cm[cm["Crayfish_scientific_name"]==sp]; s=s[s[env_cols].notna().all(axis=1)]
        if len(s)>NMAX: s=s.sample(NMAX,random_state=0)
        s=s.reset_index(drop=True)
        if len(s)<150: print(f"{sp:28s} too few"); continue
        er=PCA(n_components=N_PCA,random_state=0).fit_transform(StandardScaler().fit_transform(s[env_cols].to_numpy(float)))
        D_env=squareform(pdist(er)); D_geo=haversine_km(s.long_snap.to_numpy(),s.lat_snap.to_numpy())
        bas=s["basin_id"].to_numpy(); reg=s["reg_id"].to_numpy() if has_reg else None
        n=len(s); iu=np.triu_indices(n,k=1)
        gp=D_geo[iu]; diffb=(bas[iu[0]]!=bas[iu[1]]).astype(float)
        # (A) all pairs geo<=50
        mA=gp<=50; XA=np.c_[np.ones(mA.sum()),z(gp[mA]),diffb[mA]]
        bA,pA=fit_bbasin(D_env,iu,mA,XA,n,rng,NPERM)
        # (B) same region only
        if has_reg:
            samereg=(reg[iu[0]]==reg[iu[1]])
            mB=(gp<=50)&samereg
            if mB.sum()>50 and diffb[mB].sum()>10:
                XB=np.c_[np.ones(mB.sum()),z(gp[mB]),diffb[mB]]
                bB,pB=fit_bbasin(D_env,iu,mB,XB,n,rng,NPERM)
            else: bB,pB=np.nan,np.nan
        else: bB,pB=np.nan,np.nan
        # (C) tight geo <=25km
        mC=gp<=25
        if mC.sum()>50 and diffb[mC].sum()>10:
            XC=np.c_[np.ones(mC.sum()),z(gp[mC]),diffb[mC]]
            bC,pC=fit_bbasin(D_env,iu,mC,XC,n,rng,NPERM)
        else: bC,pC=np.nan,np.nan
        def f(b,p): 
            fl="*" if (b>0.05 and p<0.05) else ""
            return f"{b:+7.3f}{fl:1s} {p:6.3f}"
        print(f"{sp:28s} | {f(bA,pA)} | {f(bB,pB)} | {f(bC,pC)}")
    print("\nIf b_basin stays positive/significant in (B) same-region AND (C) tight-geo ->")
    print("real drainage-branch structure. If it collapses in (B)/(C) -> it was region/climate.")

if __name__ == "__main__":
    main()
