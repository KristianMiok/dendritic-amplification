"""Give the ORIGINAL multi-species basin-proxy result the confound test it never got.
GRAPH-FREE (immune to the bbox / 48%-connectivity problems of the real-distance pipeline).

Per species, MMRR-style: env_dist ~ geo_dist + different_basin_indicator, Mantel node-permutation.
b_basin = extra env-divergence for cross-basin pairs, CONTROLLING geography.
Set up to fail: if b_basin ~0 / n.s., the basin-proxy gradient was geo-confounded too.
Real result: b_basin positive & significant, varying across species (pallipes weak, leptodactylus strong).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

MASTER=os.path.expanduser("~/Desktop/Papers/sdm-robustness/data/raw/combined_data_true_master.csv")
N_PCA=10; NPERM=2000; GEO_CAP=50.0; NMAX=1800
SPECIES=["Pontastacus leptodactylus","Austropotamobius torrentium",
         "Astacus astacus","Austropotamobius pallipes"]

def haversine_km(lon,lat):
    lon=np.radians(lon); lat=np.radians(lat)
    dlon=lon[:,None]-lon[None,:]; dlat=lat[:,None]-lat[None,:]
    a=np.sin(dlat/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    return 2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))

def z(v): v=np.asarray(v,float); return (v-v.mean())/(v.std()+1e-12)

def main():
    cm=pd.read_csv(MASTER,encoding="latin-1",low_memory=False)
    cm=cm[cm["Accuracy"].astype(str).str.strip().str.lower().eq("high")]
    env_cols=[c for c in cm.columns if c.startswith(("l_","u_"))]
    cm=cm.dropna(subset=["long_snap","lat_snap","subc_id","basin_id"])
    cm=cm[cm["long_snap"].abs()>0.01].drop_duplicates("subc_id")
    rng=np.random.default_rng(0)

    print(f"MMRR (graph-free): env ~ geo + different_basin | Mantel perm, geo<= {GEO_CAP}km\n")
    print(f"{'species':30s} {'N':>5} {'b_geo':>7} {'b_basin':>8} {'p_basin':>8} {'n_diffpairs':>11}")
    print("-"*76)
    for sp in SPECIES:
        s=cm[cm["Crayfish_scientific_name"]==sp]
        s=s[s[env_cols].notna().all(axis=1)]
        if len(s)>NMAX: s=s.sample(NMAX,random_state=0)
        s=s.reset_index(drop=True)
        if len(s)<150: print(f"{sp:30s} too few"); continue

        er=PCA(n_components=N_PCA,random_state=0).fit_transform(StandardScaler().fit_transform(s[env_cols].to_numpy(float)))
        D_env=squareform(pdist(er)); D_geo=haversine_km(s.long_snap.to_numpy(),s.lat_snap.to_numpy())
        bas=s["basin_id"].to_numpy()
        n=len(s); iu=np.triu_indices(n,k=1)
        gp=D_geo[iu]; ep=D_env[iu]; diffb=(bas[iu[0]]!=bas[iu[1]]).astype(float)
        m=gp<=GEO_CAP
        y=z(ep[m]); Xg=z(gp[m]); Xb=diffb[m]  # keep basin indicator as 0/1 (not z) for interpretability
        X=np.c_[np.ones_like(y),Xg,Xb]
        coef,_,_,_=np.linalg.lstsq(X,y,rcond=None)
        b_geo,b_bas=coef[1],coef[2]
        # Mantel node-permutation for b_basin
        cnt=0
        for _ in range(NPERM):
            p=rng.permutation(n)
            yp=z(D_env[np.ix_(p,p)][iu][m])
            cp,_,_,_=np.linalg.lstsq(X,yp,rcond=None)
            if cp[2]>=b_bas: cnt+=1
        pval=(cnt+1)/(NPERM+1)
        flag="***" if (b_bas>0.05 and pval<0.05) else ("" if b_bas<0.03 else " ?")
        print(f"{sp:30s} {n:5d} {b_geo:+7.3f} {b_bas:+8.3f} {pval:8.3f} {int(diffb[m].sum()):11d}  {flag}")
    print("\nb_basin *** (positive, significant) with geo in the model = basin structure adds beyond")
    print("geography = the multi-species result is REAL and graph-free. If b_basin ~0 -> geo-confounded.")

if __name__ == "__main__":
    main()
