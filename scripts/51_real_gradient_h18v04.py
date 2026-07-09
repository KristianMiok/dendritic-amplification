"""REAL along-network distance Delta-env(d) gradient for the 3 clean species in h18v04.
net position = out_dist (distance-to-outlet, m) from Hydrography90m. subc_id == stream (confirmed).
Question: among geographically-near pairs, does larger NETWORK distance -> larger env distance,
and does that grow as geography tightens? (the real amplification test, no proxy)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pyogrio import read_dataframe
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu

GPKG="data/raw/h90m/order_vect_segment_h18v04.gpkg"
MASTER=os.path.expanduser("~/Desktop/Papers/sdm-robustness/data/raw/combined_data_true_master.csv")
SPECIES=["Austropotamobius pallipes","Pacifastacus leniusculus","Faxonius limosus"]
N_PCA=10

def haversine_km(lon,lat):
    lon=np.radians(lon); lat=np.radians(lat)
    dlon=lon[:,None]-lon[None,:]; dlat=lat[:,None]-lat[None,:]
    a=np.sin(dlat/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    return 2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))

def main():
    print("loading network out_dist (ids only + out_dist)...")
    net=read_dataframe(GPKG,columns=["stream","out_dist","strahler"],layer="merged",read_geometry=False)
    net["subc_id"]=net["stream"].astype("int64")
    netpos=net.set_index("subc_id")["out_dist"]

    cm=pd.read_csv(MASTER,encoding="latin-1",low_memory=False)
    cm=cm[cm["Accuracy"].astype(str).str.strip().str.lower().eq("high")]
    env_cols=[c for c in cm.columns if c.startswith(("l_","u_"))]
    cm=cm.dropna(subset=["long_snap","lat_snap","subc_id","basin_id"])
    cm=cm[(cm["long_snap"].abs()>0.01)]              # drop null-coord rows
    cm=cm.drop_duplicates("subc_id")

    for sp in SPECIES:
        s=cm[cm["Crayfish_scientific_name"]==sp].copy()
        s["subc_id"]=s["subc_id"].astype("int64")
        s=s[s["subc_id"].isin(netpos.index)]
        s=s[s[env_cols].notna().all(axis=1)]
        s["out_dist"]=s["subc_id"].map(netpos)
        s=s.dropna(subset=["out_dist"]).reset_index(drop=True)
        print(f"\n===== {sp} : N={len(s)} in network =====")
        if len(s)<120: 
            print("  too few; skip"); continue

        er=PCA(n_components=N_PCA,random_state=0).fit_transform(StandardScaler().fit_transform(s[env_cols].to_numpy(float)))
        D_env=squareform(pdist(er))
        D_geo=haversine_km(s.long_snap.to_numpy(),s.lat_snap.to_numpy())
        od=s["out_dist"].to_numpy(float)/1000.0
        D_net=np.abs(od[:,None]-od[None,:])
        iu=np.triu_indices(len(s),k=1); gp,ep,npd=D_geo[iu],D_env[iu],D_net[iu]

        print(f"  {'geo≤km':>7} {'n':>6} {'env|netclose':>13} {'env|netfar':>11} {'ratio':>6} {'p':>8}")
        for thr in [50,25,15,10,7]:
            n=gp<=thr
            if n.sum()<40: continue
            med=np.median(npd[n]); a=ep[n&(npd<=med)]; b=ep[n&(npd>med)]
            if len(a)>10 and len(b)>10 and np.median(a)>0:
                _,p=mannwhitneyu(b,a,alternative="greater")
                print(f"  {thr:7.0f} {n.sum():6d} {np.median(a):13.2f} {np.median(b):11.2f} "
                      f"{np.median(b)/np.median(a):6.2f} {p:8.1e}")
    print("\nReal along-network gradient: rising ratio toward small geo = network distance drives")
    print("env-divergence beyond geography. Compare which species show it (expect variation).")

if __name__ == "__main__":
    main()
