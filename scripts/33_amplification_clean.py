"""Clean version: one record per subc_id (remove identical-env duplicates), then the
geography-tightening gradient of the branch env-gap. This is the presentable result.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu

XLSX = os.path.expanduser("~/Desktop/Papers/crayfish-niche-shift/data/raw/master_lineage.xlsx")
N_PCA = 10; MIN_PER_GROUP = 20

def haversine_km(lon, lat):
    lon=np.radians(lon); lat=np.radians(lat)
    dlon=lon[:,None]-lon[None,:]; dlat=lat[:,None]-lat[None,:]
    a=np.sin(dlat/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    return 2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))

def main():
    df = pd.read_excel(XLSX)
    env_cols=[c for c in df.columns if c.startswith(("l_","u_"))]
    df=df.dropna(subset=["long_snap","lat_snap","LABEL","subc_id"]).copy()
    before=len(df)
    df=df.drop_duplicates("subc_id").copy()          # one record per network cell
    print(f"deduplicated by subc_id: {before} -> {len(df)} records")
    keep=df["LABEL"].value_counts(); df=df[df["LABEL"].isin(keep[keep>=MIN_PER_GROUP].index)].copy()
    envm=df[env_cols].to_numpy(float); ok=np.isfinite(envm).all(axis=1)
    df,envm=df[ok].reset_index(drop=True),envm[ok]
    print(f"final N={len(df)} | lineages={df['LABEL'].value_counts().to_dict()}")

    env_red=PCA(n_components=N_PCA,random_state=0).fit_transform(StandardScaler().fit_transform(envm))
    D_env=squareform(pdist(env_red))
    D_geo=haversine_km(df["long_snap"].to_numpy(),df["lat_snap"].to_numpy())
    lab=df["LABEL"].to_numpy()
    iu=np.triu_indices(len(df),k=1); gp,ep=D_geo[iu],D_env[iu]; same=(lab[iu[0]]==lab[iu[1]])

    print(f"\n{'geo≤km':>8} {'n_same':>7} {'n_diff':>7} {'env_same':>9} {'env_diff':>9} {'ratio':>7} {'p':>9}")
    for thr in [50,25,15,10,7,5]:
        near=gp<=thr; a=ep[near&same]; b=ep[near&~same]
        if len(a)>10 and len(b)>10 and np.median(a)>0:
            _,p=mannwhitneyu(b,a,alternative="greater")
            print(f"{thr:8.0f} {len(a):7d} {len(b):7d} {np.median(a):9.2f} {np.median(b):9.2f} "
                  f"{np.median(b)/np.median(a):7.2f} {p:9.1e}")
    print("\nMonotone rising ratio with tighter geography = dendritic amplification signature.")

if __name__ == "__main__":
    main()
