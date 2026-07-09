"""Direct dendritic-amplification test on existing master_lineage.xlsx (no downloads).

Question: for pairs of records that are GEOGRAPHICALLY CLOSE, does belonging to a
DIFFERENT lineage (different network branch) mean they are ENVIRONMENTALLY FAR?
If yes -> the network separates what geography places nearby = the amplification signal.

Compares, among geographically-near pairs:
  env distance | same lineage   vs   env distance | different lineage
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

XLSX = os.path.expanduser("~/Desktop/Papers/crayfish-niche-shift/data/raw/master_lineage.xlsx")
N_PCA = 10
GEO_NEAR_KM = 25.0          # "geographically close" threshold
MIN_PER_GROUP = 20

def haversine_km(lon, lat):
    lon = np.radians(lon); lat = np.radians(lat)
    dlon = lon[:,None]-lon[None,:]; dlat = lat[:,None]-lat[None,:]
    a = np.sin(dlat/2)**2 + np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    return 2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))

def main():
    df = pd.read_excel(XLSX)
    env_cols = [c for c in df.columns if c.startswith(("l_","u_"))]
    df = df.dropna(subset=["long_snap","lat_snap","LABEL"]).copy()
    keep = df["LABEL"].value_counts(); keep = keep[keep>=MIN_PER_GROUP].index
    df = df[df["LABEL"].isin(keep)].copy()
    envm = df[env_cols].to_numpy(float)
    ok = np.isfinite(envm).all(axis=1)
    df, envm = df[ok].reset_index(drop=True), envm[ok]
    print(f"N={len(df)} | lineages={df['LABEL'].value_counts().to_dict()}")

    env_red = PCA(n_components=N_PCA, random_state=0).fit_transform(StandardScaler().fit_transform(envm))
    D_env = squareform(pdist(env_red))                      # env distance (standardized PCA space)
    D_geo = haversine_km(df["long_snap"].to_numpy(), df["lat_snap"].to_numpy())
    lab = df["LABEL"].to_numpy()

    iu = np.triu_indices(len(df), k=1)
    geo_pair = D_geo[iu]; env_pair = D_env[iu]
    same = (lab[iu[0]] == lab[iu[1]])
    near = geo_pair <= GEO_NEAR_KM

    print(f"\ngeographically-near pairs (<= {GEO_NEAR_KM} km): {near.sum()}")
    for mask,name in [(near & same,"NEAR + SAME lineage"), (near & ~same,"NEAR + DIFF lineage")]:
        if mask.sum()>0:
            print(f"  {name:22s} n={mask.sum():6d} | median env-dist={np.median(env_pair[mask]):.3f} "
                  f"| mean={env_pair[mask].mean():.3f}")

    a = env_pair[near & same]; b = env_pair[near & ~same]
    if len(a)>10 and len(b)>10:
        from scipy.stats import mannwhitneyu
        u,p = mannwhitneyu(b,a,alternative="greater")
        ratio = np.median(b)/np.median(a) if np.median(a)>0 else float("inf")
        print(f"\n  env-gap ratio (diff/same, medians) = {ratio:.2f}")
        print(f"  Mann-Whitney (diff > same env dist): p = {p:.2e}")
        print("\n  SIGNAL:", "YES - different network branch => env-farther even when geo-near"
              if p<0.05 and ratio>1.2 else "weak/none - lineage doesn't add env separation over geography")
    else:
        print("  too few pairs in one bin.")

if __name__ == "__main__":
    main()
