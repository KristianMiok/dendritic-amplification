"""Turn the ad-hoc lineage gradient into the actual METHOD test for an Ecography methods paper.

Does Information Imbalance (the proposed tool) detect the same 'network organizes niche
beyond geography' signal that the ad-hoc pairwise gradient found?

We compare, on deduplicated records:
  II(geo -> env)             baseline
  II(lineage_proxy -> env)   network-as-lineage proxy, with zero-distance smearing
Lineage proxy = geographic centroid offset per lineage (continuous, avoids one-hot ties):
each record represented by (its lineage centroid lon/lat) -> a continuous stand-in for
'which network branch', smeared to remove identical points.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

XLSX = os.path.expanduser("~/Desktop/Papers/crayfish-niche-shift/data/raw/master_lineage.xlsx")
N_PCA = 10; MIN_PER_GROUP = 20

def ii_smeared(X, cols_a, cols_b, k=1, jitter=1e-6, seed=0):
    from dadapy import MetricComparisons
    rng = np.random.default_rng(seed)
    Z = StandardScaler().fit_transform(X.astype(float))
    Z = Z + jitter*rng.standard_normal(Z.shape)          # smear to kill zero-distances
    mc = MetricComparisons(coordinates=Z, maxk=Z.shape[0]-1)
    a,b = mc.return_inf_imb_two_selected_coords(list(cols_a), list(cols_b), k=k)
    return float(a), float(b)

def main():
    df = pd.read_excel(XLSX)
    env_cols=[c for c in df.columns if c.startswith(("l_","u_"))]
    df=df.dropna(subset=["long_snap","lat_snap","LABEL","subc_id"]).drop_duplicates("subc_id").copy()
    keep=df["LABEL"].value_counts(); df=df[df["LABEL"].isin(keep[keep>=MIN_PER_GROUP].index)].copy()
    envm=df[env_cols].to_numpy(float); ok=np.isfinite(envm).all(axis=1)
    df,envm=df[ok].reset_index(drop=True),envm[ok]
    print(f"N={len(df)} | lineages={df['LABEL'].value_counts().to_dict()}")

    env_red=PCA(n_components=N_PCA,random_state=0).fit_transform(StandardScaler().fit_transform(envm))
    geo=df[["long_snap","lat_snap"]].to_numpy(float)

    # continuous lineage proxy: each record -> its lineage's geographic centroid
    cent=df.groupby("LABEL")[["long_snap","lat_snap"]].transform("mean").to_numpy(float)

    Xg=np.c_[geo, env_red]
    dge,_=ii_smeared(Xg,[0,1],list(range(2,2+N_PCA)))
    Xl=np.c_[cent, env_red]
    dle,_=ii_smeared(Xl,[0,1],list(range(2,2+N_PCA)))

    print("\n== Information Imbalance as the METHOD (smeared, deduplicated) ==")
    print(f"  Delta(geo             -> env) = {dge:.3f}")
    print(f"  Delta(lineage-centroid-> env) = {dle:.3f}")
    print(f"  ratio geo/lineage             = {dge/dle:.2f}")
    print("\nIf lineage-proxy Delta < geo Delta -> the METHOD reproduces the gradient finding;")
    print("II is a valid detector, and real along-stream distance should sharpen it further.")

if __name__ == "__main__":
    main()
