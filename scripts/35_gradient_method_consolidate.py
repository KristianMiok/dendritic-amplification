"""Consolidate the Delta-env(d) gradient as a METHOD on the crayfish lineage data.

Three robustness checks before investing in real along-stream distance:
  (A) per-lineage-pair decomposition  -> is the effect carried by one branch (or by the
      between-species bihariensis contrast) rather than within-torrentium network branches?
  (B) geo-control across multiple narrow bands -> does the branch env-gap hold throughout?
  (C) sensitivity to N_PCA and to env-distance metric -> not an artifact of arbitrary choices.
"""
from __future__ import annotations
import os, itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu

XLSX = os.path.expanduser("~/Desktop/Papers/crayfish-niche-shift/data/raw/master_lineage.xlsx")
MIN_PER_GROUP = 20

def haversine_km(lon, lat):
    lon=np.radians(lon); lat=np.radians(lat)
    dlon=lon[:,None]-lon[None,:]; dlat=lat[:,None]-lat[None,:]
    a=np.sin(dlat/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    return 2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))

def load(n_pca):
    df=pd.read_excel(XLSX)
    env_cols=[c for c in df.columns if c.startswith(("l_","u_"))]
    df=df.dropna(subset=["long_snap","lat_snap","LABEL","subc_id"]).drop_duplicates("subc_id").copy()
    keep=df["LABEL"].value_counts(); df=df[df["LABEL"].isin(keep[keep>=MIN_PER_GROUP].index)].reset_index(drop=True)
    envm=df[env_cols].to_numpy(float); ok=np.isfinite(envm).all(axis=1)
    df,envm=df[ok].reset_index(drop=True),envm[ok]
    env_red=PCA(n_components=n_pca,random_state=0).fit_transform(StandardScaler().fit_transform(envm))
    return df, env_red

def pairs(df, env_red, metric="euclidean"):
    D_env=squareform(pdist(env_red, metric=metric))
    D_geo=haversine_km(df["long_snap"].to_numpy(), df["lat_snap"].to_numpy())
    lab=df["LABEL"].to_numpy()
    iu=np.triu_indices(len(df),k=1)
    return D_geo[iu], D_env[iu], lab[iu[0]], lab[iu[1]]

def main():
    df, env_red = load(10)
    labs = sorted(df["LABEL"].unique())
    print(f"N={len(df)} | groups={df['LABEL'].value_counts().to_dict()}\n")

    gp, ep, li, lj = pairs(df, env_red)
    near = gp<=15

    print("=== (A) per-group-pair env-gap among geographically-near (<=15km) pairs ===")
    same = (li==lj)
    base_same = np.median(ep[near&same])
    print(f"  SAME-branch near pairs: n={ (near&same).sum() } median env={base_same:.2f}")
    for A,B in itertools.combinations(labs,2):
        m = near & (((li==A)&(lj==B)) | ((li==B)&(lj==A)))
        if m.sum()>=10:
            print(f"  {A:22s} vs {B:22s} n={m.sum():5d} median env={np.median(ep[m]):6.2f} "
                  f"ratio_vs_same={np.median(ep[m])/base_same:5.2f}")
    # torrentium-only (exclude bihariensis) to isolate WITHIN-species network effect
    tor = [l for l in labs if "torrentium" in l.lower()]
    print(f"\n  --- torrentium-only (within-species branches): {tor} ---")
    tmask = np.isin(li,tor) & np.isin(lj,tor)
    for thr in [50,25,15,10]:
        n=gp<=thr; s=n&tmask&(li==lj); d=n&tmask&(li!=lj)
        if s.sum()>10 and d.sum()>10 and np.median(ep[s])>0:
            print(f"    geo<= {thr:3d}km | same n={s.sum():5d} med={np.median(ep[s]):5.2f} | "
                  f"diff n={d.sum():4d} med={np.median(ep[d]):5.2f} | ratio={np.median(ep[d])/np.median(ep[s]):.2f}")

    print("\n=== (B) geo-control across narrow non-overlapping bands (torrentium-only) ===")
    for lo,hi in [(2,7),(7,12),(12,20),(20,35)]:
        b=(gp>=lo)&(gp<hi)&tmask; s=b&(li==lj); d=b&(li!=lj)
        if s.sum()>10 and d.sum()>10 and np.median(ep[s])>0:
            _,p=mannwhitneyu(ep[d],ep[s],alternative="greater")
            print(f"    band {lo:2d}-{hi:2d}km | same n={s.sum():5d} med={np.median(ep[s]):5.2f} | "
                  f"diff n={d.sum():4d} med={np.median(ep[d]):5.2f} | ratio={np.median(ep[d])/np.median(ep[s]):.2f} | p={p:.1e}")

    print("\n=== (C) sensitivity to N_PCA and env metric (torrentium-only, geo<=15km) ===")
    for npca in [5,10,20,30]:
        d2,er=load(npca)
        g2,e2,a2,b2=pairs(d2,er)
        t2=np.isin(a2,tor)&np.isin(b2,tor); n=g2<=15
        s=n&t2&(a2==b2); d=n&t2&(a2!=b2)
        if s.sum()>10 and d.sum()>10 and np.median(e2[s])>0:
            print(f"    N_PCA={npca:2d} | ratio={np.median(e2[d])/np.median(e2[s]):.2f} (n_diff={d.sum()})")
    for met in ["euclidean","cityblock","cosine"]:
        g2,e2,a2,b2=pairs(df,env_red,metric=met)
        t2=np.isin(a2,tor)&np.isin(b2,tor); n=g2<=15
        s=n&t2&(a2==b2); d=n&t2&(a2!=b2)
        if s.sum()>10 and d.sum()>10 and np.median(e2[s])>0:
            print(f"    metric={met:9s} | ratio={np.median(e2[d])/np.median(e2[s]):.2f}")

if __name__ == "__main__":
    main()
