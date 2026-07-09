"""Stress-test the signal: does the env-gap between network branches GROW as geography
tightens? And is it robust to controlling for residual geographic distance?
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
    df=df.dropna(subset=["long_snap","lat_snap","LABEL"]).copy()
    keep=df["LABEL"].value_counts(); df=df[df["LABEL"].isin(keep[keep>=MIN_PER_GROUP].index)].copy()
    envm=df[env_cols].to_numpy(float); ok=np.isfinite(envm).all(axis=1)
    df,envm=df[ok].reset_index(drop=True),envm[ok]
    env_red=PCA(n_components=N_PCA,random_state=0).fit_transform(StandardScaler().fit_transform(envm))
    D_env=squareform(pdist(env_red)); 
    D_geo=haversine_km(df["long_snap"].to_numpy(),df["lat_snap"].to_numpy())
    lab=df["LABEL"].to_numpy()
    iu=np.triu_indices(len(df),k=1)
    gp,ep=D_geo[iu],D_env[iu]; same=(lab[iu[0]]==lab[iu[1]])

    print("=== Does the branch env-gap GROW as geography tightens? ===")
    print(f"{'geo≤km':>8} {'n_same':>8} {'n_diff':>8} {'env same':>9} {'env diff':>9} {'ratio':>7}")
    for thr in [50,25,15,10,5,2]:
        near=gp<=thr
        a=ep[near&same]; b=ep[near&~same]
        if len(a)>10 and len(b)>10:
            r=np.median(b)/np.median(a)
            print(f"{thr:8.0f} {len(a):8d} {len(b):8d} {np.median(a):9.2f} {np.median(b):9.2f} {r:7.2f}")
        else:
            print(f"{thr:8.0f} {len(a):8d} {len(b):8d}  (too few)")

    print("\n=== Control for residual geo distance: match same/diff pairs in the SAME geo band ===")
    # within a tight geo band (5-15 km), are diff-lineage pairs still env-farther?
    band=(gp>=5)&(gp<=15)
    a=ep[band&same]; b=ep[band&~same]
    if len(a)>10 and len(b)>10:
        _,p=mannwhitneyu(b,a,alternative="greater")
        print(f"geo band 5-15km | same n={len(a)} med={np.median(a):.2f} | diff n={len(b)} med={np.median(b):.2f} "
              f"| ratio={np.median(b)/np.median(a):.2f} | p={p:.1e}")
        print("Interpretation: if ratio still >1.2 within a fixed geo band, effect is NOT just residual geography.")

if __name__ == "__main__":
    main()
