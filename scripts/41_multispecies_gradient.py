"""Delta-env(d) gradient across multiple NATIVE crayfish species (basin_id = network branch).
Tests whether the amplification gradient depends on species' geographic range.
Restricted to native, single-continent species to avoid invasion-spread confounding.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

MASTER = os.path.expanduser("~/Desktop/Papers/sdm-robustness/data/raw/combined_data_true_master.csv")
SPECIES = ["Austropotamobius torrentium","Austropotamobius pallipes",
           "Astacus astacus","Pontastacus leptodactylus"]
N_PCA=10; NEAR_KM=15

def rd():
    for e in ("utf-8","latin-1","cp1252"):
        try: return pd.read_csv(MASTER, encoding=e, low_memory=False)
        except UnicodeDecodeError: pass
    raise RuntimeError("enc")

def haversine_km(lon,lat):
    lon=np.radians(lon); lat=np.radians(lat)
    dlon=lon[:,None]-lon[None,:]; dlat=lat[:,None]-lat[None,:]
    a=np.sin(dlat/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    return 2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))

def main():
    df=rd()
    env_cols=[c for c in df.columns if c.startswith(("l_","u_"))]
    hi=df[df["Accuracy"].astype(str).str.strip().str.lower().eq("high")].copy()
    print(f"{'species':32s} {'N':>6} {'basins':>7} | env-gap ratio by geo threshold (diff-basin vs same-basin)")
    print(f"{'':32s} {'':>6} {'':>7} |   50km   25km   15km   10km    5km")
    for sp in SPECIES:
        s=hi[hi["Crayfish_scientific_name"]==sp].dropna(subset=["long_snap","lat_snap","subc_id","basin_id"]).drop_duplicates("subc_id")
        em=s[env_cols].to_numpy(float); ok=np.isfinite(em).all(axis=1)
        s,em=s[ok].reset_index(drop=True),em[ok]
        if len(s)<150: 
            print(f"{sp:32s} {len(s):6d}  too few"); continue
        er=PCA(n_components=N_PCA,random_state=0).fit_transform(StandardScaler().fit_transform(em))
        D_env=squareform(pdist(er)); D_geo=haversine_km(s.long_snap.to_numpy(),s.lat_snap.to_numpy())
        bas=s.basin_id.to_numpy(); iu=np.triu_indices(len(s),k=1)
        gp,ep=D_geo[iu],D_env[iu]; same=(bas[iu[0]]==bas[iu[1]])
        row=[]
        for thr in [50,25,15,10,5]:
            n=gp<=thr; a=ep[n&same]; b=ep[n&~same]
            if len(a)>10 and len(b)>10 and np.median(a)>0:
                row.append(f"{np.median(b)/np.median(a):6.2f}")
            else:
                row.append("   -- ")
        print(f"{sp:32s} {len(s):6d} {s.basin_id.nunique():7d} | {' '.join(row)}")
    print("\nRising ratio toward smaller km = amplification present. Compare across species'")
    print("ranges: does a wide native (astacus) show weaker/stronger amplification than a narrow one (torrentium/pallipes)?")

if __name__ == "__main__":
    main()
