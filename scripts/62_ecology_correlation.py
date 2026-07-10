"""Does the STRENGTH of drainage-network niche organization (b_basin) track species ecology?
Key hypothesis: headwater specialists (low median strahler) show stronger b_basin
-> would connect to m19 (calibration fails at headwaters) and support an ecological (not
purely geometric) interpretation.

Per species: b_basin (env~geo+diff_basin coefficient) vs ecological traits:
  - median strahler order (LOW = headwater specialist)
  - geo span (range), n_basins (occupancy breadth)
Then Spearman correlations across species, plus partial(b_basin, strahler | n_basins)
to separate ecology from a basin-count geometric artifact.
EXPLORATORY: ~20-28 species, phylogenetically non-independent -> hypothesis-generating.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, rankdata

MASTER=os.path.expanduser("~/Desktop/Papers/sdm-robustness/data/raw/combined_data_true_master.csv")
N_PCA=10; NMAX=1600; MIN_N=150; MIN_BASINS=3; SPREAD_KM=8000

def haversine_km(lon,lat):
    lon=np.radians(lon); lat=np.radians(lat)
    dlon=lon[:,None]-lon[None,:]; dlat=lat[:,None]-lat[None,:]
    a=np.sin(dlat/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    return 2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))

def z(v): v=np.asarray(v,float); return (v-v.mean())/(v.std()+1e-12)

def b_basin(s,env_cols):
    er=PCA(n_components=N_PCA,random_state=0).fit_transform(StandardScaler().fit_transform(s[env_cols].to_numpy(float)))
    D_env=squareform(pdist(er)); D_geo=haversine_km(s.long_snap.to_numpy(),s.lat_snap.to_numpy())
    bas=s["basin_id"].to_numpy(); n=len(s); iu=np.triu_indices(n,k=1)
    gp=D_geo[iu]; diffb=(bas[iu[0]]!=bas[iu[1]]).astype(float); m=gp<=50
    if diffb[m].sum()<10: return np.nan
    y=z(D_env[iu][m]); X=np.c_[np.ones(m.sum()),z(gp[m]),diffb[m]]
    coef,_,_,_=np.linalg.lstsq(X,y,rcond=None); return coef[2]

def partial_spear(x,y,zz):
    rx,ry,rz=rankdata(x),rankdata(y),rankdata(zz)
    def resid(a,b):
        b=np.c_[np.ones_like(b),b]; c,_,_,_=np.linalg.lstsq(b,a,rcond=None); return a-b@c
    return np.corrcoef(resid(rx,rz),resid(ry,rz))[0,1]

def main():
    cm=pd.read_csv(MASTER,encoding="latin-1",low_memory=False)
    cm=cm[cm["Accuracy"].astype(str).str.strip().str.lower().eq("high")]
    env_cols=[c for c in cm.columns if c.startswith(("l_","u_"))]
    cm=cm.dropna(subset=["long_snap","lat_snap","subc_id","basin_id","strahler"])
    cm=cm[cm["long_snap"].abs()>0.01].drop_duplicates("subc_id")

    rows=[]
    for sp,g in cm.groupby("Crayfish_scientific_name"):
        g=g[g[env_cols].notna().all(axis=1)]
        n_all=g["subc_id"].nunique(); nb=g["basin_id"].nunique()
        if n_all<MIN_N or nb<MIN_BASINS: continue
        span=(g.long_snap.max()-g.long_snap.min())*111
        med_str=float(np.median(g["strahler"].to_numpy(float)))
        s=g.sample(NMAX,random_state=0) if len(g)>NMAX else g
        b=b_basin(s.reset_index(drop=True),env_cols)
        if np.isnan(b): continue
        rows.append(dict(species=sp,b_basin=b,med_strahler=med_str,span_km=span,
                         n_basins=nb,N=n_all,spread=span>SPREAD_KM))
    df=pd.DataFrame(rows).sort_values("b_basin",ascending=False)
    print(f"{len(df)} species\n")
    print(df.to_string(index=False,
          formatters={"b_basin":"{:+.3f}".format,"med_strahler":"{:.1f}".format,
                      "span_km":"{:.0f}".format}))

    print("\n=== correlations of b_basin with ecology (Spearman) ===")
    for sub,name in [(df,"ALL species"),(df[~df.spread],"NATIVE-COMPACT only (span<8000)")]:
        if len(sub)<6: continue
        print(f"\n-- {name} (n={len(sub)}) --")
        for trait in ["med_strahler","span_km","n_basins"]:
            r,p=spearmanr(sub["b_basin"],sub[trait])
            print(f"  b_basin vs {trait:13s}: rho={r:+.3f}  p={p:.3f}")
        # partial: does headwater affinity predict b_basin beyond basin count?
        rp=partial_spear(sub["b_basin"].to_numpy(),sub["med_strahler"].to_numpy(),sub["n_basins"].to_numpy())
        print(f"  b_basin vs med_strahler | n_basins (partial): {rp:+.3f}")
    print("\nHypothesis: NEGATIVE b_basin~strahler (headwater specialists=low strahler=stronger effect)")
    print("would link to m19 (headwater calibration failure) and support ecological interpretation.")
    print("If no ecology correlation -> effect is landscape geometry, uniform across species (still valid,")
    print("but descriptive). Exploratory: small N, phylogenetic non-independence not modelled.")

if __name__ == "__main__":
    main()
