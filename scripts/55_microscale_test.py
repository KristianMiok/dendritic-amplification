"""Is the sub-5km network effect REAL or another artifact? Skeptical checks before chasing it.

For torrentium AND a null species (pallipes), at fine geo scales:
  (A) partial corr(env, net | geo) WITHIN each fine bin -- controls geo even at small scale
  (B) drop near-duplicate pairs (geo < 0.5 km) that could trivially inflate correlation
  (C) permutation null: shuffle env, recompute -- is observed corr beyond chance?
  (D) does the effect appear in the NULL species too? (if yes -> generic artifact, not amplification)
"""
from __future__ import annotations
import os, time
import numpy as np
import pandas as pd
import networkx as nx
from pyogrio import read_dataframe
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, rankdata

MASTER=os.path.expanduser("~/Desktop/Papers/sdm-robustness/data/raw/combined_data_true_master.csv")
N_PCA=10
CASES=[("Austropotamobius torrentium","data/raw/h90m/order_vect_segment_h20v04.gpkg"),
       ("Austropotamobius pallipes","data/raw/h90m/order_vect_segment_h18v04.gpkg")]

def haversine_km(lon,lat):
    lon=np.radians(lon); lat=np.radians(lat)
    dlon=lon[:,None]-lon[None,:]; dlat=lat[:,None]-lat[None,:]
    a=np.sin(dlat/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    return 2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))

def partial_spear(x,y,z):
    rx,ry,rz=rankdata(x),rankdata(y),rankdata(z)
    def resid(a,b):
        b=np.c_[np.ones_like(b),b]; c,_,_,_=np.linalg.lstsq(b,a,rcond=None); return a-b@c
    return np.corrcoef(resid(rx,rz),resid(ry,rz))[0,1]

def build(sp,gpkg):
    cm=pd.read_csv(MASTER,encoding="latin-1",low_memory=False)
    cm=cm[cm["Accuracy"].astype(str).str.strip().str.lower().eq("high")]
    env_cols=[c for c in cm.columns if c.startswith(("l_","u_"))]
    cm=cm.dropna(subset=["long_snap","lat_snap","subc_id","basin_id"])
    cm=cm[cm["long_snap"].abs()>0.01].drop_duplicates("subc_id")
    s=cm[cm["Crayfish_scientific_name"]==sp].copy()
    s["subc_id"]=s["subc_id"].astype("int64"); s=s[s[env_cols].notna().all(axis=1)].reset_index(drop=True)
    pad=0.3; bbox=(s.long_snap.min()-pad,s.lat_snap.min()-pad,s.long_snap.max()+pad,s.lat_snap.max()+pad)
    net=read_dataframe(gpkg,columns=["stream","next_stream","length"],layer="merged",bbox=bbox,read_geometry=False)
    net["stream"]=net["stream"].astype("int64"); net["next_stream"]=net["next_stream"].astype("int64")
    G=nx.Graph(); v=net[net["next_stream"]!=0]
    G.add_weighted_edges_from(zip(v["stream"].to_numpy(),v["next_stream"].to_numpy(),v["length"].to_numpy()/1000.0))
    s=s[s["subc_id"].isin(G)].reset_index(drop=True)
    tg=list(s["subc_id"]); ts=set(tg); ix={n:i for i,n in enumerate(tg)}
    D=np.full((len(tg),len(tg)),np.nan)
    for src in tg:
        L=nx.single_source_dijkstra_path_length(G,src,weight="weight"); i=ix[src]
        for dst in ts&L.keys(): D[i,ix[dst]]=L[dst]
    er=PCA(n_components=N_PCA,random_state=0).fit_transform(StandardScaler().fit_transform(s[env_cols].to_numpy(float)))
    D_env=squareform(pdist(er)); D_geo=haversine_km(s.long_snap.to_numpy(),s.lat_snap.to_numpy())
    iu=np.triu_indices(len(s),k=1)
    return D_geo[iu],D_env[iu],D[iu],len(s)

def main():
    for sp,gpkg in CASES:
        print(f"\n########## {sp} ##########")
        gp,ep,npd,N=build(sp,gpkg); ok=np.isfinite(npd)
        print(f"N={N} pts")
        for lo,hi in [(0,2),(2,5),(0,5),(5,10)]:
            b=ok&(gp>=lo)&(gp<hi)
            # (B) drop near-duplicate geo pairs
            b2=b&(gp>=0.5)
            if b2.sum()<30: 
                print(f"  geo {lo}-{hi}km: too few"); continue
            r=spearmanr(npd[b2],ep[b2]).correlation
            rp=partial_spear(npd[b2],ep[b2],gp[b2])
            # (C) permutation null on the partial-style relationship
            rng=np.random.default_rng(0); perm=[]
            e=ep[b2].copy()
            for _ in range(500):
                rng.shuffle(e); perm.append(spearmanr(npd[b2],e).correlation)
            perm=np.array(perm); pval=(np.sum(perm>=r)+1)/(len(perm)+1)
            print(f"  geo {lo}-{hi}km (n={b2.sum():5d}): corr(env,net)={r:+.3f} | "
                  f"partial|geo={rp:+.3f} | perm_p={pval:.3f}")
        print("  ^ torrentium should show POSITIVE partial|geo at <5km if the micro-effect is real;")
        print("    pallipes (null) should NOT -- if it does, the effect is a generic artifact.")

if __name__ == "__main__":
    main()
