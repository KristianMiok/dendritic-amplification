"""Address both valid critiques from the skeptic:
  (1) pair non-independence -> permute NODES (Mantel-style), not pairs.
  (2) multiple testing -> ONE pre-fixed scale (<5km), applied identically to all species.
Plus the key discriminating control the skeptic ignored: run the SAME procedure on null
species. A real effect = signal species significant under node-permutation, null species not.
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
from scipy.stats import rankdata

MASTER=os.path.expanduser("~/Desktop/Papers/sdm-robustness/data/raw/combined_data_true_master.csv")
N_PCA=10
GEO_MAX=5.0        # ONE pre-fixed scale, no bin shopping
N_PERM=2000
CASES=[
  ("Austropotamobius torrentium","data/raw/h90m/order_vect_segment_h20v04.gpkg","signal"),
  ("Pontastacus leptodactylus","data/raw/h90m/order_vect_segment_h20v04.gpkg","signal"),
  ("Austropotamobius pallipes","data/raw/h90m/order_vect_segment_h18v04.gpkg","null"),
  ("Pacifastacus leniusculus","data/raw/h90m/order_vect_segment_h18v04.gpkg","null"),
]

def haversine_km(lon,lat):
    lon=np.radians(lon); lat=np.radians(lat)
    dlon=lon[:,None]-lon[None,:]; dlat=lat[:,None]-lat[None,:]
    a=np.sin(dlat/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    return 2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))

def partial_spear_stat(De,Dn,Dg,mask):
    """partial Spearman corr(env,net|geo) over masked pairs, returned as scalar."""
    x,y,z=Dn[mask],De[mask],Dg[mask]
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
    if len(s)<120: return None
    tg=list(s["subc_id"]); ts=set(tg); ix={n:i for i,n in enumerate(tg)}
    D=np.full((len(tg),len(tg)),np.nan)
    for src in tg:
        L=nx.single_source_dijkstra_path_length(G,src,weight="weight"); i=ix[src]
        for dst in ts&L.keys(): D[i,ix[dst]]=L[dst]
    er=PCA(n_components=N_PCA,random_state=0).fit_transform(StandardScaler().fit_transform(s[env_cols].to_numpy(float)))
    return squareform(pdist(er)), haversine_km(s.long_snap.to_numpy(),s.lat_snap.to_numpy()), D

def main():
    print(f"ONE fixed scale (<{GEO_MAX}km), NODE-permutation (respects pair dependence), {N_PERM} perms\n")
    print(f"{'species':30s} {'type':7s} {'N':>4} {'obs partial|geo':>16} {'node-perm p':>12}")
    print("-"*76)
    rng=np.random.default_rng(0)
    for sp,gpkg,typ in CASES:
        r=build(sp,gpkg)
        if r is None: print(f"{sp:30s} {typ:7s}  too few"); continue
        De,Dgeo,Dnet=r; n=len(Dgeo)
        iu=np.triu_indices(n,k=1)
        De_p,Dg_p,Dn_p=De[iu],Dgeo[iu],Dnet[iu]
        base_mask=np.isfinite(Dn_p)&(Dg_p>=0.5)&(Dg_p<GEO_MAX)
        if base_mask.sum()<30: print(f"{sp:30s} {typ:7s} {n:4d}  too few <{GEO_MAX}km"); continue
        obs=partial_spear_stat(De_p,Dn_p,Dg_p,base_mask)
        # NODE permutation: shuffle env rows (whole points), rebuild env-dist, recompute
        # (permute the PCA-embedded points, not pair values)
        er_idx=np.arange(n)
        Dfull_e=De  # env distance matrix
        cnt=0
        for _ in range(N_PERM):
            perm=rng.permutation(n)
            # permuted env-distance = De[perm][:,perm]
            Deperm=De[np.ix_(perm,perm)][iu]
            st=partial_spear_stat(Deperm,Dn_p,Dg_p,base_mask)
            if st>=obs: cnt+=1
        pval=(cnt+1)/(N_PERM+1)
        flag="***" if (obs>0.15 and pval<0.05) else ("" if obs<0.1 else " ?")
        print(f"{sp:30s} {typ:7s} {n:4d} {obs:+16.3f} {pval:12.3f}  {flag}")
    print("\nNode-permutation respects that the same point appears in many pairs.")
    print("REAL effect: signal species *** ; null species blank, under IDENTICAL fixed procedure.")

if __name__ == "__main__":
    main()
