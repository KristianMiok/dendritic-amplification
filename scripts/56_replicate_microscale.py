"""Replication of the <5km network>geography effect across multiple species.
Signal species (had proxy signal): torrentium, leptodactylus.
Null species (control): pallipes, leniusculus.
Test per species at fine scale: partial corr(env,net|geo) + permutation p.
Real & general effect -> signal species POSITIVE at <5km, null species NOT.
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
# (species, tile with enough coverage, expected: signal/null)
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
    if len(s)<120: return None
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
    print(f"{'species':30s} {'type':7s} {'N':>4} | {'<5km partial|geo':>16} {'perm_p':>8} {'5-10km p|geo':>12}")
    print("-"*90)
    for sp,gpkg,typ in CASES:
        r=build(sp,gpkg)
        if r is None: print(f"{sp:30s} {typ:7s}  --  too few"); continue
        gp,ep,npd,N=r; ok=np.isfinite(npd)
        # <5km, drop near-dup
        b=ok&(gp>=0.5)&(gp<5)
        if b.sum()<30:
            print(f"{sp:30s} {typ:7s} {N:4d} | too few <5km pairs"); continue
        rp=partial_spear(npd[b],ep[b],gp[b])
        rng=np.random.default_rng(0); e=ep[b].copy(); perm=[]
        for _ in range(1000):
            rng.shuffle(e); perm.append(partial_spear(npd[b],e,gp[b]))
        perm=np.array(perm); pval=(np.sum(perm>=rp)+1)/(len(perm)+1)
        # 5-10km for contrast
        b2=ok&(gp>=5)&(gp<10)
        rp2=partial_spear(npd[b2],ep[b2],gp[b2]) if b2.sum()>30 else np.nan
        flag = "***" if (rp>0.15 and pval<0.05) else ("" if rp<0.1 else " ?")
        print(f"{sp:30s} {typ:7s} {N:4d} | {rp:+16.3f} {pval:8.3f} {rp2:+12.3f}  {flag}")
    print("\n*** = strong positive fine-scale effect controlling for geography (permutation-significant).")
    print("Expected pattern for a REAL effect: signal species ***; null species blank.")

if __name__ == "__main__":
    main()
