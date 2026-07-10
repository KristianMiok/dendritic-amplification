"""MMRR (Wang 2013) as the DECISIVE test, set up to be able to FAIL.
env-dist ~ geo-dist + network-dist, Mantel permutation (permute response node labels,
predictors fixed -> respects pair dependence). Run on all pairs (primary confirmatory test).
Reports b_net WITH geo in the model. If b_net is not significant -> amplification NOT confirmed.
Also reports the <5km-restricted b_net as EXPLORATORY context (not confirmatory).
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

MASTER=os.path.expanduser("~/Desktop/Papers/sdm-robustness/data/raw/combined_data_true_master.csv")
N_PCA=10; NPERM=2000
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

def z(v): 
    v=np.asarray(v,float); return (v-v.mean())/(v.std()+1e-12)

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

def mmrr(D_env,D_geo,D_net,pair_mask,nperm,rng):
    n=D_env.shape[0]; iu=np.triu_indices(n,k=1)
    m=pair_mask
    y=z(D_env[iu][m]); Xg=z(D_geo[iu][m]); Xn=z(D_net[iu][m])
    X=np.c_[np.ones_like(y),Xg,Xn]
    coef,_,_,_=np.linalg.lstsq(X,y,rcond=None)
    yhat=X@coef; r2=1-np.sum((y-yhat)**2)/np.sum((y-y.mean())**2)
    b_geo,b_net=coef[1],coef[2]
    cN=cG=0
    for _ in range(nperm):
        p=rng.permutation(n)
        yp=z(D_env[np.ix_(p,p)][iu][m])
        cp,_,_,_=np.linalg.lstsq(X,yp,rcond=None)
        if abs(cp[2])>=abs(b_net): cN+=1
        if abs(cp[1])>=abs(b_geo): cG+=1
    return b_geo,b_net,r2,(cG+1)/(nperm+1),(cN+1)/(nperm+1)

def main():
    rng=np.random.default_rng(0)
    print("MMRR: env-dist ~ geo-dist + network-dist | Mantel permutation (respects dependence)\n")
    print(f"{'species':28s} {'type':6s} {'N':>4} | {'b_geo':>7} {'b_net':>7} {'R2':>5} {'p_geo':>6} {'p_net':>6}")
    print("-"*82)
    for sp,gpkg,typ in CASES:
        r=build(sp,gpkg)
        if r is None: print(f"{sp:28s} {typ:6s} too few"); continue
        De,Dg,Dn=r; n=len(Dg); iu=np.triu_indices(n,k=1)
        allmask=np.isfinite(Dn[iu])
        bg,bn,r2,pg,pn=mmrr(De,Dg,Dn,allmask,NPERM,rng)
        flag="***" if (bn>0.05 and pn<0.05) else ("" if bn<0.03 else " ?")
        print(f"{sp:28s} {typ:6s} {n:4d} | {bg:+7.3f} {bn:+7.3f} {r2:5.2f} {pg:6.3f} {pn:6.3f}  {flag} [ALL pairs]")
        # exploratory: <5km only (NOT a clean matrix test; context only)
        fine=allmask&(Dg[iu]>=0.5)&(Dg[iu]<5)
        if fine.sum()>50:
            y=z(De[iu][fine]); X=np.c_[np.ones(fine.sum()),z(Dg[iu][fine]),z(Dn[iu][fine])]
            c,_,_,_=np.linalg.lstsq(X,y,rcond=None)
            print(f"{'':28s} {'':6s} {'':>4} | {c[1]:+7.3f} {c[2]:+7.3f} {'':>5} {'':>6} {'':>6}   [<5km, exploratory]")
    print("\nPRIMARY test = [ALL pairs]. b_net *** with geo in model -> network adds beyond geography.")
    print("Given partial corr was ~0 globally, b_net is EXPECTED to be weak on all pairs -> honest null.")
    print("The [<5km] row is exploratory context only (subset, not a clean Mantel matrix).")

if __name__ == "__main__":
    main()
