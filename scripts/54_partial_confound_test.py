"""DECISIVE confound test (addressing the other chat's valid critique):
Does network distance explain env-divergence BEYOND geographic distance?

Three checks on torrentium true-distance pairs:
  (1) Partial Spearman: corr(env, net | geo)  -- if ~0, the signal is just geography.
  (2) Matched test: within NARROW geo bins, is net still predictive? (removes geo confound)
  (3) Same-basin only: restrict to pairs in the SAME basin (removes the 48%-connected /
      cross-basin contamination the other chat flagged), then re-test net vs env.
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

GPKG="data/raw/h90m/order_vect_segment_h20v04.gpkg"
MASTER=os.path.expanduser("~/Desktop/Papers/sdm-robustness/data/raw/combined_data_true_master.csv")
SP="Austropotamobius torrentium"; N_PCA=10

def haversine_km(lon,lat):
    lon=np.radians(lon); lat=np.radians(lat)
    dlon=lon[:,None]-lon[None,:]; dlat=lat[:,None]-lat[None,:]
    a=np.sin(dlat/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    return 2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))

def partial_spearman(x,y,z):
    """Spearman corr(x,y | z) via rank residuals."""
    rx,ry,rz=rankdata(x),rankdata(y),rankdata(z)
    def resid(a,b):
        b=np.c_[np.ones_like(b),b]; coef,_,_,_=np.linalg.lstsq(b,a,rcond=None); return a-b@coef
    return np.corrcoef(resid(rx,rz),resid(ry,rz))[0,1]

def main():
    t0=time.time()
    cm=pd.read_csv(MASTER,encoding="latin-1",low_memory=False)
    cm=cm[cm["Accuracy"].astype(str).str.strip().str.lower().eq("high")]
    env_cols=[c for c in cm.columns if c.startswith(("l_","u_"))]
    cm=cm.dropna(subset=["long_snap","lat_snap","subc_id","basin_id"])
    cm=cm[cm["long_snap"].abs()>0.01].drop_duplicates("subc_id")
    s=cm[cm["Crayfish_scientific_name"]==SP].copy()
    s["subc_id"]=s["subc_id"].astype("int64")
    s=s[s[env_cols].notna().all(axis=1)].reset_index(drop=True)

    pad=0.3; bbox=(s.long_snap.min()-pad,s.lat_snap.min()-pad,s.long_snap.max()+pad,s.lat_snap.max()+pad)
    net=read_dataframe(GPKG,columns=["stream","next_stream","length"],layer="merged",bbox=bbox,read_geometry=False)
    net["stream"]=net["stream"].astype("int64"); net["next_stream"]=net["next_stream"].astype("int64")
    G=nx.Graph(); v=net[net["next_stream"]!=0]
    G.add_weighted_edges_from(zip(v["stream"].to_numpy(),v["next_stream"].to_numpy(),v["length"].to_numpy()/1000.0))
    s=s[s["subc_id"].isin(G)].reset_index(drop=True)
    print(f"{SP}: {len(s)} pts in graph")

    targets=list(s["subc_id"]); tset=set(targets); idx={n:i for i,n in enumerate(targets)}
    D=np.full((len(targets),len(targets)),np.nan)
    for src in targets:
        L=nx.single_source_dijkstra_path_length(G,src,weight="weight"); i=idx[src]
        for dst in tset & L.keys(): D[i,idx[dst]]=L[dst]

    er=PCA(n_components=N_PCA,random_state=0).fit_transform(StandardScaler().fit_transform(s[env_cols].to_numpy(float)))
    D_env=squareform(pdist(er)); D_geo=haversine_km(s.long_snap.to_numpy(),s.lat_snap.to_numpy())
    bas=s["basin_id"].to_numpy()
    iu=np.triu_indices(len(s),k=1)
    gp,ep,npd=D_geo[iu],D_env[iu],D[iu]; sameb=(bas[iu[0]]==bas[iu[1]])
    ok=np.isfinite(npd)

    print("\n=== (1) Partial Spearman: env vs network | geography ===")
    m=ok
    r_env_net=spearmanr(npd[m],ep[m]).correlation
    r_env_geo=spearmanr(gp[m],ep[m]).correlation
    r_partial=partial_spearman(npd[m],ep[m],gp[m])
    print(f"  corr(env,net)       = {r_env_net:+.3f}")
    print(f"  corr(env,geo)       = {r_env_geo:+.3f}")
    print(f"  corr(env,net | geo) = {r_partial:+.3f}   <- if ~0, signal is just geography")

    print("\n=== (2) Within NARROW geo bins, is net still predictive of env? ===")
    for lo,hi in [(0,5),(5,10),(10,15),(15,25)]:
        b=ok&(gp>=lo)&(gp<hi)
        if b.sum()<40: continue
        r=spearmanr(npd[b],ep[b]).correlation
        print(f"  geo {lo:2d}-{hi:2d}km (n={b.sum():5d}): corr(env,net)={r:+.3f}")

    print("\n=== (3) SAME-BASIN pairs only (removes cross-basin contamination) ===")
    m=ok&sameb
    print(f"  same-basin finite pairs: {m.sum()}")
    if m.sum()>50:
        r=spearmanr(npd[m],ep[m]).correlation
        rp=partial_spearman(npd[m],ep[m],gp[m])
        print(f"  corr(env,net)={r:+.3f} | corr(env,net|geo)={rp:+.3f}")
        for thr in [25,15,10,7]:
            n=m&(gp<=thr)
            if n.sum()<30: continue
            md=np.median(npd[n]); a=ep[n&(npd<=md)]; b2=ep[n&(npd>md)]
            if len(a)>10 and len(b2)>10 and np.median(a)>0:
                print(f"    geo<= {thr:2d}km: ratio netfar/netclose = {np.median(b2)/np.median(a):.2f} (n={n.sum()})")

    print("\nINTERPRETATION:")
    print("  partial corr(env,net|geo) clearly >0 AND within-bin corr >0 AND same-basin holds")
    print("  -> network distance adds beyond geography = REAL. Otherwise -> geo confound (other chat right).")
    print(f"\ntotal {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
