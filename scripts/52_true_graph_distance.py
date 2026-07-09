"""TRUE along-stream distance — spatial bbox crop BEFORE building the graph.
Read only network segments within the occurrence bounding box (pyogrio bbox filter),
so the graph is ~100k nodes not 13.9M. Then Dijkstra is fast.
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
from scipy.stats import mannwhitneyu

GPKG="data/raw/h90m/order_vect_segment_h18v04.gpkg"
MASTER=os.path.expanduser("~/Desktop/Papers/sdm-robustness/data/raw/combined_data_true_master.csv")
SPECIES=["Austropotamobius pallipes","Pacifastacus leniusculus"]
N_PCA=10

def haversine_km(lon,lat):
    lon=np.radians(lon); lat=np.radians(lat)
    dlon=lon[:,None]-lon[None,:]; dlat=lat[:,None]-lat[None,:]
    a=np.sin(dlat/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    return 2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))

def load_species():
    cm=pd.read_csv(MASTER,encoding="latin-1",low_memory=False)
    cm=cm[cm["Accuracy"].astype(str).str.strip().str.lower().eq("high")]
    env_cols=[c for c in cm.columns if c.startswith(("l_","u_"))]
    cm=cm.dropna(subset=["long_snap","lat_snap","subc_id","basin_id"])
    cm=cm[cm["long_snap"].abs()>0.01].drop_duplicates("subc_id")
    return cm, env_cols

def main():
    t0=time.time()
    cm, env_cols = load_species()

    for sp in SPECIES:
        s=cm[cm["Crayfish_scientific_name"]==sp].copy()
        s["subc_id"]=s["subc_id"].astype("int64")
        s=s[s[env_cols].notna().all(axis=1)].reset_index(drop=True)
        if len(s)<120: print(f"\n{sp}: too few; skip"); continue
        # bbox around this species' points (small pad)
        pad=0.3
        bbox=(s.long_snap.min()-pad, s.lat_snap.min()-pad, s.long_snap.max()+pad, s.lat_snap.max()+pad)
        print(f"\n===== {sp}: {len(s)} pts | bbox lon[{bbox[0]:.1f},{bbox[2]:.1f}] lat[{bbox[1]:.1f},{bbox[3]:.1f}] =====")

        tt=time.time()
        net=read_dataframe(GPKG, columns=["stream","next_stream","length"], layer="merged",
                           bbox=bbox, read_geometry=False)
        net["stream"]=net["stream"].astype("int64"); net["next_stream"]=net["next_stream"].astype("int64")
        print(f"  segments in bbox: {len(net)} ({time.time()-tt:.0f}s)")

        G=nx.Graph()
        valid=net[net["next_stream"]!=0]
        G.add_weighted_edges_from(zip(valid["stream"].to_numpy(), valid["next_stream"].to_numpy(),
                                      valid["length"].to_numpy()/1000.0))
        # keep only occurrence points present in this cropped graph
        s=s[s["subc_id"].isin(G)].reset_index(drop=True)
        print(f"  graph {G.number_of_nodes()} nodes | occurrence pts in graph: {len(s)}")
        if len(s)<100: print("  too few in cropped graph; skip"); continue

        tt=time.time()
        targets=list(s["subc_id"]); tset=set(targets); idx={n:i for i,n in enumerate(targets)}
        D=np.full((len(targets),len(targets)),np.nan)
        for src in targets:
            L=nx.single_source_dijkstra_path_length(G,src,weight="weight")
            i=idx[src]
            for dst in tset & L.keys(): D[i,idx[dst]]=L[dst]
        print(f"  distances computed ({time.time()-tt:.0f}s), finite frac={np.isfinite(D).mean():.2f}")

        er=PCA(n_components=N_PCA,random_state=0).fit_transform(StandardScaler().fit_transform(s[env_cols].to_numpy(float)))
        D_env=squareform(pdist(er)); D_geo=haversine_km(s.long_snap.to_numpy(),s.lat_snap.to_numpy())
        iu=np.triu_indices(len(s),k=1); gp,ep,npd=D_geo[iu],D_env[iu],D[iu]
        ok=np.isfinite(npd); gp,ep,npd=gp[ok],ep[ok],npd[ok]
        print(f"  finite net-dist pairs: {ok.sum()} / {len(ok)} ({100*ok.mean():.0f}% connected)")
        print(f"  {'geo≤km':>7} {'n':>6} {'env|netclose':>13} {'env|netfar':>11} {'ratio':>6} {'p':>9}")
        for thr in [50,25,15,10,7]:
            n=gp<=thr
            if n.sum()<40: continue
            med=np.median(npd[n]); a=ep[n&(npd<=med)]; b=ep[n&(npd>med)]
            if len(a)>10 and len(b)>10 and np.median(a)>0:
                _,p=mannwhitneyu(b,a,alternative="greater")
                print(f"  {thr:7.0f} {n.sum():6d} {np.median(a):13.2f} {np.median(b):11.2f} "
                      f"{np.median(b)/np.median(a):6.2f} {p:9.1e}")
    print(f"\ntotal {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
