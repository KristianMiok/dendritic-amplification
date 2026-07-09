"""Extract real network position (out_dist) for torrentium h18v04 segments from the
Hydrography90m order_vect network, then run the Delta-env(d) gradient with REAL along-network
distance instead of lineage/basin proxy.

net = out_dist (distance-to-outlet, metres) -> continuous, no ties.
Also builds pairwise |out_dist_i - out_dist_j| as an along-network distance proxy
(true shortest-path can come later; out_dist difference is monotone along a branch).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pyogrio import read_dataframe
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

GPKG = "data/raw/h90m/order_vect_segment_h18v04.gpkg"
XLSX = os.path.expanduser("~/Desktop/Papers/crayfish-niche-shift/data/raw/master_lineage.xlsx")
N_PCA = 10

def haversine_km(lon,lat):
    lon=np.radians(lon); lat=np.radians(lat)
    dlon=lon[:,None]-lon[None,:]; dlat=lat[:,None]-lat[None,:]
    a=np.sin(dlat/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    return 2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))

def main():
    # torrentium records with env + geo + subc_id (from master_lineage, h18v04 subset)
    df = pd.read_excel(XLSX)
    env_cols=[c for c in df.columns if c.startswith(("l_","u_"))]
    df=df.dropna(subset=["long_snap","lat_snap","subc_id","LABEL"]).copy()
    df=df[df["LABEL"].astype(str).str.contains("torrentium")].drop_duplicates("subc_id")

    # pull network attributes only for the segments we need
    want=set(df["subc_id"].astype("int64"))
    print("reading network (filtered to our segments)...")
    net = read_dataframe(GPKG, columns=["stream","out_dist","strahler","flow_accum","cum_length"],
                         layer="merged", read_geometry=False)
    net = net[net["stream"].astype("int64").isin(want)].copy()
    net = net.rename(columns={"stream":"subc_id"})
    print(f"matched {len(net)} / {len(want)} torrentium segments in the network")

    m = df.merge(net, on="subc_id", how="inner")
    em=m[env_cols].to_numpy(float); ok=np.isfinite(em).all(axis=1)
    m,em=m[ok].reset_index(drop=True),em[ok]
    print(f"final N with env + network position: {len(m)}")
    if len(m)<100:
        print("too few matched; stop."); return

    er=PCA(n_components=N_PCA,random_state=0).fit_transform(StandardScaler().fit_transform(em))
    D_env=squareform(pdist(er))
    D_geo=haversine_km(m.long_snap.to_numpy(), m.lat_snap.to_numpy())
    # real along-network distance proxy: absolute difference in distance-to-outlet (km)
    od=m["out_dist"].to_numpy(float)/1000.0
    D_net=np.abs(od[:,None]-od[None,:])

    iu=np.triu_indices(len(m),k=1)
    gp,ep,npair=D_geo[iu],D_env[iu],D_net[iu]

    print("\n=== Does env-divergence track NETWORK distance beyond geographic distance? ===")
    print("Among geographically-near pairs (<=15 km), split by network distance:")
    near=gp<=15
    net_near = npair<=np.median(npair[near])   # closer along network
    for lab,mask in [("net-CLOSE (<=med)",near&net_near),("net-FAR (>med)",near&~net_near)]:
        if mask.sum()>10:
            print(f"  {lab:20s} n={mask.sum():5d} median env-dist={np.median(ep[mask]):.2f}")
    a=ep[near&net_near]; b=ep[near&~net_near]
    if len(a)>10 and len(b)>10 and np.median(a)>0:
        from scipy.stats import mannwhitneyu
        _,p=mannwhitneyu(b,a,alternative="greater")
        print(f"  ratio (net-far / net-close) = {np.median(b)/np.median(a):.2f} | p={p:.1e}")

    print("\n=== Gradient: env-gap between network-distant pairs as geography tightens ===")
    print(f"{'geo≤km':>8} {'n':>7} {'ratio netfar/netclose':>22}")
    for thr in [50,25,15,10,7,5]:
        n=gp<=thr
        if n.sum()<50: continue
        med=np.median(npair[n]); nc=n&(npair<=med); nf=n&(npair>med)
        a=ep[nc]; b=ep[nf]
        if len(a)>10 and len(b)>10 and np.median(a)>0:
            print(f"{thr:8.0f} {n.sum():7d} {np.median(b)/np.median(a):22.2f}")
    print("\nRising ratio = network distance drives env-divergence beyond geography (real Delta-env(d)).")

if __name__ == "__main__":
    main()
