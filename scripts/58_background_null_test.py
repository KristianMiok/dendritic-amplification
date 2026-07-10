"""SAME-REGION BACKGROUND NULL for dendritic amplification.

The prior signal-vs-null contrast compared species in DIFFERENT tiles
(torrentium/leptodactylus in h20v04 vs pallipes/leniusculus in h18v04),
so "signal vs null" was fully confounded with region/terrain.

This removes that confound: target species vs TARGET-GROUP BACKGROUND --
other crayfish occurrences in the SAME basins (same network, same terrain,
same sampling) -- using the identical partial-Spearman(env, net | geo)
statistic at the same fixed <5km scale.

  species >> background  -> organization is SPECIES-SPECIFIC (niche),
                            not terrain. Rules out pure landscape geometry.
  species within backgr. -> shared with same-region crayfish -> landscape
                            geometry / shared habitat, NOT species-specific;
                            the cross-tile contrast was a regional artifact.

Note: target-group background points may themselves carry niche-network
structure, so this is CONSERVATIVE (background can be "too organized").
If the species still exceeds it, the effect is strong.
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

MASTER   = os.path.expanduser("~/Desktop/Papers/sdm-robustness/data/raw/combined_data_true_master.csv")
SP       = "Austropotamobius torrentium"
GPKG     = "data/raw/h90m/order_vect_segment_h20v04.gpkg"
N_PCA    = 10
GEO_MIN  = 0.5
GEO_MAX  = 5.0        # same fixed scale as the node-permutation test
N_BG     = 300        # background subsets
POOL_CAP = 1000       # cap on background candidate pool (Dijkstra cost ~ pool size)
MIN_POOL = 150        # below this the null is unreliable
MIN_PAIR = 30         # min valid <5km connected pairs for a subset stat to count
SEED     = 0

def haversine_km(lon, lat):
    lon = np.radians(lon); lat = np.radians(lat)
    dlon = lon[:, None] - lon[None, :]; dlat = lat[:, None] - lat[None, :]
    a = np.sin(dlat/2)**2 + np.cos(lat[:, None])*np.cos(lat[None, :])*np.sin(dlon/2)**2
    return 2*6371*np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def partial_spear(De, Dn, Dg, mask):
    x, y, z = Dn[mask], De[mask], Dg[mask]
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b = np.c_[np.ones_like(b), b]
        c, _, _, _ = np.linalg.lstsq(b, a, rcond=None)
        return a - b @ c
    return np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1]

def all_pairs_net(G, ids):
    ids = list(ids); idx = {n: i for i, n in enumerate(ids)}; idset = set(ids)
    D = np.full((len(ids), len(ids)), np.nan)
    for src in ids:
        L = nx.single_source_dijkstra_path_length(G, src, weight="weight")
        i = idx[src]
        for dst in idset & L.keys():
            D[i, idx[dst]] = L[dst]
    return D

def main():
    t0 = time.time(); rng = np.random.default_rng(SEED)

    cm = pd.read_csv(MASTER, encoding="latin-1", low_memory=False)
    cm = cm[cm["Accuracy"].astype(str).str.strip().str.lower().eq("high")]
    env_cols = [c for c in cm.columns if c.startswith(("l_", "u_"))]
    cm = cm.dropna(subset=["long_snap", "lat_snap", "subc_id", "basin_id"])
    cm = cm[cm["long_snap"].abs() > 0.01].drop_duplicates("subc_id")
    cm["subc_id"] = cm["subc_id"].astype("int64")
    cm = cm[cm[env_cols].notna().all(axis=1)].reset_index(drop=True)

    s = cm[cm["Crayfish_scientific_name"] == SP].copy()
    basins_sp = set(s["basin_id"].unique())
    print(f"{SP}: {len(s)} env-complete high-acc points in {len(basins_sp)} basins")

    bg = cm[(cm["Crayfish_scientific_name"] != SP) &
            (cm["basin_id"].isin(basins_sp)) &
            (~cm["subc_id"].isin(set(s["subc_id"])))].copy()
    print(f"target-group background candidates (other crayfish, same basins): {len(bg)}")
    if len(bg) < MIN_POOL:
        print(f"POOL TOO SMALL (<{MIN_POOL}); relax basin->buffer or widen. STOP."); return

    # union bbox so BOTH species and background fall inside the graph
    lon = pd.concat([s["long_snap"], bg["long_snap"]]); lat = pd.concat([s["lat_snap"], bg["lat_snap"]])
    pad = 0.3; bbox = (lon.min()-pad, lat.min()-pad, lon.max()+pad, lat.max()+pad)
    tt = time.time()
    net = read_dataframe(GPKG, columns=["stream", "next_stream", "length"],
                         layer="merged", bbox=bbox, read_geometry=False)
    net["stream"] = net["stream"].astype("int64"); net["next_stream"] = net["next_stream"].astype("int64")
    G = nx.Graph(); v = net[net["next_stream"] != 0]
    G.add_weighted_edges_from(zip(v["stream"].to_numpy(), v["next_stream"].to_numpy(),
                                  v["length"].to_numpy()/1000.0))
    print(f"graph {G.number_of_nodes()} nodes ({time.time()-tt:.0f}s)")

    s  = s[s["subc_id"].isin(G)].reset_index(drop=True)
    bg = bg[bg["subc_id"].isin(G)].drop_duplicates("subc_id").reset_index(drop=True)
    print(f"in-graph: species={len(s)}  background pool={len(bg)}")
    if len(s) < 100 or len(bg) < MIN_POOL:
        print("too few after graph filter; STOP."); return
    if len(bg) > POOL_CAP:
        bg = bg.iloc[rng.choice(len(bg), POOL_CAP, replace=False)].reset_index(drop=True)
        print(f"pool capped to {POOL_CAP}")

    # common env embedding (fit on union so species & background share the SAME env space)
    env_s = s[env_cols].to_numpy(float); env_b = bg[env_cols].to_numpy(float)
    sc = StandardScaler().fit(np.vstack([env_s, env_b]))
    pca = PCA(n_components=N_PCA, random_state=0).fit(sc.transform(np.vstack([env_s, env_b])))
    Es = pca.transform(sc.transform(env_s)); Eb = pca.transform(sc.transform(env_b))

    # species statistic
    tt = time.time()
    Dn_s = all_pairs_net(G, s["subc_id"]); De_s = squareform(pdist(Es))
    Dg_s = haversine_km(s["long_snap"].to_numpy(), s["lat_snap"].to_numpy())
    iu = np.triu_indices(len(s), k=1)
    m = np.isfinite(Dn_s[iu]) & (Dg_s[iu] >= GEO_MIN) & (Dg_s[iu] < GEO_MAX)
    obs = partial_spear(De_s[iu], Dn_s[iu], Dg_s[iu], m)
    print(f"species: {m.sum()} valid <{GEO_MAX}km pairs | partial r(env,net|geo) = {obs:+.3f}  ({time.time()-tt:.0f}s)")

    # background all-pairs (the expensive step)
    tt = time.time()
    Dn_b = all_pairs_net(G, bg["subc_id"]); De_b = squareform(pdist(Eb))
    Dg_b = haversine_km(bg["long_snap"].to_numpy(), bg["lat_snap"].to_numpy())
    print(f"background all-pairs computed ({time.time()-tt:.0f}s)")

    # background null: random n-sized subsets from the same-region pool
    n_sp = len(s); npool = len(bg); stats = []
    for _ in range(N_BG):
        idx = rng.choice(npool, n_sp, replace=False)
        De = De_b[np.ix_(idx, idx)]; Dn = Dn_b[np.ix_(idx, idx)]; Dg = Dg_b[np.ix_(idx, idx)]
        ii = np.triu_indices(n_sp, k=1)
        mm = np.isfinite(Dn[ii]) & (Dg[ii] >= GEO_MIN) & (Dg[ii] < GEO_MAX)
        if mm.sum() < MIN_PAIR: continue
        stats.append(partial_spear(De[ii], Dn[ii], Dg[ii], mm))
    stats = np.array(stats)
    print(f"\nvalid background subsets: {len(stats)}/{N_BG}")
    if len(stats) < 30:
        print("too few valid subsets (background sparser than species at <5km); "
              "tighten background to a buffer around species points before trusting this.")
        return

    p_bg = (np.sum(stats >= obs) + 1) / (len(stats) + 1)
    z = (obs - stats.mean()) / (stats.std() + 1e-9)
    print(f"\n=== SAME-REGION BACKGROUND NULL: {SP} ===")
    print(f"  species partial r           : {obs:+.3f}")
    print(f"  background mean +/- sd       : {stats.mean():+.3f} +/- {stats.std():.3f}")
    print(f"  background 5th/50th/95th     : {np.percentile(stats,5):+.3f} / "
          f"{np.percentile(stats,50):+.3f} / {np.percentile(stats,95):+.3f}")
    print(f"  species z vs background      : {z:+.2f}")
    print(f"  p(background >= species)     : {p_bg:.3f}")
    print(f"\n  species >> background (p<0.05, high z) -> organization is SPECIES-SPECIFIC")
    print(f"      (niche), not terrain. Rules out pure landscape geometry.")
    print(f"  species within background    -> SHARED with same-region crayfish ->")
    print(f"      landscape geometry / shared habitat; cross-tile contrast was regional.")
    print(f"\ntotal {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
