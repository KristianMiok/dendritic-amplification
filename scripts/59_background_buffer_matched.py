"""Buffer-matched same-terrain background null for dendritic amplification.

Fixes two problems in the basin-matched version:
  (1) other crayfish in the same *basin* are scattered basin-wide, so few fall
      near torrentium at the <5km fine scale -> pool too thin / no valid subsets.
  (2) basin membership does not guarantee the SAME fine-scale terrain.

Here background = non-torrentium crayfish occurrences within BUFFER_KM of ANY
torrentium point (same fine-scale terrain, same network, same sampling), using
the identical partial-Spearman(env, net | geo) at the same fixed <5km scale.

  species >> buffer background -> organization is SPECIES-SPECIFIC (niche).
  species within background     -> shared same-terrain structure; landscape
                                   geometry, not a torrentium-specific effect.

Background occurrences are themselves species, so they may carry their own
niche-network structure -> this is CONSERVATIVE. If torrentium still exceeds
it, the effect is strong.
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
GEO_MAX  = 5.0
BUFFERS  = [15.0, 25.0, 40.0]   # try tightest first; widen only if pool too thin
N_BG     = 300
MIN_POOL = 150
MIN_PAIR = 30
SEED     = 0

def haversine_km(lon, lat):
    lon = np.radians(lon); lat = np.radians(lat)
    dlon = lon[:, None] - lon[None, :]; dlat = lat[:, None] - lat[None, :]
    a = np.sin(dlat/2)**2 + np.cos(lat[:, None])*np.cos(lat[None, :])*np.sin(dlon/2)**2
    return 2*6371*np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def min_dist_to_set_km(lon, lat, slon, slat):
    """min haversine km from each (lon,lat) to the set (slon,slat), chunked."""
    out = np.empty(len(lon))
    sla = np.radians(slat); slo = np.radians(slon)
    for i in range(len(lon)):
        la = np.radians(lat[i]); lo = np.radians(lon[i])
        dlon = slo - lo; dlat = sla - la
        a = np.sin(dlat/2)**2 + np.cos(la)*np.cos(sla)*np.sin(dlon/2)**2
        out[i] = np.min(2*6371*np.arcsin(np.sqrt(np.clip(a, 0, 1))))
    return out

def partial_spear(De, Dn, Dg, mask):
    x, y, zz = Dn[mask], De[mask], Dg[mask]
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(zz)
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

    s = cm[cm["Crayfish_scientific_name"] == SP].copy().reset_index(drop=True)
    slon = s["long_snap"].to_numpy(); slat = s["lat_snap"].to_numpy()
    print(f"{SP}: {len(s)} env-complete high-acc points")

    others = cm[(cm["Crayfish_scientific_name"] != SP) &
                (~cm["subc_id"].isin(set(s["subc_id"])))].copy().reset_index(drop=True)
    d2sp = min_dist_to_set_km(others["long_snap"].to_numpy(), others["lat_snap"].to_numpy(), slon, slat)
    others["km_to_sp"] = d2sp

    chosen = None
    for buf in BUFFERS:
        pool = others[others["km_to_sp"] <= buf]
        print(f"  buffer {buf:>4.0f} km -> background candidates: {len(pool)}")
        if len(pool) >= MIN_POOL:
            chosen = (buf, pool); break
    if chosen is None:
        buf, pool = BUFFERS[-1], others[others["km_to_sp"] <= BUFFERS[-1]]
        print(f"  no buffer reached {MIN_POOL}; using widest {buf:.0f} km with {len(pool)} (interpret cautiously)")
    else:
        buf, pool = chosen
    if len(pool) < 60:
        print("background pool hopelessly thin even at widest buffer; "
              "torrentium simply has few same-terrain crayfish neighbors. STOP."); return

    lon = pd.concat([s["long_snap"], pool["long_snap"]]); lat = pd.concat([s["lat_snap"], pool["lat_snap"]])
    pad = 0.3; bbox = (lon.min()-pad, lat.min()-pad, lon.max()+pad, lat.max()+pad)
    tt = time.time()
    net = read_dataframe(GPKG, columns=["stream", "next_stream", "length"],
                         layer="merged", bbox=bbox, read_geometry=False)
    net["stream"] = net["stream"].astype("int64"); net["next_stream"] = net["next_stream"].astype("int64")
    G = nx.Graph(); v = net[net["next_stream"] != 0]
    G.add_weighted_edges_from(zip(v["stream"].to_numpy(), v["next_stream"].to_numpy(),
                                  v["length"].to_numpy()/1000.0))
    print(f"  graph {G.number_of_nodes()} nodes ({time.time()-tt:.0f}s)")

    s2 = s[s["subc_id"].isin(G)].reset_index(drop=True)
    bg = pool[pool["subc_id"].isin(G)].drop_duplicates("subc_id").reset_index(drop=True)
    print(f"  in-graph: species={len(s2)}  buffer-background pool={len(bg)}")
    if len(s2) < 100 or len(bg) < 60:
        print("too few after graph filter; STOP."); return

    env_s = s2[env_cols].to_numpy(float); env_b = bg[env_cols].to_numpy(float)
    sc = StandardScaler().fit(np.vstack([env_s, env_b]))
    pca = PCA(n_components=N_PCA, random_state=0).fit(sc.transform(np.vstack([env_s, env_b])))
    Es = pca.transform(sc.transform(env_s)); Eb = pca.transform(sc.transform(env_b))

    tt = time.time()
    Dn_s = all_pairs_net(G, s2["subc_id"]); De_s = squareform(pdist(Es))
    Dg_s = haversine_km(s2["long_snap"].to_numpy(), s2["lat_snap"].to_numpy())
    iu = np.triu_indices(len(s2), k=1)
    m = np.isfinite(Dn_s[iu]) & (Dg_s[iu] >= GEO_MIN) & (Dg_s[iu] < GEO_MAX)
    obs = partial_spear(De_s[iu], Dn_s[iu], Dg_s[iu], m)
    print(f"  species: {m.sum()} valid <{GEO_MAX}km pairs | partial r = {obs:+.3f}  ({time.time()-tt:.0f}s)")

    tt = time.time()
    Dn_b = all_pairs_net(G, bg["subc_id"]); De_b = squareform(pdist(Eb))
    Dg_b = haversine_km(bg["long_snap"].to_numpy(), bg["lat_snap"].to_numpy())
    print(f"  background all-pairs computed ({time.time()-tt:.0f}s)")

    # subsample size = min(species n, pool) so replace=False always valid
    sub_n = min(len(s2), len(bg)); npool = len(bg); stats = []
    for _ in range(N_BG):
        idx = rng.choice(npool, sub_n, replace=False)
        De = De_b[np.ix_(idx, idx)]; Dn = Dn_b[np.ix_(idx, idx)]; Dg = Dg_b[np.ix_(idx, idx)]
        ii = np.triu_indices(sub_n, k=1)
        mm = np.isfinite(Dn[ii]) & (Dg[ii] >= GEO_MIN) & (Dg[ii] < GEO_MAX)
        if mm.sum() < MIN_PAIR: continue
        stats.append(partial_spear(De[ii], Dn[ii], Dg[ii], mm))
    stats = np.array(stats)
    print(f"\n  buffer used: {buf:.0f} km | subsample size: {sub_n} | valid subsets: {len(stats)}/{N_BG}")
    if len(stats) < 30:
        print("  still too few valid background subsets: torrentium's same-terrain crayfish")
        print("  neighbors are too sparse at <5km to form a background. This itself suggests")
        print("  torrentium occupies fine-scale positions few other crayfish share -- report as")
        print("  a limitation; the background null is not computable for this species/scale.")
        return

    p_bg = (np.sum(stats >= obs) + 1) / (len(stats) + 1)
    zsc = (obs - stats.mean()) / (stats.std() + 1e-9)
    print(f"\n=== BUFFER-MATCHED SAME-TERRAIN BACKGROUND NULL: {SP} ===")
    print(f"  species partial r        : {obs:+.3f}")
    print(f"  background mean +/- sd    : {stats.mean():+.3f} +/- {stats.std():.3f}")
    print(f"  background 5/50/95 pct    : {np.percentile(stats,5):+.3f} / "
          f"{np.percentile(stats,50):+.3f} / {np.percentile(stats,95):+.3f}")
    print(f"  species z vs background   : {zsc:+.2f}")
    print(f"  p(background >= species)  : {p_bg:.3f}")
    print(f"\n  p<0.05 & high z -> SPECIES-SPECIFIC (niche), beyond same-terrain geometry.")
    print(f"  within background        -> same-terrain landscape structure, not species-specific.")
    print(f"\ntotal {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
