"""Real (geo, net, env) benchmark for Austropotamobius torrentium + Information Imbalance pilot.

env joined per WoCID from GF_l_*.csv (all ~1700 records); each layer contributes ONLY its
l_*/u_* predictor columns (drop reg_id/basin_id/subc_id/strahler to avoid merge collisions).
env reduced with PCA (~10 comps) before II.
net = [strahler, log10(upstream area)] -> ordinal/continuous hierarchy position, LOWER bound.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dendamp.imbalance import compare_geo_net_env

ECO = os.path.expanduser("~/Desktop/Lucian/Global/Descriptive Paper/Data/Eco Variables/eco_variables_tables")
SPECIES = "Austropotamobius torrentium"
N_PCA = 10


def rd(name, cols=None):
    last = None
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(os.path.join(ECO, name), usecols=cols, encoding=enc)
        except (UnicodeDecodeError, ValueError) as e:
            last = e
    raise last


def env_layer(name):
    d = rd(name)
    keep = ["WoCID"] + [c for c in d.columns if c.startswith(("l_", "u_"))]
    return d[keep]


def main():
    woc = rd("WoCcut.csv", ["WoCID", "Accuracy", "Crayfish_scientific_name"])
    woc["sp"] = woc["Crayfish_scientific_name"].astype(str).str.strip().str.lower()
    sp = woc[woc["sp"] == SPECIES.lower()]
    high = sp[sp["Accuracy"].astype(str).str.strip().str.lower().eq("high")][["WoCID"]]
    print(f"{SPECIES}: High-Accuracy records = {len(high)}")

    snapped = rd("WoC_snapped.csv", ["WoCID", "long_snap", "lat_snap", "subc_id", "strahler"])
    la = rd("GF_l_subcatchment_area.csv", ["WoCID", "area_sqm"])
    ua = rd("GF_u_subcatchment_area.csv", ["WoCID", "sum_area_sqm"])

    env = env_layer("GF_l_climate.csv")
    for f in ["GF_l_topography.csv", "GF_l_soil.csv", "GF_l_landcover.csv"]:
        env = env.merge(env_layer(f), on="WoCID", how="inner")
    env_cols = [c for c in env.columns if c.startswith(("l_", "u_"))]
    print(f"env matrix: {env.shape[0]} rows x {len(env_cols)} predictors")

    df = (high
          .merge(snapped, on="WoCID", how="inner")
          .merge(la, on="WoCID", how="left")
          .merge(ua, on="WoCID", how="left")
          .merge(env, on="WoCID", how="inner"))
    df = df.dropna(subset=["long_snap", "lat_snap", "strahler"]).drop_duplicates("WoCID")

    envm = df[env_cols].to_numpy(float)
    row_ok = np.isfinite(envm).all(axis=1)
    df, envm = df[row_ok].reset_index(drop=True), envm[row_ok]
    print(f"benchmark after join + NA filter: N = {len(df)} "
          f"| strahler {int(df['strahler'].min())}-{int(df['strahler'].max())}")
    if len(df) < 100:
        print("Still too few rows; stop and inspect."); return

    geo = df[["long_snap", "lat_snap"]].to_numpy(float)
    area = df["sum_area_sqm"].fillna(df["area_sqm"]).clip(lower=1.0).to_numpy(float)
    net = np.c_[df["strahler"].to_numpy(float), np.log10(area)]

    Xs = StandardScaler().fit_transform(envm)
    pca = PCA(n_components=N_PCA, random_state=0).fit(Xs)
    env_red = pca.transform(Xs)
    print(f"env reduced to {N_PCA} PCA comps (var explained ~ {pca.explained_variance_ratio_.sum():.2f})")

    res = compare_geo_net_env(geo, net, env_red)
    print("\n== Information Imbalance (REAL data, torrentium; net = lower-bound proxy) ==")
    print(f"  N = {len(df)}")
    print(f"  Delta(geo->env) = {res['geo->env']:.3f}")
    print(f"  Delta(net->env) = {res['net->env']:.3f}")
    print(f"  ratio geo/net   = {res['ratio_geo_over_net']:.2f}")
    print("\nDecision: net more informative than geo?  ->", res["net->env"] < res["geo->env"])
    print("(strahler+area proxy, not along-stream distance -> this is a LOWER bound.)")


if __name__ == "__main__":
    main()
