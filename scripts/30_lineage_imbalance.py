"""Real test on master_lineage.xlsx: does network-defined lineage predict environment
better than geographic position? No downloads, no network reconstruction needed.

II(lineage_onehot -> env)  vs  II(geo -> env), plus a categorical-aware check.
Lineage = Haplogroup (phylogenetic lines defined by position in the river network).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dendamp.imbalance import compare_geo_net_env, imbalance

XLSX = os.path.expanduser("~/Desktop/Papers/crayfish-niche-shift/data/raw/master_lineage.xlsx")
N_PCA = 10
MIN_PER_GROUP = 20

def main():
    df = pd.read_excel(XLSX)
    env_cols = [c for c in df.columns if c.startswith(("l_", "u_"))]
    df = df.dropna(subset=["long_snap", "lat_snap", "LABEL"]).copy()

    keep = df["LABEL"].value_counts()
    keep = keep[keep >= MIN_PER_GROUP].index.tolist()
    df = df[df["LABEL"].isin(keep)].copy()
    envm = df[env_cols].to_numpy(float)
    ok = np.isfinite(envm).all(axis=1)
    df, envm = df[ok].reset_index(drop=True), envm[ok]
    print(f"N = {len(df)} | lineages kept (>= {MIN_PER_GROUP}): {df['LABEL'].value_counts().to_dict()}")
    print(f"env predictors: {len(env_cols)}")

    geo = df[["long_snap", "lat_snap"]].to_numpy(float)
    env_red = PCA(n_components=N_PCA, random_state=0).fit_transform(StandardScaler().fit_transform(envm))

    codes = pd.Categorical(df["LABEL"]).codes.reshape(-1, 1).astype(float)
    onehot = pd.get_dummies(df["LABEL"]).to_numpy(float)

    X_geo = np.c_[geo, env_red]
    d_geo_env, _ = imbalance(X_geo, [0, 1], list(range(2, 2 + N_PCA)))

    ng = onehot.shape[1]
    X_lin = np.c_[onehot, env_red]
    d_lin_env, _ = imbalance(X_lin, list(range(ng)), list(range(ng, ng + N_PCA)))

    print("\n== Information Imbalance (real lineage data) ==")
    print(f"  Delta(geo      -> env) = {d_geo_env:.3f}")
    print(f"  Delta(lineage  -> env) = {d_lin_env:.3f}")
    print(f"  ratio geo/lineage      = {d_geo_env / d_lin_env:.2f}")
    print("\nDecision: lineage more informative about env than geography?  ->",
          d_lin_env < d_geo_env)
    print("(lineage = network-defined phylogenetic group; geo = snapped coordinates)")

if __name__ == "__main__":
    main()
