"""Scan combined master for crayfish species usable in the Delta-env(d) gradient method,
using basin_id as the 'network branch' grouping (general replacement for lineage).
Report per species: n High-accuracy records, n basins, geographic spread (range proxy).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

MASTER = os.path.expanduser("~/Desktop/Papers/sdm-robustness/data/raw/combined_data_true_master.csv")

def rd():
    for e in ("utf-8","latin-1","cp1252"):
        try: return pd.read_csv(MASTER, encoding=e, low_memory=False)
        except (UnicodeDecodeError,) : pass
    raise RuntimeError("encoding")

def main():
    df = rd()
    print("master shape:", df.shape)
    acc = "Accuracy"
    hi = df[df[acc].astype(str).str.strip().str.lower().eq("high")].copy()
    hi = hi.dropna(subset=["long_snap","lat_snap","subc_id","basin_id","Crayfish_scientific_name"])
    hi = hi.drop_duplicates("subc_id")   # one per network cell (dedup as agreed)
    print(f"High-accuracy, dedup by subc_id: {len(hi)} records\n")

    g = hi.groupby("Crayfish_scientific_name").agg(
        n=("subc_id","nunique"),
        n_basins=("basin_id","nunique"),
        lon_span=("long_snap", lambda s: s.max()-s.min()),
        lat_span=("lat_snap", lambda s: s.max()-s.min()),
    ).reset_index()
    g["geo_span_km"] = np.maximum(g.lon_span, g.lat_span)*111
    g = g[(g.n>=150) & (g.n_basins>=3)].sort_values("n", ascending=False)
    print("species usable for the method (>=150 cells, >=3 basins):")
    print(f"{'species':40s} {'n_cells':>8} {'n_basins':>9} {'geo_span_km':>12}")
    for _,r in g.iterrows():
        print(f"{r['Crayfish_scientific_name']:40s} {int(r.n):8d} {int(r.n_basins):9d} {r.geo_span_km:12.0f}")

    print("\nSuggested mobility/range gradient picks:")
    print("  - NARROW specialist (small geo span):", g.nsmallest(3,'geo_span_km')['Crayfish_scientific_name'].tolist())
    print("  - WIDE generalist (large geo span):  ", g.nlargest(3,'geo_span_km')['Crayfish_scientific_name'].tolist())

if __name__ == "__main__":
    main()
