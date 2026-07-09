"""Snap coordinates to the nearest Hydrography90m segment.

Contract:
    snap(points_gdf, network_gdf) -> DataFrame[segment_id, snap_distance_m]

Notes:
- Use a projected CRS (metres) for distances; reproject before nearest join.
- geopandas.sjoin_nearest is the natural tool; rtree/pygeos speeds it up.
- This is reused both for the benchmark build and for re-snapping jittered points.
"""

from __future__ import annotations


def snap(points_gdf, network_gdf):
    raise NotImplementedError(
        "Implement with geopandas.sjoin_nearest in a metric CRS; return segment_id + snap_distance_m."
    )
