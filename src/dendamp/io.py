"""Loading: network, per-segment predictor table, and occurrences (WoC / RivFishTIME).

Contracts (to implement once asset formats are confirmed):

    load_network(path) -> geopandas.GeoDataFrame
        one row per Hydrography90m segment, with geometry and SEGMENT_ID_COL.

    load_predictors(path) -> pandas.DataFrame
        indexed by SEGMENT_ID_COL; columns = GeoFRESH predictors (l_* local, u_* upstream)
        plus topology columns (strahler, flow_accum, subcatchment_id).

    load_occurrences(source, taxon) -> pandas.DataFrame
        columns: lon, lat, accuracy (High/Low or continuous), plus source-specific fields.
        source in {"woc", "rivfishtime", "gbif"}.
"""

from __future__ import annotations


def load_network(path):
    raise NotImplementedError("Implement after confirming network file format (gpkg/parquet).")


def load_predictors(path):
    raise NotImplementedError("Implement after confirming predictor table columns / key.")


def load_occurrences(source: str, taxon: str):
    raise NotImplementedError("Implement per source: woc | rivfishtime | gbif.")
