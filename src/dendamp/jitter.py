"""Controlled displacement — the core experimental manipulation.

Two modes, deliberately kept separate:

1. geographic(points, d_m, rng)
   Displace each (lon, lat) by a known distance d_m in a random bearing, then RE-SNAP to the
   network. Faithful to how real positional error occurs in space; requires the network geometry.

2. network_space(segment_ids, network, d_m, rng)
   Reassign each record from segment S to a neighbouring segment S' at a controlled network
   distance ~ d_m, drawn from the EXISTING per-segment set. Uses only the predictor table you
   already have -> ZERO new GeoFRESH extraction. Also the most direct operationalisation of
   dendritic amplification (you know exactly which topological boundaries were crossed).

Both return, per record: new_segment_id (+ realised displacement), so downstream code can look
up predictors and log topology effects.

Design choice to settle with data in hand: which mode is primary. network_space is cheaper and
cleaner for the amplification claim; geographic is more intuitive for reviewers. Likely: report
network_space as primary, geographic as a robustness check.
"""

from __future__ import annotations


def geographic(points, d_m, rng):
    raise NotImplementedError("Displace by d_m in random bearing (metric CRS), then re-snap.")


def network_space(segment_ids, network, d_m, rng):
    raise NotImplementedError("Reassign to a neighbour segment at controlled network distance ~ d_m.")
