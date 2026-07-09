"""Information Imbalance layer (OPTIONAL).

Isolated wrapper around DADApy's Information Imbalance (Glielmo, Zeni, Cheng, Csanyi, Laio,
"Ranking the information content of distance measures", PNAS Nexus 1(2):pgac039, 2022;
implemented in DADApy, Glielmo et al., Patterns 3:100589, 2022). Lazy-imported so the core
instrument stays dependency-light and reproducible without dadapy installed.

Information Imbalance  Delta(A -> B), in [0, ~1], asymmetric:
    ~0  -> distances in space A predict the nearest-neighbour structure of B (A informs B)
    ~1  -> A and B are independent

Pilot decision rule (see scripts/10_pilot_imbalance.py):
    Delta(net -> env)  clearly <  Delta(geo -> env)      -> the NETWORK metric is more
        informative about environment -> Information Imbalance earns backbone status.
    terrestrial control Delta(geo -> env) is LOW         -> the poor geo-predictiveness seen
        in the dendritic case is network-specific, not a generic artifact.

NOTE for real data: `net` here is a network *coordinate/embedding* and Euclidean distance is
used. For production, network distance should be along-stream distance. Two clean options when
the real network is wired: (a) supply a low-distortion 1-D network embedding as `net`; or
(b) extend to compare a precomputed along-stream distance matrix via dadapy's distance-based API.
Kept as a coordinate here because it is exact for the synthetic pilot and unblocks the decision.
"""

from __future__ import annotations

import numpy as np


def _zscore(A):
    A = np.asarray(A, float)
    return (A - A.mean(0)) / (A.std(0) + 1e-12)


def _mc(X):
    from dadapy import MetricComparisons
    X = np.atleast_2d(_zscore(X))
    return MetricComparisons(coordinates=X, maxk=X.shape[0] - 1)


def imbalance(X, cols_a, cols_b, k: int = 1):
    """Return (Delta(A->B), Delta(B->A)) between two column-subsets of X (z-scored, Euclidean)."""
    mc = _mc(X)
    a, b = mc.return_inf_imb_two_selected_coords(list(cols_a), list(cols_b), k=k)
    return float(a), float(b)


def compare_geo_net_env(geo, net, env, k: int = 1) -> dict:
    """Key Information Imbalances for one entity.

    geo : (N, 2) geographic coordinates
    net : (N, 1) network coordinate / embedding
    env : (N, d) environmental predictor vectors
    """
    geo = np.asarray(geo, float).reshape(len(geo), -1)
    net = np.asarray(net, float).reshape(len(net), -1)
    env = np.asarray(env, float).reshape(len(env), -1)
    X = np.c_[geo, net, env]
    ng, nn = geo.shape[1], net.shape[1]
    g = list(range(0, ng))
    n = list(range(ng, ng + nn))
    e = list(range(ng + nn, X.shape[1]))
    d_geo_env, d_env_geo = imbalance(X, g, e, k)
    d_net_env, d_env_net = imbalance(X, n, e, k)
    return {
        "geo->env": d_geo_env,
        "net->env": d_net_env,
        "env->geo": d_env_geo,
        "env->net": d_env_net,
        "ratio_geo_over_net": (d_geo_env / d_net_env) if d_net_env > 0 else float("inf"),
    }
