"""CORE (model-free): the amplification function Δenv(d).

This is the novel, cheap contribution. No SDM is fitted here.

Idea:
    For each High-Accuracy benchmark record on segment S:
      for each displacement level d in DISPLACEMENT_GRID_M:
        for each replicate:
          displace -> new segment S'  (jitter.network_space or jitter.geographic)
          look up predictor vectors at S and S'
          Δenv = standardized distance between them (e.g. Euclidean in z-scored predictor space)
    Return a tidy table: [entity, d, replicate, record_id, delta_env, delta_strahler,
                          crossed_catchment, delta_flow_acc]

Outputs feed two claims:
  (1) Δenv(d) curve  -> is env shift proportional (terrestrial) or discontinuous (dendritic)?
  (2) mechanism      -> is Δenv driven by boundary crossings rather than metres?

Cost: pure lookups + vector distances. Seconds-to-minutes on a laptop.

Contract:
    delta_env(benchmark_df, predictors, network, d_grid, n_rep, rng, mode="network") -> DataFrame
"""

from __future__ import annotations

from . import config


def delta_env(benchmark_df, predictors, network, d_grid=None, n_rep=None, rng=None, mode="network"):
    if d_grid is None:
        d_grid = config.DISPLACEMENT_GRID_M
    if n_rep is None:
        n_rep = config.N_REPLICATES
    raise NotImplementedError(
        "Implement the displace -> lookup -> standardized-distance loop; return the tidy table."
    )
