"""Degradation(d): the downstream consequence of Δenv, measured on the bounded regional domain.

For each entity / algorithm / displacement level:
  - fit on the displaced training set (frozen params)
  - predict suitability across the REGIONAL accessible network (bounded -> laptop-scale)
  - compare to the clean-benchmark surface

Metrics (as in m21, kept for continuity):
  - schoener_d(surf_a, surf_b)
  - warren_i(surf_a, surf_b)
  - range_area_change(surf_a, surf_b, threshold=0.5)

Contract:
    degradation_curve(entity, algo, d_grid, n_rep, rng) -> DataFrame[entity, algo, d, replicate, metric...]
"""

from __future__ import annotations

import numpy as np


def schoener_d(p, q):
    """Schoener's D between two suitability vectors over the same set of segments."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    return 1.0 - 0.5 * np.abs(p - q).sum()


def warren_i(p, q):
    """Warren's I between two suitability vectors over the same set of segments."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    h = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
    return 1.0 - 0.5 * h * h


def range_area_change(baseline, contaminated, threshold=0.5):
    """Fractional change in predicted suitable area at a binarisation threshold."""
    b = (np.asarray(baseline) >= threshold).sum()
    c = (np.asarray(contaminated) >= threshold).sum()
    if b == 0:
        return float("nan")
    return (c - b) / b


def degradation_curve(entity, algo, d_grid, n_rep, rng):
    raise NotImplementedError("Fit (frozen) -> predict on regional domain -> compare to benchmark.")
