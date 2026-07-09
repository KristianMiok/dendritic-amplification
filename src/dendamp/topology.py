"""What each displacement does to network context — the amplification mechanism, logged.

For every displaced record we record whether the move crossed a topological boundary. These
are the predictors of Δenv: the paper's argument is that env shift is driven by *these* discrete
crossings, not by the metric distance alone.

Contract:
    displacement_effects(seg_from, seg_to, predictors) -> dict(
        delta_strahler = int,       # change in stream order
        crossed_catchment = bool,   # moved to a different subcatchment
        delta_flow_acc = float,     # change in flow accumulation
    )

Aggregate use: relate P(env jump) to fraction of moves that cross an order/catchment boundary,
stratified by local network topology (headwater vs mainstem) — ties to the m19 headwaters result.
"""

from __future__ import annotations

from . import config


def displacement_effects(seg_from, seg_to, predictors):
    raise NotImplementedError(
        "Look up STRAHLER_COL / FLOW_ACC_COL / CATCHMENT_COL for both segments; return the deltas."
    )
