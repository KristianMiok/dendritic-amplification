"""Information Imbalance pilot.

Purpose: decide whether Information Imbalance (II) earns *backbone* status in the paper, using
a cheap empirical test rather than an a-priori commitment.

Two modes:

  --synthetic   Runs NOW (needs only dadapy). Generates a dendritic-like "serpentine" dataset
                with KNOWN ground truth (environment driven by network position; geography is a
                folded embedding so geographic distance predicts environment poorly) plus a
                terrestrial control (environment driven by geography). Confirms dadapy installs,
                the wrapper works, and the decision rule fires as designed.

  (default)     Real data. Reads per-entity benchmarks (geo, net, env) once config is wired to
                the located assets, and reports the same II table per entity.

Decision rule printed at the end:
  dendritic:  Delta(net->env) << Delta(geo->env)   -> network metric more informative
  control:    Delta(geo->env) low                   -> poor geo-predictiveness is network-specific
"""

from __future__ import annotations

import argparse

import numpy as np

from dendamp.imbalance import compare_geo_net_env


def _make_synthetic(seed: int = 0):
    rng = np.random.default_rng(seed)

    M, P = 30, 20
    N = M * P
    dx = 0.20 / P
    xs, ys = [], []
    for m in range(M):
        yy = np.linspace(0, 1, P)
        if m % 2 == 1:
            yy = yy[::-1]
        xs.append(np.full(P, m * dx))
        ys.append(yy)
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    t = np.linspace(0, 1, N)
    env = np.c_[t, np.sin(2 * np.pi * 4 * t)] + 0.01 * rng.standard_normal((N, 2))
    geo = np.c_[xs, ys] + 0.002 * rng.standard_normal((N, 2))
    net = t.reshape(-1, 1)

    gx, gy = rng.uniform(0, 1, N), rng.uniform(0, 1, N)
    geo_c = np.c_[gx, gy]
    env_c = np.c_[gx, gy] + 0.01 * rng.standard_normal((N, 2))
    net_c = geo_c[:, [0]]

    return (geo, net, env), (geo_c, net_c, env_c)


def _print_row(label, r):
    print(f"{label:12s}  Delta(geo->env)={r['geo->env']:.3f}   "
          f"Delta(net->env)={r['net->env']:.3f}   ratio geo/net={r['ratio_geo_over_net']:.2f}")


def run_synthetic():
    (dgeo, dnet, denv), (cgeo, cnet, cenv) = _make_synthetic()
    dend = compare_geo_net_env(dgeo, dnet, denv)
    ctrl = compare_geo_net_env(cgeo, cnet, cenv)
    print("== Information Imbalance pilot (synthetic ground truth) ==")
    _print_row("DENDRITIC", dend)
    _print_row("TERRESTRIAL", ctrl)
    print("---- decision rule ----")
    print(" network more informative than geo (dendritic)?",
          dend["net->env"] < dend["geo->env"])
    print(" geo predictive in terrestrial (control low)?  ",
          ctrl["geo->env"] < dend["geo->env"])
    print("")
    print("Interpretation: if the same pattern holds on the REAL benchmark, II earns backbone.")


def run_real():
    from dendamp import config
    if not config.TAXA:
        raise NotImplementedError(
            "Wire config.TAXA and build per-entity benchmarks first (scripts/01_build_benchmark.py). "
            "Each benchmark must expose geo (N,2), net (N,1), env (N,d). Then this will loop entities."
        )
    raise NotImplementedError("Real-data path: load benchmarks per entity, then compare_geo_net_env.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="run the synthetic ground-truth pilot")
    args = ap.parse_args()
    run_synthetic() if args.synthetic else run_real()
