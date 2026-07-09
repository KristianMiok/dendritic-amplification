# Dendritic amplification: a controlled-displacement reliability instrument for network SDMs

## One-line claim
A controlled *jittering* instrument that **measures** how positional error propagates
through dendritic river-network topology into (a) environmental predictor shifts `Δenv(d)`
and (b) SDM degradation — isolating **position as a causal factor**, unlike the confounded
real-data degradation reported in the crayfish robustness paper (m21).

## Why this is a separate paper (not m20, not m21)
- **Not m20** ("cleaning becomes biasing" — filtering as covariate shift). Different mechanism.
- **Not m21** (Lucian-first, descriptive robustness case study on real low-accuracy records).
  m21 *asserts* dendritic amplification; this paper is the instrument that *measures* it.
- The contribution is the **instrument + the measured amplification function**, generalised
  across a mobility × network-position gradient.

## Design (lean, laptop-runnable — the whole point)
- **Regional** prediction domain (1–2 European basins), NOT global. This is the single lever
  that moves the heavy Tier-3 prediction cost off VEGA and onto a MacBook.
- **Taxa panel** spanning mobility × network position:
  - 2 crayfish: one specialist (e.g. *A. torrentium*), one generalist (e.g. *P. leptodactylus*)
  - 1–2 fish from RivFishTIME (mobile; optional benthic sedentary for within-fish contrast)
  - optional 1 non-network taxon as a **negative control** (proves amplification is
    network-specific, not a generic jitter artifact)
- **Algorithms:** Random Forest + XGBoost. **Maxent dropped** (slow; importance extraction
  was already problematic in m21). Justify the drop explicitly in the manuscript.
- **Core measurement `Δenv(d)`:** model-free (fast) — the novel, cheap contribution.
- **Degradation(d):** Schoener's D, Warren's I, predicted range-area change on the bounded domain.

## Data
- Crayfish occurrences: World of Crayfish (WoC), High-Accuracy benchmark (snap ≤ 200 m).
- Fish occurrences: RivFishTIME (Comte et al. 2021, GEB 30:38–50; DOI 10.1111/geb.13210;
  data: iDiv repo DOI 10.25829/idiv.1873-10-4000). Stream-reach surveys → snappable to Hydrography90m.
- Network + predictors: Hydrography90m (Amatulli et al. 2022) + GeoFRESH per-segment table
  (Domisch et al. 2024). Re-snapping jittered points is a **table lookup** within the already
  extracted accessible network → no new GeoFRESH extraction needed for the regional subset.

## Status
Scaffold only. Core modules (`snapping`, `jitter`, `envshift`) are specified stubs — they
depend on the exact format of the located regional network + predictor table, so they are
implemented once those assets are confirmed.

## Layout
```
data/{raw,interim,processed}   # gitignored
src/dendamp/                   # package
scripts/                       # entry points (build benchmark, run envshift, run degradation, bench)
results/{figures,tables}
tests/
```
