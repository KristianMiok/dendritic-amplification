"""Run the model-free core: Delta_env(d) for every entity. Cheap; the paper's novelty.

Writes results/tables/envshift.parquet  ->  [entity, d, replicate, record_id,
delta_env, delta_strahler, crossed_catchment, delta_flow_acc].
"""

from dendamp import config


def main():
    raise NotImplementedError("Load benchmarks + predictors + network, call dendamp.envshift.delta_env.")


if __name__ == "__main__":
    main()
