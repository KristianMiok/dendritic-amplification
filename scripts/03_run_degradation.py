"""Run degradation(d): fit (frozen) -> predict on regional domain -> compare to benchmark.

Writes results/tables/degradation.parquet -> [entity, algo, d, replicate, schoener_d, warren_i,
range_area_change].
"""

from dendamp import config


def main():
    raise NotImplementedError("Loop entities x algorithms x displacement grid; call degradation_curve.")


if __name__ == "__main__":
    main()
