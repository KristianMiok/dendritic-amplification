"""Build the clean High-Accuracy benchmark per entity (crayfish from WoC, fish from RivFishTIME).

Steps (to implement once assets are located):
  1. load occurrences per taxon
  2. snap to Hydrography90m (dendamp.snapping.snap)
  3. keep High-Accuracy / snap <= BENCHMARK_MAX_SNAP_M; dedupe to one record per segment
  4. attach per-segment predictors
  5. write data/processed/<entity>_benchmark.parquet
"""

from dendamp import config


def main():
    raise NotImplementedError("Wire config paths after locating assets, then implement steps 1-5.")


if __name__ == "__main__":
    main()
