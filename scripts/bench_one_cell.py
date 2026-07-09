"""Time ONE cell so we can estimate the whole grid before committing compute.

A "cell" = one entity x one algorithm x one displacement level x one replicate, plus one
Δenv computation. Run this once config.py points at a small located entity, then paste the
printed timings back — that turns the grid-size (entities x algos x d-levels x replicates)
into a concrete wall-clock number.

Until assets are wired, this prints what it needs.
"""

import time

from dendamp import config


def main():
    missing = [k for k in ("NETWORK_PATH", "PREDICTOR_TABLE") if getattr(config, k) is None]
    if missing or not config.TAXA:
        print("Not wired yet. Fill config.py:", ", ".join(missing) or "TAXA is empty")
        print("Then this will time: load -> snap/jitter one cell -> one RF fit -> one Δenv cell.")
        return

    # --- template for the real timing once wired ---
    t0 = time.perf_counter()
    # load_predictors / load_network / one benchmark entity ...
    t_load = time.perf_counter() - t0

    print(f"load:        {t_load:8.3f} s")
    print("fit_rf:      <implement>")
    print("delta_env:   <implement>")
    print("predict_dom: <implement>  # regional domain prediction (the heavy tail)")


if __name__ == "__main__":
    main()
