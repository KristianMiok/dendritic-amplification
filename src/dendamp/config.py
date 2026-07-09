"""Central configuration.

This file *is* the experimental design in code form. Filling the TODO slots after we
locate the regional Hydrography90m + predictor assets is the first real step.
"""

from pathlib import Path

# ----------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

# ----------------------------------------------------------------- located assets (TODO)
# Set these after running the "locate assets" step on the MacBook.
REGION_BASIN_IDS: list = []          # selected Hydrography90m basin id(s) for the study region
NETWORK_PATH: Path | None = None     # regional Hydrography90m segments (gpkg / parquet)
PREDICTOR_TABLE: Path | None = None  # per-segment GeoFRESH predictors (parquet), key = SEGMENT_ID_COL
SEGMENT_ID_COL = "segment_id"        # confirm the actual column name from the located table

# ------------------------------------------------------------------------ taxa panel
# entity_id -> metadata. Fill once the co-occurrence region is chosen.
TAXA: dict = {
    # "a_torrentium":    {"source": "woc",         "role": "crayfish_specialist"},
    # "p_leptodactylus": {"source": "woc",         "role": "crayfish_generalist"},
    # "fish_mobile":     {"source": "rivfishtime", "role": "fish_mobile"},
    # "fish_benthic":    {"source": "rivfishtime", "role": "fish_sedentary"},   # optional
    # "control_terr":    {"source": "gbif",        "role": "non_network_control"},  # optional
}

# ------------------------------------------------------------------ experiment grid
ALGORITHMS = ["rf", "xgboost"]              # Maxent intentionally dropped (justify in ms)
DISPLACEMENT_GRID_M = [100, 250, 500, 1000, 2500, 5000]  # d in metres
N_REPLICATES = 20
PREDICTION_DOMAIN = "regional"              # bounded — the lever that makes it laptop-scale
BENCHMARK_MAX_SNAP_M = 200                  # High-Accuracy benchmark threshold (as in m21)
SEED = 42

# ------------------------------------------------------------ topology bookkeeping
# predictor columns used to characterise what each displacement does (the mechanism).
# Confirm exact names from the located predictor table.
STRAHLER_COL = "strahler"          # stream order
FLOW_ACC_COL = "flow_accum"        # flow accumulation
CATCHMENT_COL = "subcatchment_id"  # to detect catchment crossings
