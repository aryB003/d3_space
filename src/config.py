"""Runtime configuration — single source of truth for artifact paths."""
import os
from pathlib import Path

# On an HF Space the artifacts are pulled into /tmp/artifacts at startup.
# Locally, point ARTIFACTS_DIR at the folder containing the required model and
# data files.
ARTIFACTS_DIR = Path(os.environ.get(
    "ARTIFACTS_DIR",
    str(Path(__file__).resolve().parent.parent / "assets"),
))

# Space detection — `spaces` package is only importable inside HF Spaces.
IS_HF_SPACE = os.environ.get("SPACE_ID") is not None

# Files the app expects under ARTIFACTS_DIR.
FT_WEIGHTS         = "ft_ss_best.pt"
STANDARD_SCALER    = "standard_scaler.joblib"
SELECTED_INDICES   = "selected_feature_indices.npy"
FEATURE_NAMES_JSON = "feature_names.json"

# Two detectors are shipped. Same architecture; they differ in the scaler they
# were fitted against and in the column order they were trained on.
#
# FT-SS consumes the arrays as stored, which are sliced by
# selected_feature_indices.npy (sorted original index). FT-QT was trained on the
# Phase 3 export, whose columns are in descending Gini-importance order, and that
# ordering was never persisted -- it was recovered afterwards by matching the
# quantile transformer's per-column profiles. Feeding FT-QT the stored order
# scores it at chance (0.509 accuracy against 0.966), so `permute` is not
# cosmetic: it is the difference between a working detector and a broken one.
PERM_B2A = "perm_orderB_to_orderA.npy"

DETECTORS = {
    "FT-SS": {
        "weights": "ft_ss_best.pt",
        "scaler":  "standard_scaler.joblib",
        "permute": False,
        "scores":  "ember_demo_scores.npy",
        "blurb":   "StandardScaler. Carries the SHAP attributions and the cached reports.",
    },
    "FT-QT": {
        "weights": "ft_best.pt",
        "scaler":  "quantile_transformer.joblib",
        "permute": True,
        "scores":  "ember_demo_scores_qt.npy",
        "blurb":   "QuantileTransformer. Stronger on both datasets; the RQ1 detector.",
    },
}
DEFAULT_DETECTOR = "FT-SS"

# Deployment subset: 1,000 stratified files from the held-out test set,
# including every file that has a pre-generated report. Kept small so the
# app fits a 1 GB host; built by phase8_demo_assets.py.
DEMO_X       = "ember_demo_X500.npy"
DEMO_Y       = "ember_demo_y.npy"
DEMO_SRCIDX  = "ember_demo_srcidx.npy"   # index back into the full 200K test set
# Scores are precomputed offline. Scoring the whole slice at load time would
# allocate ~8 MB of attention per sample, which does not fit a 1 GB host.
DEMO_SCORES  = "ember_demo_scores.npy"

# Cached reports used as a fallback when no GPU is available locally.
REPORTS_CACHE = "reports_cache.json"

# Report generator. The repo path must stay exact — it is the HuggingFace
# address. GENERATOR_NAME is the display label used in the interface.
GENERATOR_ID   = "Qwen/Qwen2.5-7B-Instruct"
GENERATOR_NAME = "Qwen2.5-Instruct"
