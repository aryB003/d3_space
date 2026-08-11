"""Feature-name lookup for the 500 selected EMBER columns.

The test arrays themselves are loaded directly by the app from the deployment
subset (`ember_demo_*`). The earlier `load_testset()` helper read the full 200K
slice, which was stored in a different column ordering to the one the detector
was trained on — it has been removed along with that file to prevent the
mismatch recurring.
"""
from __future__ import annotations
import json
from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def feature_names() -> dict:
    """{"names": [...], "groups": [...]} indexed by model input column."""
    with open(config.ARTIFACTS_DIR / config.FEATURE_NAMES_JSON) as f:
        return json.load(f)
