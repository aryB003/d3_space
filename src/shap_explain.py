"""SHAP GradientExplainer on the FT logit for a single flagged file."""
from __future__ import annotations
from typing import List, Dict, Any

import numpy as np
import torch

from . import config, data
from .model import load_detector, FTLogitWrapper


TOP_K = 10


from functools import lru_cache


@lru_cache(maxsize=1)
def _background(variant: str = config.DEFAULT_DETECTOR,
                n: int = 8, seed: int = 0) -> np.ndarray:
    """Balanced reference distribution in `variant`'s input space, drawn from the
    deployment slice. The same rows are picked for either detector, so a
    difference between their attributions is the model and not the reference set.

    Background size is cheap — GradientExplainer draws one reference per
    gradient sample rather than forwarding the whole set, so this array costs
    ~35 MB per *sample*, not per reference. `nsamples` in explain() is the knob
    that actually governs peak memory; see the note there."""
    from .model import prepare
    d = config.ARTIFACTS_DIR
    X = np.load(d / config.DEMO_X, mmap_mode="r")
    y = np.load(d / config.DEMO_Y)
    rng = np.random.default_rng(seed)
    half = n // 2
    pick = np.concatenate([
        rng.choice(np.where(y == 1)[0], size=half, replace=False),
        rng.choice(np.where(y == 0)[0], size=n - half, replace=False),
    ])
    return prepare(np.asarray(X[np.sort(pick)]), variant)


# Expected gradients are averaged over independent passes rather than taken in
# one batch. shap forwards all `nsamples` through FT attention simultaneously —
# 501 tokens x 8 heads, ~35 MB per sample — so a single nsamples=50 call peaks
# near 2.7 GB and is killed on a 1 GB host. Averaging CHUNKS passes of
# NSAMPLES_PER_CHUNK is mathematically the same estimator (expected gradients is
# a mean over sampled paths) at a fraction of the peak.
CHUNKS = 5
NSAMPLES_PER_CHUNK = 4


def explain(x_scaled: np.ndarray, variant: str = config.DEFAULT_DETECTOR,
            device: str = "cpu") -> np.ndarray:
    """Return SHAP values of shape [500] for the single row `x_scaled`.

    Values are indexed in the detector's own column space, which for a permuted
    variant is not the stored order; top_k() maps them back before naming."""
    import shap  # imported lazily so the app starts fast
    ft = load_detector(variant, device)
    wrapper = FTLogitWrapper(ft).to(device)
    bg = torch.from_numpy(_background(variant)).to(device)
    x = torch.from_numpy(x_scaled).to(device)
    explainer = shap.GradientExplainer(wrapper, bg)

    acc = None
    for c in range(CHUNKS):
        vals = explainer.shap_values(x, nsamples=NSAMPLES_PER_CHUNK, rseed=c)
        if isinstance(vals, list):
            vals = vals[0]
        vals = np.asarray(vals).squeeze()
        if vals.ndim == 2 and vals.shape[0] == 1:
            vals = vals[0]
        acc = vals if acc is None else acc + vals
    return (acc / CHUNKS).astype(np.float32)


def top_k(shap_row: np.ndarray, x_scaled_row: np.ndarray, probability: float,
          k: int = TOP_K, variant: str = config.DEFAULT_DETECTOR) -> List[Dict[str, Any]]:
    """Build the per-file top-k structure used by the prompt template.

    `shap_row` is indexed in the detector's column space. Under a permuted
    variant, its column j is original column perm[j], so the index has to be
    mapped back before it is used to look up a feature name -- otherwise every
    attribution would be reported against the wrong feature."""
    from .model import _perm
    meta = data.feature_names()
    names  = meta["names"]
    groups = meta["groups"]
    to_orig = _perm() if config.DETECTORS[variant]["permute"] else np.arange(len(names))
    order  = np.argsort(-np.abs(shap_row))[:k]
    return [
        {
            "rank":      int(r + 1),
            "feature":   names[int(to_orig[int(j)])],
            "group":     groups[int(to_orig[int(j)])],
            "shap":      float(shap_row[int(j)]),
            "direction": "malicious" if shap_row[int(j)] > 0 else "benign",
            "feature_value_zscore": float(x_scaled_row[0, int(j)]),
        }
        for r, j in enumerate(order)
    ]


def prompt_entry(probability: float, top: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"probability": float(probability), "features": top}


def top_k_for_row(x_scaled_row: np.ndarray, k: int = TOP_K,
                  variant: str = config.DEFAULT_DETECTOR,
                  device: str = "cpu") -> List[Dict[str, Any]]:
    """Attribute one scaled [1, 500] row and return its top-k contributors."""
    vals = explain(x_scaled_row, variant=variant, device=device)
    return top_k(vals, x_scaled_row, probability=0.0, k=k, variant=variant)
