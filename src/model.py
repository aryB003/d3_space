"""FT-SS detector — rtdl_revisiting_models paper defaults, 500 continuous inputs."""
from __future__ import annotations
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from rtdl_revisiting_models import FTTransformer

from . import config


def _build_ft() -> nn.Module:
    kw = FTTransformer.get_default_kwargs()
    return FTTransformer(n_cont_features=500, cat_cardinalities=[], d_out=1, **kw)


@lru_cache(maxsize=1)
def load_detector(device: str = "cpu") -> nn.Module:
    m = _build_ft()
    state = torch.load(config.ARTIFACTS_DIR / config.FT_WEIGHTS, map_location=device)
    m.load_state_dict(state)
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def score(x_scaled: np.ndarray, device: str = "cpu") -> tuple[float, float]:
    """Return (logit, probability) for a single [1, 500] scaled input."""
    m = load_detector(device)
    t = torch.from_numpy(x_scaled).to(device)
    logit = m(t, None).squeeze().item()
    prob = float(torch.sigmoid(torch.tensor(logit)))
    return float(logit), prob


@torch.no_grad()
def score_batch(X_scaled: np.ndarray, device: str = "cpu",
                batch: int = 16) -> np.ndarray:
    """Probabilities for [N, 500] scaled inputs. Batched because FT attention is
    quadratic in the 501 tokens — a large batch allocates enormous intermediates."""
    m = load_detector(device)
    out = []
    for i in range(0, len(X_scaled), batch):
        t = torch.from_numpy(X_scaled[i:i + batch]).to(device)
        out.append(torch.sigmoid(m(t, None).squeeze(-1)).float().cpu().numpy())
    return np.concatenate(out)


class FTLogitWrapper(nn.Module):
    """Squeezes FT's (x_cont, x_cat) signature into a Tensor -> Tensor callable
    for shap.GradientExplainer. Attributes in logit space — sign semantics are
    preserved and gradients are numerically better-behaved than through sigmoid."""
    def __init__(self, ft: nn.Module):
        super().__init__()
        self.ft = ft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ft(x, None).squeeze(-1).unsqueeze(-1)  # [B, 1]
