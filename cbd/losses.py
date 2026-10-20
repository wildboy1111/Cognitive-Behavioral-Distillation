"""InfoNCE alignment and private-feature orthogonality from the paper code."""

import torch
from torch import nn
from torch.nn import functional as F


class OrthogonalityLoss(nn.Module):
    """Penalize squared cosine similarity between private and semantic features."""

    def forward(self, private, semantic):
        return F.cosine_similarity(private, semantic, dim=-1, eps=1e-8).square().mean()


class MutualInfoLoss(nn.Module):
    """One-way InfoNCE with paired trials as positives (original temperature 0.5)."""

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, feature_a, feature_b):
        feature_a = F.normalize(feature_a, dim=1)
        feature_b = F.normalize(feature_b, dim=1)
        similarities = feature_a @ feature_b.T / self.temperature
        targets = torch.arange(feature_a.size(0), device=feature_a.device)
        return F.cross_entropy(similarities, targets)
