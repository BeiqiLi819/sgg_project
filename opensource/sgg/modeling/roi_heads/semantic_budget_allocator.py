"""Minimal scene-adaptive semantic-group budget allocator."""

from __future__ import annotations

import torch
from torch import nn


class SemanticBudgetAllocator(nn.Module):
    """Predict one allocation logit per active semantic object-pair group."""

    def __init__(
        self,
        *,
        num_classes: int,
        numeric_dim: int,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.numeric_dim = int(numeric_dim)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.class_embedding = nn.Embedding(self.num_classes, self.embedding_dim)
        input_dim = 2 * self.embedding_dim + self.numeric_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        subject_classes: torch.Tensor,
        object_classes: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        if numeric_features.ndim != 2 or numeric_features.size(1) != self.numeric_dim:
            raise ValueError(
                f"Expected numeric features [G,{self.numeric_dim}], got "
                f"{tuple(numeric_features.shape)}"
            )
        subject = self.class_embedding(subject_classes.long())
        object_ = self.class_embedding(object_classes.long())
        return self.mlp(torch.cat((subject, object_, numeric_features), dim=1)).squeeze(1)


@torch.no_grad()
def largest_remainder_allocation(
    logits: torch.Tensor,
    capacities: torch.Tensor,
    budget: int,
    group_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Capacity-aware deterministic largest-remainder allocation.

    Saturated groups are removed and their excess continuous mass is
    redistributed according to the remaining softmax weights. Final integer
    ties use ascending semantic group ID.
    """
    scores = logits.detach().double().cpu().reshape(-1)
    caps = capacities.detach().long().cpu().reshape(-1)
    if scores.numel() != caps.numel():
        raise ValueError("Logit and capacity counts differ")
    if (caps < 0).any() or not torch.isfinite(scores).all():
        raise ValueError("Allocation inputs contain negative capacity or non-finite logits")
    total = min(max(int(budget), 0), int(caps.sum()))
    if scores.numel() == 0:
        if total != 0:
            raise ValueError("Non-zero budget requested for empty group set")
        return caps
    ids = (
        torch.arange(scores.numel(), dtype=torch.long)
        if group_ids is None
        else group_ids.detach().long().cpu().reshape(-1)
    )
    if ids.numel() != scores.numel() or torch.unique(ids).numel() != ids.numel():
        raise ValueError("Semantic group IDs must be unique and aligned")

    weights = torch.softmax(scores, dim=0)
    continuous = torch.zeros_like(weights)
    active = caps > 0
    remaining = float(total)
    while active.any() and remaining > 1e-12:
        active_weights = weights[active]
        weight_sum = float(active_weights.sum())
        if weight_sum <= 0:
            active_weights = torch.ones_like(active_weights)
            weight_sum = float(active_weights.sum())
        proposal = remaining * active_weights / weight_sum
        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        available = caps[active_indices].double() - continuous[active_indices]
        saturated = proposal >= available - 1e-12
        if not saturated.any():
            continuous[active_indices] += proposal
            remaining = 0.0
            break
        saturated_indices = active_indices[saturated]
        continuous[saturated_indices] += available[saturated]
        remaining -= float(available[saturated].sum())
        active[saturated_indices] = False
    continuous = torch.minimum(continuous, caps.double())
    allocation = torch.floor(continuous + 1e-12).long()
    leftover = total - int(allocation.sum())
    fractions = continuous - allocation.double()
    while leftover > 0:
        available = allocation < caps
        if not available.any():
            raise AssertionError("Capacity-aware rounding cannot satisfy exact budget")
        candidates = torch.nonzero(available, as_tuple=False).flatten().tolist()
        candidates.sort(key=lambda index: (-float(fractions[index]), int(ids[index])))
        progressed = 0
        for index in candidates:
            if leftover <= 0:
                break
            if allocation[index] < caps[index]:
                allocation[index] += 1
                fractions[index] = 0.0
                leftover -= 1
                progressed += 1
        if progressed == 0:
            raise AssertionError("Largest-remainder allocation made no progress")
    if int(allocation.sum()) != total or (allocation > caps).any():
        raise AssertionError("Largest-remainder allocation violates budget/capacity")
    return allocation
