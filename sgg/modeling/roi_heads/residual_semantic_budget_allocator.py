"""Preference-guided residual correction around a fixed semantic quota."""

from __future__ import annotations

import torch
from torch import nn

from .semantic_budget_allocator import largest_remainder_allocation


DECREASE, KEEP, INCREASE = 0, 1, 2


class ResidualPreferenceAllocator(nn.Module):
    """Small group preference classifier plus scene correction-mass head."""

    def __init__(
        self,
        *,
        num_classes: int,
        numeric_dim: int,
        semantic_dim: int,
        scene_dim: int,
        use_semantic_priors: bool,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.numeric_dim = int(numeric_dim)
        self.semantic_dim = int(semantic_dim)
        self.scene_dim = int(scene_dim)
        self.use_semantic_priors = bool(use_semantic_priors)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.class_embedding = nn.Embedding(self.num_classes, self.embedding_dim)
        group_input = 2 * self.embedding_dim + self.numeric_dim
        if self.use_semantic_priors:
            group_input += self.semantic_dim
        self.preference_mlp = nn.Sequential(
            nn.Linear(group_input, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 3),
        )
        intensity_hidden = max(self.hidden_dim // 2, 16)
        self.intensity_mlp = nn.Sequential(
            nn.Linear(self.scene_dim, intensity_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(intensity_hidden, 1),
        )

    def forward(
        self,
        subject_classes: torch.Tensor,
        object_classes: torch.Tensor,
        numeric_features: torch.Tensor,
        semantic_features: torch.Tensor,
        scene_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if numeric_features.ndim != 2 or numeric_features.size(1) != self.numeric_dim:
            raise ValueError("Residual allocator numeric feature shape differs")
        if semantic_features.ndim != 2 or semantic_features.size(1) != self.semantic_dim:
            raise ValueError("Residual allocator semantic feature shape differs")
        if scene_features.numel() != self.scene_dim:
            raise ValueError("Residual allocator scene feature shape differs")
        group_parts = [
            self.class_embedding(subject_classes.long()),
            self.class_embedding(object_classes.long()),
            numeric_features,
        ]
        if self.use_semantic_priors:
            group_parts.append(semantic_features)
        preference_logits = self.preference_mlp(torch.cat(group_parts, dim=1))
        normalized_mass = torch.sigmoid(
            self.intensity_mlp(scene_features.reshape(1, -1)).squeeze()
        )
        return preference_logits, normalized_mass


@torch.no_grad()
def budget_conserving_residual_transfer(
    quota_budgets: torch.Tensor,
    capacities: torch.Tensor,
    preference_logits: torch.Tensor,
    correction_mass: float,
    group_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Transfer predicted slots from DECREASE to INCREASE groups."""
    quota = quota_budgets.detach().long().cpu().reshape(-1)
    caps = capacities.detach().long().cpu().reshape(-1)
    logits = preference_logits.detach().float().cpu()
    ids = group_ids.detach().long().cpu().reshape(-1)
    if quota.numel() != caps.numel() or logits.shape != (quota.numel(), 3):
        raise ValueError("Residual-transfer tensors are not aligned")
    if (quota < 0).any() or (quota > caps).any():
        raise ValueError("Fixed quota is outside group capacity")
    requested = max(int(round(float(correction_mass))), 0)
    predicted = logits.argmax(dim=1)
    recipient_mask = (predicted == INCREASE) & (caps > quota)
    donor_mask = (predicted == DECREASE) & (quota > 0)
    recipient_indices = torch.nonzero(recipient_mask, as_tuple=False).flatten()
    donor_indices = torch.nonzero(donor_mask, as_tuple=False).flatten()
    recipient_capacity = caps[recipient_indices] - quota[recipient_indices]
    donor_capacity = quota[donor_indices]
    feasible = min(
        requested,
        int(recipient_capacity.sum()) if recipient_capacity.numel() else 0,
        int(donor_capacity.sum()) if donor_capacity.numel() else 0,
    )
    result = quota.clone()
    if feasible > 0:
        recipient_confidence = logits[recipient_indices, INCREASE] - torch.maximum(
            logits[recipient_indices, KEEP], logits[recipient_indices, DECREASE]
        )
        donor_confidence = logits[donor_indices, DECREASE] - torch.maximum(
            logits[donor_indices, KEEP], logits[donor_indices, INCREASE]
        )
        received = largest_remainder_allocation(
            recipient_confidence,
            recipient_capacity,
            feasible,
            ids[recipient_indices],
        )
        donated = largest_remainder_allocation(
            donor_confidence,
            donor_capacity,
            feasible,
            ids[donor_indices],
        )
        result[recipient_indices] += received
        result[donor_indices] -= donated
    if int(result.sum()) != int(quota.sum()) or (result < 0).any() or (result > caps).any():
        raise AssertionError("Residual transfer violated budget conservation or capacity")
    diagnostics = {
        "requested_correction_mass": requested,
        "transferred_slots": feasible,
        "recipient_groups": int(recipient_indices.numel()),
        "donor_groups": int(donor_indices.numel()),
        "predicted_decrease_groups": int((predicted == DECREASE).sum()),
        "predicted_keep_groups": int((predicted == KEEP).sum()),
        "predicted_increase_groups": int((predicted == INCREASE).sum()),
    }
    return result, diagnostics
