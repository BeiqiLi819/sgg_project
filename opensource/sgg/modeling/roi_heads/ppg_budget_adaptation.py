"""Whole-image adaptation primitives for the released STAR PPG.

This module deliberately does not describe or reconstruct the unpublished PPG
training recipe.  It only exposes a trainable copy of the released inference
function and the STAR-Q1 image-level objectives defined by this project.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from sgg.modeling.roi_heads.ppg import PairProposalGenerator, _Autoencoder


@dataclass(frozen=True)
class WholeImageBudgetConfig:
    budget: int = 10_000
    strict_quota: int = 800
    relaxed_quota: int = 1_200
    temperature: float = 5.0
    budget_weight: float = 0.1
    score_chunk_size: int = 65_536


class ReleasedPPGStudent(nn.Module):
    """A trainable, exact architectural copy of a released PPG checkpoint."""

    def __init__(
        self,
        *,
        input_dim: int,
        encoding_dim: int = 25,
        hidden_dim1: int = 50,
        hidden_dim2: int = 50,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.encoding_dim = int(encoding_dim)
        self.hidden_dim1 = int(hidden_dim1)
        self.hidden_dim2 = int(hidden_dim2)
        self.model1 = _Autoencoder(
            self.input_dim, self.encoding_dim, self.hidden_dim1, self.hidden_dim2
        )
        self.model2 = _Autoencoder(
            self.input_dim, self.encoding_dim, self.hidden_dim1, self.hidden_dim2
        )

    @classmethod
    def from_released_generator(
        cls, generator: PairProposalGenerator
    ) -> "ReleasedPPGStudent":
        if not generator.loaded:
            raise RuntimeError("A loaded released PPG is required")
        model = cls(
            input_dim=generator.input_dim,
            encoding_dim=generator.encoding_dim,
            hidden_dim1=generator.hidden_dim1,
            hidden_dim2=generator.hidden_dim2,
        )
        model.model1.load_state_dict(copy.deepcopy(generator.model1.state_dict()))
        model.model2.load_state_dict(copy.deepcopy(generator.model2.state_dict()))
        return model

    @classmethod
    def from_checkpoint_payload(
        cls, payload: Mapping[str, object]
    ) -> "ReleasedPPGStudent":
        architecture = payload["architecture"]
        if not isinstance(architecture, Mapping):
            raise TypeError("Student checkpoint architecture must be a mapping")
        model = cls(
            input_dim=int(architecture["input_dim"]),
            encoding_dim=int(architecture["encoding_dim"]),
            hidden_dim1=int(architecture["hidden_dim1"]),
            hidden_dim2=int(architecture["hidden_dim2"]),
        )
        state = payload.get("student_state_dict")
        if not isinstance(state, Mapping):
            raise TypeError("Student checkpoint lacks student_state_dict")
        model.load_state_dict(state)
        return model

    def reconstruction_losses(self, features: torch.Tensor) -> torch.Tensor:
        inputs = features.float()
        out1 = self.model1(inputs)
        out2 = self.model2(out1)
        return 0.5 * (
            (out1 - inputs).square().mean(dim=1)
            + (out2 - inputs).square().mean(dim=1)
        )

    def scores(self, features: torch.Tensor, *, chunk_size: int = 65_536) -> torch.Tensor:
        """Return exact inference utility: negative reconstruction loss."""

        if features.ndim != 2 or features.size(1) != self.input_dim:
            raise ValueError(
                f"Expected features [M,{self.input_dim}], got {tuple(features.shape)}"
            )
        parts = []
        for start in range(0, int(features.size(0)), max(int(chunk_size), 1)):
            parts.append(self.reconstruction_losses(features[start : start + chunk_size]).neg())
        return torch.cat(parts, dim=0) if parts else features.new_zeros((0,))

    def architecture_manifest(self) -> dict[str, int]:
        return {
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim,
            "hidden_dim1": self.hidden_dim1,
            "hidden_dim2": self.hidden_dim2,
        }


def stable_descending_order(scores: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Sort by score descending, breaking exact ties by directed endpoints."""

    if scores.ndim != 1 or pairs.shape != (scores.numel(), 2):
        raise ValueError("scores/pairs shapes are inconsistent")
    if scores.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=scores.device)
    pairs_cpu = pairs.detach().long().cpu()
    width = max(int(pairs_cpu.max().item()) + 1, 1)
    canonical = torch.argsort(
        pairs_cpu[:, 0] * width + pairs_cpu[:, 1], stable=True
    )
    order = canonical[torch.argsort(scores.detach().cpu()[canonical], descending=True, stable=True)]
    return order.to(scores.device)


def semantic_group_ids(
    pairs: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    if pairs.ndim != 2 or pairs.size(1) != 2:
        raise ValueError("pairs must have shape [M,2]")
    labels = labels.long().to(pairs.device)
    return labels[pairs[:, 0]] * int(num_classes) + labels[pairs[:, 1]]


def soft_topk_occupancy(
    scores: torch.Tensor,
    *,
    budget: int,
    temperature: float,
    iterations: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable soft Top-K occupancy with an implicit boundary gradient.

    Bisection computes the exact forward threshold.  The straight expression
    attached afterwards supplies the derivative obtained by implicit
    differentiation of ``sum(sigmoid((s-tau)/T)) = K``.  Consequently a
    constant shift of every score leaves the occupancy unchanged in both the
    forward and backward passes.
    """

    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    count = int(scores.numel())
    effective = min(max(int(budget), 0), count)
    if count == 0:
        return scores, scores.new_zeros(())
    if effective == 0:
        return scores.new_zeros(scores.shape), scores.max().detach() + 1
    if effective == count:
        return scores.new_ones(scores.shape), scores.min().detach() - 1
    temp = float(temperature)
    if not temp > 0:
        raise ValueError("temperature must be positive")
    detached = scores.detach().float()
    margin = max(32.0 * temp, 1.0e-6)
    low = detached.min() - margin
    high = detached.max() + margin
    target = float(effective)
    for _ in range(int(iterations)):
        midpoint = (low + high) * 0.5
        mass = torch.sigmoid((detached - midpoint) / temp).sum()
        # Occupancy decreases as tau grows.
        low, high = torch.where(mass > target, midpoint, low), torch.where(
            mass > target, high, midpoint
        )
    tau_value = (low + high) * 0.5
    initial = torch.sigmoid((detached - tau_value) / temp)
    slope = (initial * (1.0 - initial)).clamp_min(0)
    weights = slope / slope.sum().clamp_min(torch.finfo(slope.dtype).eps)
    tau = tau_value.to(scores.dtype) + (
        weights.to(scores.dtype) * (scores - scores.detach())
    ).sum()
    occupancy = torch.sigmoid((scores - tau) / temp)
    return occupancy, tau


def group_aware_anchor_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    strict_quota: int,
) -> torch.Tensor:
    """Teacher score MSE whose group mass is capped by the historical quota."""

    if not (
        student_scores.shape == teacher_scores.shape == group_ids.shape
        and student_scores.ndim == 1
    ):
        raise ValueError("anchor inputs must be equally sized vectors")
    if student_scores.numel() == 0:
        return student_scores.sum() * 0
    unique, inverse, counts = torch.unique(
        group_ids.long(), sorted=True, return_inverse=True, return_counts=True
    )
    del unique
    per_edge = (student_scores - teacher_scores).square()
    sums = per_edge.new_zeros((counts.numel(),))
    sums.scatter_add_(0, inverse, per_edge)
    means = sums / counts.to(sums.dtype).clamp_min(1)
    weights = counts.clamp(max=max(int(strict_quota), 0)).to(means.dtype)
    return (means * weights).sum() / weights.sum().clamp_min(1)


def semantic_budget_loss(
    occupancy: torch.Tensor,
    group_ids: torch.Tensor,
    quota_targets: torch.Tensor,
    *,
    num_groups: int,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-sided squared excess against a hard-quota allocation target."""

    if occupancy.shape != group_ids.shape:
        raise ValueError("occupancy and group_ids must have the same shape")
    if quota_targets.shape != (int(num_groups),):
        raise ValueError("quota_targets must be dense over num_groups")
    masses = occupancy.new_zeros((int(num_groups),))
    if occupancy.numel():
        masses.scatter_add_(0, group_ids.long(), occupancy)
    overflow = torch.relu(masses - quota_targets.to(masses))
    loss = overflow.square().sum() / max(int(budget), 1)
    return loss, masses, overflow


def whole_image_objective(
    model: ReleasedPPGStudent,
    *,
    features: torch.Tensor,
    teacher_scores: torch.Tensor,
    group_ids: torch.Tensor,
    quota_targets: torch.Tensor,
    num_groups: int,
    config: WholeImageBudgetConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    student_scores = model.scores(features, chunk_size=config.score_chunk_size)
    anchor = group_aware_anchor_loss(
        student_scores,
        teacher_scores,
        group_ids,
        strict_quota=config.strict_quota,
    )
    occupancy, tau = soft_topk_occupancy(
        student_scores,
        budget=config.budget,
        temperature=config.temperature,
    )
    budget_loss, masses, overflow = semantic_budget_loss(
        occupancy,
        group_ids,
        quota_targets,
        num_groups=num_groups,
        budget=config.budget,
    )
    total = anchor + float(config.budget_weight) * budget_loss
    return total, {
        "student_scores": student_scores,
        "anchor_loss": anchor,
        "budget_loss": budget_loss,
        "soft_occupancy": occupancy,
        "soft_mass": occupancy.sum(),
        "tau": tau,
        "group_masses": masses,
        "overflow": overflow,
    }


@torch.no_grad()
def whole_image_score_gradient(
    *,
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    group_ids: torch.Tensor,
    quota_targets: torch.Tensor,
    num_groups: int,
    config: WholeImageBudgetConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate the objective and its exact derivative w.r.t. edge scores.

    This supports very large images without retaining millions of MLP
    activations.  Training first scores every shard without gradients, computes
    this complete-image derivative, then recomputes each shard and applies the
    corresponding score-gradient slice.  The resulting parameter gradient is
    mathematically identical to the materialized objective.
    """

    if not (
        student_scores.shape == teacher_scores.shape == group_ids.shape
        and student_scores.ndim == 1
    ):
        raise ValueError("score-gradient inputs must be equally sized vectors")
    group_ids = group_ids.long()
    counts = torch.bincount(group_ids, minlength=int(num_groups)).to(student_scores)
    weights = counts.clamp(max=max(int(config.strict_quota), 0))
    denominator = weights.sum().clamp_min(1)
    edge_scale = weights[group_ids] / counts[group_ids].clamp_min(1) / denominator
    difference = student_scores - teacher_scores
    anchor_loss = (difference.square() * edge_scale).sum()
    gradient = 2.0 * difference * edge_scale

    occupancy, tau = soft_topk_occupancy(
        student_scores, budget=config.budget, temperature=config.temperature
    )
    budget_loss, masses, overflow = semantic_budget_loss(
        occupancy,
        group_ids,
        quota_targets,
        num_groups=num_groups,
        budget=config.budget,
    )
    slope = occupancy * (1.0 - occupancy)
    if slope.sum() > 0:
        occupancy_cost = (2.0 / max(int(config.budget), 1)) * overflow[group_ids]
        centered_cost = occupancy_cost - (
            occupancy_cost * slope
        ).sum() / slope.sum().clamp_min(torch.finfo(slope.dtype).eps)
        budget_gradient = slope * centered_cost / float(config.temperature)
        gradient = gradient + float(config.budget_weight) * budget_gradient
    else:
        budget_gradient = torch.zeros_like(gradient)
    total = anchor_loss + float(config.budget_weight) * budget_loss
    return gradient, {
        "total_loss": total,
        "anchor_loss": anchor_loss,
        "budget_loss": budget_loss,
        "soft_mass": occupancy.sum(),
        "tau": tau,
        "group_masses": masses,
        "overflow": overflow,
        "raw_budget_score_gradient_norm": budget_gradient.norm(),
        "total_score_gradient_norm": gradient.norm(),
    }


def save_student_checkpoint(
    path: str | Path,
    model: ReleasedPPGStudent,
    *,
    metadata: Mapping[str, object],
) -> None:
    payload = {
        "schema_version": 1,
        "kind": "independent_adaptation_of_released_ppg",
        "architecture": model.architecture_manifest(),
        "student_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "metadata": dict(metadata),
    }
    torch.save(payload, Path(path))
