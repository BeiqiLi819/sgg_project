"""Cheap whole-image context features for STAR-Q2 proposal scoring.

All inputs are available before union/relation feature extraction.  The feature
definition is deliberately analytic and split-independent; it has no learned
normalization statistics and never consumes relation labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


SET_CONTEXT_FEATURE_NAMES = (
    "signed_log1p_base_score_div10",
    "log1p_group_size_div16",
    "group_candidate_share",
    "log1p_group_budget_pressure_div16",
    "within_group_base_rank_percentile",
    "log1p_candidate_pressure_div10",
    "active_group_fraction",
)


Q5_SET_CONTEXT_FEATURE_NAMES = (
    "signed_log1p_base_score_div10",
    "log1p_group_size_div16",
    "group_candidate_share",
    "log1p_group_uniform_budget_pressure_div16",
    "within_group_base_rank_percentile",
    "log1p_candidate_pressure_div10",
    "active_group_fraction",
)


@dataclass(frozen=True)
class ImageSetStatistics:
    candidate_count: int
    active_group_count: int
    group_counts: torch.Tensor
    quota_targets: torch.Tensor
    within_group_percentile: torch.Tensor


def exact_within_group_percentile(
    base_scores: torch.Tensor, group_ids: torch.Tensor
) -> torch.Tensor:
    """Return 0 for each group's best edge and 1 for its worst edge.

    Candidate order is the endpoint-canonical order used by the STAR cache, so
    two stable sorts reproduce the released score/end-point tie rule.
    """

    if base_scores.ndim != 1 or base_scores.shape != group_ids.shape:
        raise ValueError("base_scores/group_ids must be equally-sized vectors")
    count = int(base_scores.numel())
    if count == 0:
        return base_scores.float()
    score_order = torch.argsort(base_scores, descending=True, stable=True)
    grouped_order = score_order[
        torch.argsort(group_ids.long()[score_order], stable=True)
    ]
    sorted_groups = group_ids.long()[grouped_order]
    boundaries = torch.ones(count, dtype=torch.bool, device=base_scores.device)
    boundaries[1:] = sorted_groups[1:] != sorted_groups[:-1]
    starts = torch.nonzero(boundaries, as_tuple=False).flatten()
    lengths = torch.diff(
        torch.cat((starts, starts.new_tensor([count])))
    )
    repeated_starts = torch.repeat_interleave(starts, lengths)
    positions = torch.arange(count, device=base_scores.device) - repeated_starts
    denominators = torch.repeat_interleave((lengths - 1).clamp_min(1), lengths)
    sorted_percentile = positions.float() / denominators.float()
    percentile = torch.empty(count, dtype=torch.float32, device=base_scores.device)
    percentile[grouped_order] = sorted_percentile
    return percentile


def build_image_set_statistics(
    *,
    base_scores: torch.Tensor,
    group_ids: torch.Tensor,
    quota_targets: torch.Tensor,
    num_groups: int,
) -> ImageSetStatistics:
    if base_scores.ndim != 1 or base_scores.shape != group_ids.shape:
        raise ValueError("base_scores/group_ids must be equally-sized vectors")
    if quota_targets.shape != (int(num_groups),):
        raise ValueError("quota_targets must be dense over num_groups")
    groups = group_ids.long()
    counts = torch.bincount(groups, minlength=int(num_groups))
    return ImageSetStatistics(
        candidate_count=int(base_scores.numel()),
        active_group_count=int((counts > 0).sum()),
        group_counts=counts,
        quota_targets=quota_targets,
        within_group_percentile=exact_within_group_percentile(base_scores, groups),
    )


def set_context_features(
    *,
    base_scores: torch.Tensor,
    group_ids: torch.Tensor,
    within_group_percentile: torch.Tensor,
    group_counts: torch.Tensor,
    quota_targets: torch.Tensor,
    candidate_count: int,
    active_group_count: int,
    num_groups: int,
    budget: int,
) -> torch.Tensor:
    """Construct the frozen seven-dimensional STAR-Q2 feature vector."""

    if not (
        base_scores.ndim == 1
        and base_scores.shape == group_ids.shape == within_group_percentile.shape
    ):
        raise ValueError("edge context inputs must be equally-sized vectors")
    groups = group_ids.long()
    counts = group_counts.to(base_scores.device).float()[groups]
    quotas = quota_targets.to(base_scores.device).float()[groups]
    total = max(int(candidate_count), 1)
    effective_budget = max(min(int(budget), total), 1)
    signed_log_score = torch.sign(base_scores) * torch.log1p(base_scores.abs()) / 10.0
    features = torch.stack(
        (
            signed_log_score.clamp(-4.0, 4.0),
            (torch.log1p(counts) / 16.0).clamp(0.0, 2.0),
            (counts / float(total)).clamp(0.0, 1.0),
            (torch.log1p(counts / (quotas + 1.0)) / 16.0).clamp(0.0, 2.0),
            within_group_percentile.float().clamp(0.0, 1.0),
            base_scores.new_full(
                base_scores.shape,
                min(torch.log1p(torch.tensor(total / effective_budget)).item() / 10.0, 2.0),
            ),
            base_scores.new_full(
                base_scores.shape,
                float(active_group_count) / max(int(num_groups), 1),
            ),
        ),
        dim=1,
    )
    if not torch.isfinite(features).all():
        raise FloatingPointError("Non-finite STAR-Q2 set-context feature")
    return features


def q5_set_context_features(
    *,
    base_scores: torch.Tensor,
    group_ids: torch.Tensor,
    within_group_percentile: torch.Tensor,
    group_counts: torch.Tensor,
    candidate_count: int,
    active_group_count: int,
    num_groups: int,
    budget: int,
) -> torch.Tensor:
    """Hard-Quota-free seven-dimensional whole-image context.

    It preserves the Q4 scorer width and every target-free feature except the
    HardQuota-derived allocation pressure.  That coordinate is replaced by
    pressure relative to an analytic uniform share ``K / active_groups``.
    """

    if not (
        base_scores.ndim == 1
        and base_scores.shape == group_ids.shape == within_group_percentile.shape
    ):
        raise ValueError("Q5 edge context inputs must be equally-sized vectors")
    groups = group_ids.long()
    counts = group_counts.to(base_scores.device).float()[groups]
    total = max(int(candidate_count), 1)
    effective_budget = max(min(int(budget), total), 1)
    active = max(int(active_group_count), 1)
    uniform_group_budget = float(effective_budget) / float(active)
    signed_log_score = torch.sign(base_scores) * torch.log1p(base_scores.abs()) / 10.0
    features = torch.stack(
        (
            signed_log_score.clamp(-4.0, 4.0),
            (torch.log1p(counts) / 16.0).clamp(0.0, 2.0),
            (counts / float(total)).clamp(0.0, 1.0),
            (
                torch.log1p(counts / (uniform_group_budget + 1.0)) / 16.0
            ).clamp(0.0, 2.0),
            within_group_percentile.float().clamp(0.0, 1.0),
            base_scores.new_full(
                base_scores.shape,
                min(torch.log1p(torch.tensor(total / effective_budget)).item() / 10.0, 2.0),
            ),
            base_scores.new_full(
                base_scores.shape,
                float(active_group_count) / max(int(num_groups), 1),
            ),
        ),
        dim=1,
    )
    if not torch.isfinite(features).all():
        raise FloatingPointError("Non-finite STAR-Q5 set-context feature")
    return features
