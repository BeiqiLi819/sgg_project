"""Structured hard-quota preference primitives for STAR-Q3."""

from __future__ import annotations

import torch


def deterministic_preference_matching(
    *,
    base_scores: torch.Tensor,
    ordinary_indices: torch.Tensor,
    hard_quota_indices: torch.Tensor,
    boundary_pool_size: int = 2_048,
    maximum_pairs: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match promoted/displaced edges by their ordered released-score distributions.

    The highest-scoring promoted edges and lowest-scoring displaced edges form
    two boundary pools.  They are matched by the minimum-distance 1-D sorted
    bijection, after which the closest pairs are retained.  This makes the
    supervision explicitly boundary/score-close without a quadratic Cartesian
    preference set.
    """

    if base_scores.ndim != 1:
        raise ValueError("base_scores must be one-dimensional")
    ordinary = set(ordinary_indices.detach().long().cpu().tolist())
    hard = set(hard_quota_indices.detach().long().cpu().tolist())
    promoted = sorted(hard - ordinary)
    displaced = sorted(ordinary - hard)
    if len(promoted) != len(displaced):
        raise ValueError("Equal-cardinality Top-K sets must have equal promotion/displacement counts")
    if not promoted:
        empty = torch.zeros(0, dtype=torch.long, device=base_scores.device)
        return empty, empty
    promoted_tensor = torch.tensor(promoted, dtype=torch.long, device=base_scores.device)
    displaced_tensor = torch.tensor(displaced, dtype=torch.long, device=base_scores.device)
    pool = min(len(promoted), max(int(boundary_pool_size), 1))
    promoted_order = torch.argsort(base_scores[promoted_tensor], descending=True, stable=True)[:pool]
    # The displaced boundary is the lowest-scoring end of ordinary Top-K.
    displaced_boundary = torch.argsort(base_scores[displaced_tensor], descending=False, stable=True)[:pool]
    promoted_tensor = promoted_tensor[promoted_order]
    displaced_tensor = displaced_tensor[displaced_boundary]
    # Sorted same-direction pairing is the minimum-total-distance 1-D bijection.
    promoted_tensor = promoted_tensor[torch.argsort(base_scores[promoted_tensor], descending=True, stable=True)]
    displaced_tensor = displaced_tensor[torch.argsort(base_scores[displaced_tensor], descending=True, stable=True)]
    gaps = (base_scores[promoted_tensor] - base_scores[displaced_tensor]).abs()
    keep = torch.argsort(gaps, stable=True)[: min(pool, max(int(maximum_pairs), 1))]
    return promoted_tensor[keep], displaced_tensor[keep]


@torch.no_grad()
def anchor_score_loss_gradient(
    *,
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    group_ids: torch.Tensor,
    num_groups: int,
    strict_quota: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Q1/Q2 group-aware anchor loss and derivative w.r.t. scores."""

    if not (student_scores.shape == teacher_scores.shape == group_ids.shape):
        raise ValueError("anchor inputs must be equally-sized vectors")
    if student_scores.numel() == 0:
        zero = student_scores.sum()
        return zero, student_scores
    groups = group_ids.long()
    counts = torch.bincount(groups, minlength=int(num_groups)).to(student_scores)
    weights = counts.clamp(max=max(int(strict_quota), 0))
    denominator = weights.sum().clamp_min(1)
    edge_scale = weights[groups] / counts[groups].clamp_min(1) / denominator
    difference = student_scores - teacher_scores
    return (difference.square() * edge_scale).sum(), 2.0 * difference * edge_scale


@torch.no_grad()
def preference_loss_gradient(
    *,
    scores: torch.Tensor,
    promoted_indices: torch.Tensor,
    displaced_indices: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Pairwise logistic ranking loss and exact derivative over all edge scores."""

    promoted = promoted_indices.long().to(scores.device)
    displaced = displaced_indices.long().to(scores.device)
    if promoted.shape != displaced.shape or promoted.ndim != 1:
        raise ValueError("promoted/displaced indices must be equally-sized vectors")
    gradient = torch.zeros_like(scores)
    if promoted.numel() == 0:
        zero = scores.new_zeros(())
        return zero, gradient, {
            "pairs": scores.new_zeros((), dtype=torch.long),
            "accuracy": zero,
            "ties": zero,
            "mean_margin": zero,
        }
    temp = float(temperature)
    if temp <= 0:
        raise ValueError("preference temperature must be positive")
    margin = scores[promoted] - scores[displaced]
    loss = torch.nn.functional.softplus(-margin / temp).mean()
    derivative = -torch.sigmoid(-margin / temp) / (temp * promoted.numel())
    gradient.scatter_add_(0, promoted, derivative)
    gradient.scatter_add_(0, displaced, -derivative)
    return loss, gradient, {
        "pairs": scores.new_tensor(promoted.numel(), dtype=torch.long),
        "accuracy": (margin > 0).float().mean(),
        "ties": (margin == 0).float().mean(),
        "mean_margin": margin.mean(),
    }


@torch.no_grad()
def positive_preservation_loss_gradient(
    *,
    scores: torch.Tensor,
    protected_indices: torch.Tensor,
    comparator_indices: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Preserve teacher-retained annotated positives above teacher rejects.

    This is the same normalized logistic ranking primitive as Q3 preference
    distillation.  Only the source of the ordered pairs differs: the promoted
    side is ``GT ∩ HardQuota`` from STAR train, while the comparator side is a
    deterministic set of HardQuota-rejected candidates near the Top-K
    boundary.  Comparators are ordering references, not negative relation
    labels.
    """

    protected = protected_indices.long().to(scores.device)
    comparators = comparator_indices.long().to(scores.device)
    if protected.shape != comparators.shape or protected.ndim != 1:
        raise ValueError(
            "protected/comparator indices must be equally-sized vectors"
        )
    if protected.numel() and torch.isin(protected, comparators).any():
        raise ValueError("A protected positive cannot be its own comparator")
    loss, gradient, diagnostics = preference_loss_gradient(
        scores=scores,
        promoted_indices=protected,
        displaced_indices=comparators,
        temperature=temperature,
    )
    return loss, gradient, {
        "pairs": diagnostics["pairs"],
        "accuracy": diagnostics["accuracy"],
        "ties": diagnostics["ties"],
        "mean_margin": diagnostics["mean_margin"],
    }


@torch.no_grad()
def preference_selection_diagnostics(
    *,
    current_rank: torch.Tensor,
    selected_indices: torch.Tensor,
    ordinary_indices: torch.Tensor,
    hard_quota_indices: torch.Tensor,
    promoted_indices: torch.Tensor,
    displaced_indices: torch.Tensor,
) -> dict[str, float]:
    selected = set(selected_indices.detach().long().cpu().tolist())
    ordinary = set(ordinary_indices.detach().long().cpu().tolist())
    hard = set(hard_quota_indices.detach().long().cpu().tolist())
    promoted = promoted_indices.detach().long().cpu()
    displaced = displaced_indices.detach().long().cpu()
    rank = current_rank.detach().long().cpu()
    count = int(promoted.numel())
    preference_accuracy = float((rank[promoted] < rank[displaced]).float().mean()) if count else 1.0
    promotion_recall = sum(int(index) in selected for index in promoted.tolist()) / max(count, 1)
    displacement_recall = sum(int(index) not in selected for index in displaced.tolist()) / max(count, 1)
    overlap_released = len(selected & ordinary)
    overlap_hard = len(selected & hard)
    return {
        "preference_pairs": count,
        "preference_accuracy": preference_accuracy,
        "promotion_recall": promotion_recall,
        "displacement_recall": displacement_recall,
        "topk_replaced_from_released": len(selected - ordinary),
        "topk_jaccard_released": overlap_released / max(len(selected | ordinary), 1),
        "topk_jaccard_hard_quota": overlap_hard / max(len(selected | hard), 1),
    }
