"""Hard-Quota-free positive-only Soft-TopK objective for STAR-Q5."""

from __future__ import annotations

import torch

from sgg.modeling.roi_heads.ppg_budget_adaptation import soft_topk_occupancy


def released_score_anchor_gradient(
    student_scores: torch.Tensor,
    released_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ordinary edge-mean MSE anchor with no quota/group weighting."""

    if student_scores.shape != released_scores.shape or student_scores.ndim != 1:
        raise ValueError("Q5 anchor scores must be equally-sized vectors")
    if student_scores.numel() == 0:
        zero = student_scores.sum()
        return zero, student_scores
    difference = student_scores - released_scores
    return difference.square().mean(), 2.0 * difference / difference.numel()


def positive_group_coverage_score_gradient(
    *,
    student_scores: torch.Tensor,
    positive_indices: torch.Tensor,
    positive_group_ids: torch.Tensor,
    budget: int = 10_000,
    temperature: float = 5.0,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return ``-mean_g log(eps+r_g)`` and its exact score gradient.

    Only annotated pair indices enter group utilities.  Unannotated candidates
    receive no classification target; their gradient arises solely through the
    implicit Soft-TopK mass constraint.
    """

    if student_scores.ndim != 1:
        raise ValueError("student_scores must be one-dimensional")
    positives = positive_indices.long().to(student_scores.device)
    groups = positive_group_ids.long().to(student_scores.device)
    if positives.ndim != 1 or positives.shape != groups.shape:
        raise ValueError("positive indices/groups must be equally-sized vectors")
    if positives.numel() and (
        int(positives.min()) < 0 or int(positives.max()) >= student_scores.numel()
    ):
        raise IndexError("positive candidate index is out of range")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    scores = student_scores.detach().float().requires_grad_(True)
    occupancy, tau = soft_topk_occupancy(
        scores, budget=budget, temperature=temperature
    )
    if positives.numel() == 0:
        zero = scores.new_zeros(())
        return zero, torch.zeros_like(student_scores), {
            "positive_pairs": scores.new_zeros((), dtype=torch.long),
            "positive_groups": scores.new_zeros((), dtype=torch.long),
            "micro_positive_coverage": zero,
            "macro_group_coverage": zero,
            "minimum_group_coverage": zero,
            "soft_mass": occupancy.detach().sum(),
            "tau": tau.detach(),
        }

    unique, inverse, counts = torch.unique(
        groups, sorted=True, return_inverse=True, return_counts=True
    )
    sums = occupancy.new_zeros(unique.numel())
    sums.scatter_add_(0, inverse, occupancy[positives])
    recalls = sums / counts.to(sums).clamp_min(1)
    loss = -torch.log(recalls + float(epsilon)).mean()
    if loss.requires_grad:
        gradient = torch.autograd.grad(loss, scores, create_graph=False)[0].detach()
    else:
        # M <= K makes every occupancy exactly one.  Coverage is already
        # complete and invariant to scores, so its exact derivative is zero.
        gradient = torch.zeros_like(scores)
    return loss.detach().to(student_scores), gradient.to(student_scores), {
        "positive_pairs": student_scores.new_tensor(positives.numel(), dtype=torch.long),
        "positive_groups": student_scores.new_tensor(unique.numel(), dtype=torch.long),
        "micro_positive_coverage": occupancy[positives].mean().detach().to(student_scores),
        "macro_group_coverage": recalls.mean().detach().to(student_scores),
        "minimum_group_coverage": recalls.min().detach().to(student_scores),
        "soft_mass": occupancy.detach().sum().to(student_scores),
        "tau": tau.detach().to(student_scores),
    }
