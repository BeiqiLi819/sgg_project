"""Online Q5 Budget-Aware Top-K filtering for current task proposals."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch
from torch import nn

from sgg.modeling.roi_heads.ppg import PairProposalGenerator
from sgg.modeling.roi_heads.ppg_budget_adaptation import (
    semantic_group_ids,
    stable_descending_order,
)
from sgg.modeling.roi_heads.residual_ppg_scorer import SetConditionedPPGResidual
from sgg.modeling.roi_heads.set_context import (
    Q5_SET_CONTEXT_FEATURE_NAMES,
    exact_within_group_percentile,
    q5_set_context_features,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Q5PairProposalFilter(nn.Module):
    """Apply the frozen PredCls-trained Q5 scorer to current proposal objects.

    The residual network is proposal-task agnostic: it consumes only released
    PPG scores and analytic whole-image/class-pair statistics. SGCls/SGDet
    therefore rebuild their pair scores from their current proposal boxes and
    the label source selected by ``ROIRelationHead``; no PredCls endpoint cache
    is reused.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        rel = cfg["MODEL"]["ROI_RELATION_HEAD"]
        self.filter_method = "Q5"
        self.topk = int(rel.get("Q5_TOPK", rel.get("PPG_TOPK", 10000)))
        self.threshold = int(rel.get("Q5_PAIR_THRESHOLD", self.topk))
        self.chunk_size = max(1, int(rel.get("Q5_CHUNK_SIZE", 65536)))
        self.num_classes = int(cfg["MODEL"]["ROI_BOX_HEAD"]["NUM_CLASSES"])
        checkpoint_path = Path(
            rel.get(
                "Q5_CHECKPOINT",
                "outputs/star_q5_hard_quota_free_budgeted_proposal/"
                "training/budgeted_seed1029/residual_final.pth",
            )
        )
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        if metadata.get("mode") != "budgeted-coverage":
            raise ValueError("Q5 online filter requires budgeted-coverage checkpoint")
        if bool(metadata.get("hard_quota_training_signal", True)):
            raise ValueError("Q5 online filter refuses HardQuota-trained checkpoint")
        if list(payload["architecture"].get("feature_names", [])) != list(
            Q5_SET_CONTEXT_FEATURE_NAMES
        ):
            raise ValueError("Q5 online filter feature schema mismatch")
        self.residual = SetConditionedPPGResidual.from_checkpoint_payload(payload)
        for parameter in self.residual.parameters():
            parameter.requires_grad_(False)
        self.residual.eval()

        ppg_cfg = copy.deepcopy(cfg)
        ppg_rel = ppg_cfg["MODEL"]["ROI_RELATION_HEAD"]
        ppg_rel["TEST_FILTER_METHOD"] = "PPG"
        self.ppg = PairProposalGenerator(ppg_cfg)
        if not self.ppg.loaded:
            raise RuntimeError("Q5 online filter failed to load released PPG")
        expected_ppg = str(metadata.get("released_ppg_sha256", ""))
        if expected_ppg and _sha256(self.ppg.model_path) != expected_ppg:
            raise RuntimeError("Q5 residual and released PPG checkpoint hash mismatch")
        self.checkpoint_path = str(checkpoint_path)
        self.checkpoint_sha256 = _sha256(checkpoint_path)
        print(
            "[Q5] online task-proposal filter: "
            f"topk={self.topk}, threshold={self.threshold}, "
            f"checkpoint={checkpoint_path}",
            flush=True,
        )

    def train(self, mode: bool = True):
        super().train(False)
        self.residual.eval()
        self.ppg.eval()
        return self

    def filter_pairs(self, proposal, pair_idx: torch.Tensor) -> torch.Tensor:
        if pair_idx.ndim != 2 or pair_idx.size(1) != 2:
            raise ValueError("Q5 pair_idx must have shape [M,2]")
        if pair_idx.size(0) <= self.topk:
            return pair_idx
        labels = proposal.get_field("labels").long()
        if labels.numel() != len(proposal):
            raise ValueError("Q5 proposal labels do not align with boxes")
        score_parts = []
        for start in range(0, pair_idx.size(0), self.chunk_size):
            chunk = pair_idx[start : start + self.chunk_size]
            features = self.ppg.pair_features(proposal, chunk).float()
            losses = self.ppg.reconstruction_losses_from_features(
                features, inference=True
            )
            score_parts.append(losses.neg())
        base_scores = torch.cat(score_parts) if score_parts else pair_idx.new_zeros((0,)).float()
        groups = semantic_group_ids(pair_idx, labels, self.num_classes).long()
        group_counts = torch.bincount(
            groups, minlength=self.num_classes * self.num_classes
        )
        percentile = exact_within_group_percentile(base_scores, groups)
        active_groups = int((group_counts > 0).sum())
        q5_parts = []
        for start in range(0, pair_idx.size(0), self.chunk_size):
            end = min(start + self.chunk_size, pair_idx.size(0))
            context = q5_set_context_features(
                base_scores=base_scores[start:end],
                group_ids=groups[start:end],
                within_group_percentile=percentile[start:end],
                group_counts=group_counts,
                candidate_count=int(pair_idx.size(0)),
                active_group_count=active_groups,
                num_groups=self.num_classes * self.num_classes,
                budget=self.topk,
            )
            with torch.inference_mode():
                q5_parts.append(
                    base_scores[start:end] + self.residual(context)
                )
        scores = torch.cat(q5_parts) if q5_parts else base_scores
        order = stable_descending_order(scores, pair_idx)
        selected = pair_idx.index_select(0, order[: self.topk])
        proposal.add_field("q5_candidate_count", pair_idx.new_tensor([pair_idx.size(0)]))
        proposal.add_field("q5_selected_count", pair_idx.new_tensor([selected.size(0)]))
        return selected
