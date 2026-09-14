from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


class RelationLossEvaluator:
    def __init__(self, cfg: dict):
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        self.task = str(cfg["MODEL"].get("TASK", "sgdet")).strip().lower()
        self.compute_object_refine_loss = self.task != "predcls"
        self.num_rel_classes = int(cfg["MODEL"]["ROI_RELATION_HEAD"]["NUM_CLASSES"])
        self.num_obj_classes = int(cfg["MODEL"]["ROI_BOX_HEAD"]["NUM_CLASSES"])
        self.loss_type = str(rel_cfg.get("PREDICATE_LOSS_TYPE", "ce")).lower()
        self.cb_beta = float(rel_cfg.get("PREDICATE_CLASS_BALANCED_BETA", 0.999))
        self.cb_fg_min_weight = float(
            rel_cfg.get("PREDICATE_CLASS_BALANCED_FG_MIN_WEIGHT", 0.0)
        )
        if self.cb_fg_min_weight < 0.0:
            raise ValueError(
                "PREDICATE_CLASS_BALANCED_FG_MIN_WEIGHT must be non-negative"
            )
        self.bg_loss_weight = float(rel_cfg.get("PREDICATE_BG_LOSS_WEIGHT", 1.0))
        self.logit_adjust_tau = float(rel_cfg.get("PREDICATE_LOGIT_ADJUST_TAU", 1.0))
        self.aux_logit_adjust_weight = float(rel_cfg.get("PREDICATE_AUX_LOGIT_ADJUST_WEIGHT", 0.0))
        self.aux_logit_adjust_tau = float(rel_cfg.get("PREDICATE_AUX_LOGIT_ADJUST_TAU", 0.5))
        self.focal_alpha = float(rel_cfg.get("PREDICATE_FOCAL_ALPHA", 0.1))
        self.focal_gamma = float(rel_cfg.get("PREDICATE_FOCAL_GAMMA", 2.0))
        self.focal_alpha_mode = str(
            rel_cfg.get(
                "PREDICATE_FOCAL_ALPHA_MODE", "foreground_background"
            )
        ).strip().lower()
        if self.focal_alpha_mode not in {
            "scalar",
            "foreground_background",
            "class_balanced",
        }:
            raise ValueError(
                "PREDICATE_FOCAL_ALPHA_MODE must be scalar, "
                "foreground_background or class_balanced, got "
                f"{self.focal_alpha_mode!r}"
            )
        self.focal_reduction = str(
            rel_cfg.get("PREDICATE_FOCAL_REDUCTION", "mean")
        ).strip().lower()
        if self.focal_reduction not in {"mean", "weighted_mean"}:
            raise ValueError(
                "PREDICATE_FOCAL_REDUCTION must be mean or weighted_mean, "
                f"got {self.focal_reduction!r}"
            )
        counts = rel_cfg.get("PREDICATE_COUNTS", [])
        self.predicate_counts = [int(v) for v in counts] if counts else []
        self.obj_loss_weight = float(rel_cfg.get("OBJECT_REFINE_LOSS_WEIGHT", 1.0))

    def _validate_predicate_counts(self) -> None:
        """Validate the train prior lazily so target-free eval can construct."""

        if len(self.predicate_counts) != self.num_rel_classes:
            raise ValueError(
                "class_balanced focal requires one train-only "
                "PREDICATE_COUNTS entry per relation class (including "
                f"background): expected {self.num_rel_classes}, got "
                f"{len(self.predicate_counts)}"
            )
        if any(value < 0 for value in self.predicate_counts):
            raise ValueError("PREDICATE_COUNTS cannot contain negative values")
        if sum(self.predicate_counts[1:]) <= 0:
            raise ValueError(
                "class_balanced focal requires at least one foreground "
                "relation in the training histogram"
            )

    def _counts_tensor(self, device: torch.device) -> torch.Tensor:
        counts = torch.ones((self.num_rel_classes,), dtype=torch.float32, device=device)
        if self.predicate_counts:
            n = min(len(self.predicate_counts), self.num_rel_classes)
            counts[:n] = torch.as_tensor(self.predicate_counts[:n], dtype=torch.float32, device=device)
        counts = counts.clamp(min=1.0)
        return counts

    def _class_balanced_weight(self, device: torch.device) -> torch.Tensor:
        counts = self._counts_tensor(device)
        if self.cb_beta <= 0.0 or self.cb_beta >= 1.0:
            weights = 1.0 / counts
        else:
            beta = torch.tensor(self.cb_beta, dtype=torch.float32, device=device)
            weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
        weights = weights / weights[1:].mean().clamp(min=1e-6) if weights.numel() > 1 else weights
        if weights.numel() > 1 and self.cb_fg_min_weight > 0.0:
            # Clamp after mean-one normalization.  This intentionally does
            # not renormalize again: doing so would lower frequent classes
            # below the requested floor and defeat the boost-only contract.
            weights[1:] = weights[1:].clamp_min(self.cb_fg_min_weight)
        weights[0] = self.bg_loss_weight
        return weights

    def _logit_adjusted(self, logits: torch.Tensor, tau: float) -> torch.Tensor:
        if tau == 0.0:
            return logits
        counts = self._counts_tensor(logits.device)
        prior = counts / counts.sum().clamp(min=1.0)
        return logits + float(tau) * prior.clamp(min=1e-12).log()

    def _apply_logit_adjustment(self, logits: torch.Tensor) -> torch.Tensor:
        if self.loss_type != "logit_adjusted":
            return logits
        return self._logit_adjusted(logits, self.logit_adjust_tau)

    def _focal_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        predicate_alpha: bool = True,
    ) -> torch.Tensor:
        """Equation (4) with an explicit multiclass alpha_t contract.

        HPL calls alpha a balancing factor.  Applying the same scalar to every
        sample cannot balance anything and only rescales the complete focal
        objective.  The default therefore uses the standard relation-SGG
        foreground/background extension: alpha for foreground targets and
        1-alpha for background.  ``scalar`` is retained only to reproduce the
        superseded implementation.  ``class_balanced`` derives one alpha per
        predicate from the effective number of train-only samples.  With
        ``weighted_mean`` the alpha values change relative class importance
        without silently shrinking the complete relation objective.
        """

        if not 0.0 <= self.focal_alpha <= 1.0 or self.focal_gamma < 0.0:
            raise ValueError("Focal alpha must be in [0,1] and gamma non-negative")
        log_probability = F.log_softmax(logits, dim=1)
        rows = torch.arange(labels.numel(), device=labels.device)
        log_p = log_probability[rows, labels]
        p = log_p.exp()
        alpha_mode = self.focal_alpha_mode if predicate_alpha else "foreground_background"
        if alpha_mode == "class_balanced":
            self._validate_predicate_counts()
            if logits.size(1) != self.num_rel_classes:
                raise ValueError(
                    "Predicate-frequency focal weights can only be applied "
                    "to relation logits"
                )
            class_weights = self._class_balanced_weight(logits.device).to(log_p.dtype)
            alpha_t = class_weights[labels]
        elif alpha_mode == "foreground_background":
            alpha_t = torch.where(
                labels > 0,
                log_p.new_full((), self.focal_alpha),
                log_p.new_full((), 1.0 - self.focal_alpha),
            )
        else:
            alpha_t = log_p.new_full(log_p.shape, self.focal_alpha)
        losses = -alpha_t * (1.0 - p).pow(self.focal_gamma) * log_p
        if self.focal_reduction == "weighted_mean":
            return losses.sum() / alpha_t.sum().clamp(min=torch.finfo(losses.dtype).eps)
        return losses.mean()

    def __call__(self, proposals: Sequence, rel_labels, relation_logits, refine_logits=None, cls_new=None):
        del cls_new
        device = None
        if isinstance(relation_logits, torch.Tensor):
            device = relation_logits.device
        elif relation_logits:
            device = relation_logits[0].device
        else:
            device = torch.device("cpu")

        if isinstance(relation_logits, list):
            relation_logits = torch.cat(relation_logits, dim=0) if relation_logits else torch.zeros((0, self.num_rel_classes), device=device)
        if isinstance(rel_labels, list):
            rel_labels = torch.cat(rel_labels, dim=0) if rel_labels else torch.zeros((0,), dtype=torch.long, device=device)

        if relation_logits.numel() == 0 or rel_labels.numel() == 0:
            loss_relation = torch.zeros((), device=device)
        else:
            labels = rel_labels.clamp(min=0, max=self.num_rel_classes - 1)
            main_logits = self._apply_logit_adjustment(relation_logits)
            weight = self._class_balanced_weight(relation_logits.device) if self.loss_type == "class_balanced" else None
            if self.loss_type in {"focal", "hpl_focal"}:
                loss_relation = self._focal_loss(main_logits, labels)
            else:
                loss_relation = F.cross_entropy(main_logits, labels, weight=weight)
            if self.aux_logit_adjust_weight > 0.0:
                aux_logits = self._logit_adjusted(relation_logits, self.aux_logit_adjust_tau)
                loss_relation = loss_relation + self.aux_logit_adjust_weight * F.cross_entropy(aux_logits, labels)

        loss_refine_obj = None
        # PredCls receives GT boxes and GT object labels by definition. Its
        # object output is protocol metadata/one-hot compatibility output, not
        # a trainable prediction target. Never construct object CE here even
        # if a legacy config leaves OBJECT_REFINE_LOSS_WEIGHT=1.
        if self.compute_object_refine_loss and refine_logits is not None:
            if isinstance(refine_logits, list):
                refine_logits = (
                    torch.cat(refine_logits, dim=0)
                    if refine_logits
                    else torch.zeros((0, self.num_obj_classes), device=device)
                )
            if refine_logits.numel() > 0:
                fg_labels = torch.cat(
                    [
                        proposal.get_field("gt_labels") if proposal.has_field("gt_labels") else proposal.get_field("labels")
                        for proposal in proposals
                    ],
                    dim=0,
                )
                fg_labels = fg_labels.clamp(min=0, max=refine_logits.size(1) - 1)
                if self.loss_type in {"focal", "hpl_focal"}:
                    # Predicate-frequency weights do not define an object-class
                    # distribution.  Keep the historical foreground/background
                    # object focal contract when the relation loss uses them.
                    loss_refine_obj = self._focal_loss(
                        refine_logits,
                        fg_labels,
                        predicate_alpha=False,
                    ) * self.obj_loss_weight
                else:
                    loss_refine_obj = F.cross_entropy(refine_logits, fg_labels) * self.obj_loss_weight

        return loss_relation, loss_refine_obj


def make_roi_relation_loss_evaluator(cfg: dict):
    return RelationLossEvaluator(cfg)
