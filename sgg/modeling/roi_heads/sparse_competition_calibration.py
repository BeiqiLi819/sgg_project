"""Group-distilled sparse competition calibration for frozen RPCM logits.

The module borrows the useful boundary of BAL/SBP -- freeze a strong SGG
model and predict a sample-specific logit bias -- while replacing the
undisclosed adversarial target learning with a deterministic, auditable
competition target.  For a foreground sample only the ground-truth class and
the strongest competing class are corrected.  Class-balanced reductions and
specialist-to-universal distillation prevent frequent predicates from
dominating the post-training objective.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


GROUP_MANIFEST_SCHEMA = "sparse-competition-group-manifest-v1"
CHECKPOINT_SCHEMA = "group-distilled-sparse-competition-calibrator-v1"


def _canonical_hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def train_frequency_bias(
    counts: torch.Tensor,
    *,
    exponent: float = 0.1,
    epsilon: float = 0.001,
) -> torch.Tensor:
    """BAL-style train-frequency prior, used as input rather than logit shift."""

    counts = counts.float().clamp_min(0.0)
    if counts.ndim != 1 or counts.numel() < 2:
        raise ValueError("predicate counts must be a 1-D class vector")
    if exponent <= 0.0 or epsilon <= 0.0:
        raise ValueError("frequency exponent and epsilon must be positive")
    proportions = counts / counts.sum().clamp_min(1.0)
    powered = proportions.clamp_min(1e-12).pow(float(exponent))
    return -(powered / powered.sum().clamp_min(1e-12) + float(epsilon)).log()


def build_group_manifest(
    class_feature_centroids: torch.Tensor,
    class_counts: Sequence[int],
    *,
    class_feature_counts: Sequence[int],
    relation_names: Sequence[str],
    union_dim: int,
    num_groups: int = 3,
    seed: int = 1029,
    source: Mapping[str, object] | None = None,
    background_class: int = 0,
) -> dict[str, object]:
    """Create deterministic TRAIN-only feature groups and prior statistics."""

    if class_feature_centroids.ndim != 2:
        raise ValueError("class_feature_centroids must be [classes, dim]")
    counts = [int(value) for value in class_counts]
    feature_counts = [int(value) for value in class_feature_counts]
    num_classes = int(class_feature_centroids.size(0))
    if len(counts) != num_classes or len(feature_counts) != num_classes:
        raise ValueError("class count vectors do not match feature centroids")
    if len(relation_names) != num_classes:
        raise ValueError("relation_names must include background and all predicates")
    if any(value < 0 for value in counts + feature_counts):
        raise ValueError("class counts must be non-negative")
    if int(union_dim) <= 0 or int(num_groups) <= 0:
        raise ValueError("union_dim and num_groups must be positive")
    valid = [
        class_id
        for class_id in range(num_classes)
        if class_id != int(background_class)
        and counts[class_id] > 0
        and feature_counts[class_id] > 0
    ]
    if len(valid) < int(num_groups):
        raise ValueError("not enough foreground predicates for requested groups")
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise RuntimeError("group construction requires scikit-learn") from exc

    assignments = KMeans(
        n_clusters=int(num_groups),
        n_init=10,
        random_state=int(seed),
    ).fit_predict(class_feature_centroids[valid].float().cpu().numpy())
    raw_groups: list[list[int]] = [[] for _ in range(int(num_groups))]
    for class_id, group_id in zip(valid, assignments.tolist()):
        raw_groups[int(group_id)].append(int(class_id))
    groups = sorted(
        (sorted(group) for group in raw_groups),
        key=lambda group: (len(group), group[0]),
    )
    class_to_group = [-1] * num_classes
    for group_id, group in enumerate(groups):
        for class_id in group:
            class_to_group[class_id] = int(group_id)
    for class_id in range(num_classes):
        if class_id != int(background_class) and class_to_group[class_id] < 0:
            class_to_group[class_id] = int(num_groups) - 1

    foreground_ids = [
        class_id
        for class_id in range(num_classes)
        if class_id != int(background_class) and counts[class_id] > 0
    ]
    foreground_prior = train_frequency_bias(
        torch.as_tensor([counts[class_id] for class_id in foreground_ids])
    )
    global_bias = [0.0] * num_classes
    for class_id, bias in zip(foreground_ids, foreground_prior.tolist()):
        global_bias[class_id] = float(bias)

    proposal_coverage = [
        float(feature_counts[class_id]) / float(max(counts[class_id], 1))
        for class_id in range(num_classes)
    ]
    payload: dict[str, object] = {
        "schema": GROUP_MANIFEST_SCHEMA,
        "split": "train",
        "validation_or_test_statistics_used": False,
        "seed": int(seed),
        "num_groups": int(num_groups),
        "num_rel_classes": num_classes,
        "relation_dim": int(class_feature_centroids.size(1)),
        "union_dim": int(union_dim),
        "background_class": int(background_class),
        "groups": groups,
        "class_to_group": class_to_group,
        "class_counts": counts,
        "class_feature_counts": feature_counts,
        "train_proposal_coverage": proposal_coverage,
        "relation_names": list(map(str, relation_names)),
        "global_bias": global_bias,
        "global_bias_exponent": 0.1,
        "global_bias_epsilon": 0.001,
        "grouping": "kmeans_raw_train_class_relation_centroids",
        "source": dict(source or {}),
    }
    payload["content_hash"] = _canonical_hash(payload)
    return payload


def save_group_manifest(payload: Mapping[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_group_manifest(
    path: str | Path,
    *,
    num_rel_classes: int | None = None,
) -> dict[str, object]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != GROUP_MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported competition-calibration manifest: {path}")
    expected = str(payload.get("content_hash", ""))
    unhashed = dict(payload)
    unhashed.pop("content_hash", None)
    if expected != _canonical_hash(unhashed):
        raise RuntimeError(f"Competition-calibration manifest hash mismatch: {path}")
    if payload.get("split") != "train":
        raise ValueError("competition-calibration grouping must be TRAIN-only")
    if payload.get("validation_or_test_statistics_used") is not False:
        raise ValueError("competition-calibration manifest is not leakage-safe")
    if num_rel_classes is not None and int(payload["num_rel_classes"]) != int(
        num_rel_classes
    ):
        raise ValueError("competition-calibration class count mismatch")
    return payload


def sparse_competition_targets(
    original_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    target_margin: float = 0.2,
    max_correction: float = 2.0,
    background_class: int = 0,
) -> dict[str, torch.Tensor]:
    """Construct the minimum symmetric GT-vs-strongest-competitor correction."""

    if original_logits.ndim != 2:
        raise ValueError("original_logits must be [samples, classes]")
    labels = labels.long()
    if labels.shape != (original_logits.size(0),):
        raise ValueError("labels must have shape [samples]")
    if target_margin < 0.0 or max_correction <= 0.0:
        raise ValueError("target margin must be non-negative and correction positive")
    valid = (
        (labels >= 0)
        & (labels < original_logits.size(1))
        & labels.ne(int(background_class))
    )
    safe_labels = labels.clamp(min=0, max=original_logits.size(1) - 1)
    masked = original_logits.detach().clone()
    if masked.numel() > 0:
        masked.scatter_(1, safe_labels.unsqueeze(1), float("-inf"))
    competitor = masked.argmax(dim=1)
    rows = torch.arange(labels.numel(), device=labels.device)
    gt_logits = original_logits.detach()[rows, safe_labels]
    competitor_logits = original_logits.detach()[rows, competitor]
    base_margin = gt_logits - competitor_logits
    required = (float(target_margin) - base_margin).clamp(
        min=0.0, max=float(max_correction)
    )
    required = required * valid.to(dtype=required.dtype)
    active = valid & required.gt(0.0)
    target = original_logits.new_zeros(original_logits.shape)
    if active.any():
        active_rows = rows[active]
        half = required[active] * 0.5
        target[active_rows, safe_labels[active]] = half
        target[active_rows, competitor[active]] = -half
    predicted = original_logits.detach().argmax(dim=1)
    stable = valid & predicted.eq(labels) & base_margin.ge(float(target_margin))
    return {
        "target_bias": target,
        "competitor": competitor,
        "base_margin": base_margin,
        "required_correction": required,
        "valid": valid,
        "active": active,
        "stable": stable,
    }


def _macro_class_mean(
    values: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if values.ndim != 1 or labels.shape != values.shape or mask.shape != values.shape:
        raise ValueError("macro reduction expects aligned 1-D tensors")
    if not mask.any():
        return values.sum() * 0.0
    terms = [
        values[mask & labels.eq(class_id)].mean()
        for class_id in labels[mask].unique(sorted=True)
    ]
    return torch.stack(terms).mean()


class _FeatureEncoder(nn.Module):
    def __init__(
        self,
        relation_dim: int,
        union_dim: int,
        num_classes: int,
        hidden_dim: int,
        projection_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.relation = nn.Sequential(
            nn.LayerNorm(relation_dim),
            nn.Linear(relation_dim, projection_dim),
            nn.GELU(),
        )
        self.union = nn.Sequential(
            nn.LayerNorm(union_dim),
            nn.Linear(union_dim, projection_dim),
            nn.GELU(),
        )
        self.logits = nn.Sequential(
            nn.LayerNorm(num_classes * 2),
            nn.Linear(num_classes * 2, projection_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(projection_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        relation_features: torch.Tensor,
        union_features: torch.Tensor,
        original_logits: torch.Tensor,
        global_bias: torch.Tensor,
    ) -> torch.Tensor:
        prior = global_bias.unsqueeze(0).expand(original_logits.size(0), -1)
        return self.fusion(
            torch.cat(
                [
                    self.relation(relation_features),
                    self.union(union_features),
                    self.logits(torch.cat([original_logits, prior], dim=1)),
                ],
                dim=1,
            )
        )


class _ResidualHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        # A non-zero outer scale is unnecessary: direct sparse-bias
        # supervision gives this zero layer gradients on the first step while
        # preserving exact base logits at construction.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor, max_bias: float) -> torch.Tensor:
        return float(max_bias) * torch.tanh(self.network(features) / 2.0)


class GroupDistilledCompetitionCalibrator(nn.Module):
    """Three grouped specialists distilled into one inference calibrator."""

    def __init__(
        self,
        manifest: Mapping[str, object],
        *,
        hidden_dim: int = 512,
        projection_dim: int = 256,
        dropout: float = 0.1,
        max_bias: float = 2.0,
        target_margin: float = 0.2,
        inference_projection: str = "full",
        inference_group_scales: Sequence[float] | None = None,
        inference_head: str = "universal",
    ):
        super().__init__()
        self.manifest_hash = str(manifest["content_hash"])
        self.num_classes = int(manifest["num_rel_classes"])
        self.num_groups = int(manifest["num_groups"])
        self.relation_dim = int(manifest["relation_dim"])
        self.union_dim = int(manifest["union_dim"])
        self.background_class = int(manifest.get("background_class", 0))
        self.max_bias = float(max_bias)
        self.target_margin = float(target_margin)
        self.inference_projection = str(inference_projection).strip().lower()
        if self.inference_projection not in {"full", "top1_positive"}:
            raise ValueError(
                "inference_projection must be 'full' or 'top1_positive'"
            )
        self.inference_head = str(inference_head).strip().lower()
        if self.inference_head not in {"universal", "routed_specialist"}:
            raise ValueError(
                "inference_head must be 'universal' or 'routed_specialist'"
            )
        self.inference_group_scales = tuple(
            float(value)
            for value in (
                inference_group_scales
                if inference_group_scales is not None
                else [1.0] * self.num_groups
            )
        )
        if len(self.inference_group_scales) != self.num_groups or any(
            value < 0.0 for value in self.inference_group_scales
        ):
            raise ValueError(
                "inference_group_scales must contain one non-negative value "
                "per specialist group"
            )
        if hidden_dim <= 0 or projection_dim <= 0 or self.max_bias <= 0.0:
            raise ValueError("calibrator dimensions and max_bias must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("calibrator dropout must be in [0, 1)")
        self.register_buffer(
            "global_bias",
            torch.as_tensor(manifest["global_bias"], dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "class_to_group",
            torch.as_tensor(manifest["class_to_group"], dtype=torch.long),
            persistent=True,
        )
        self.encoder = _FeatureEncoder(
            self.relation_dim,
            self.union_dim,
            self.num_classes,
            int(hidden_dim),
            int(projection_dim),
            float(dropout),
        )
        self.specialists = nn.ModuleList(
            [
                _ResidualHead(int(hidden_dim), self.num_classes, float(dropout))
                for _ in range(self.num_groups)
            ]
        )
        self.universal = _ResidualHead(
            int(hidden_dim), self.num_classes, float(dropout)
        )

    def encode(
        self,
        relation_features: torch.Tensor,
        union_features: torch.Tensor,
        original_logits: torch.Tensor,
    ) -> torch.Tensor:
        if relation_features.shape != (
            original_logits.size(0),
            self.relation_dim,
        ):
            raise ValueError("relation feature shape does not match manifest")
        if union_features.shape != (original_logits.size(0), self.union_dim):
            raise ValueError("union feature shape does not match manifest")
        if original_logits.size(1) != self.num_classes:
            raise ValueError("logit class count does not match manifest")
        return self.encoder(
            relation_features,
            union_features,
            original_logits.detach(),
            self.global_bias,
        )

    def predict_bias(
        self,
        relation_features: torch.Tensor,
        union_features: torch.Tensor,
        original_logits: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encode(relation_features, union_features, original_logits)
        universal_bias = self.universal(encoded, self.max_bias)
        if self.training:
            return universal_bias
        universal_centered = universal_bias - universal_bias.mean(
            dim=1, keepdim=True
        )
        _, proposed_class = universal_centered.max(dim=1)
        proposed_group = self.class_to_group.to(proposed_class.device)[
            proposed_class
        ]
        if self.inference_head == "routed_specialist":
            bias = torch.zeros_like(universal_bias)
            for group_id, specialist in enumerate(self.specialists):
                rows = proposed_group.eq(group_id)
                if rows.any():
                    bias[rows] = specialist(encoded[rows], self.max_bias)
        else:
            bias = universal_bias
        if self.inference_projection == "full":
            return bias
        # BAL's group analysis highlights category suppression as the failure
        # mode of a single global corrector.  The conservative inference
        # projection removes the unidentifiable common shift, keeps only the
        # strongest positive proposal, and never lowers a base logit.
        centered = bias - bias.mean(dim=1, keepdim=True)
        if self.inference_head == "routed_specialist":
            class_id = proposed_class
            value = centered.gather(1, class_id.unsqueeze(1)).squeeze(1)
            value = torch.where(
                proposed_group.ge(0), value, torch.zeros_like(value)
            )
        else:
            value, class_id = centered.max(dim=1)
        class_group = self.class_to_group.to(class_id.device)[class_id]
        group_scale_table = bias.new_tensor(self.inference_group_scales)
        valid_group = class_group.ge(0)
        scale = torch.ones_like(value)
        scale[valid_group] = group_scale_table[class_group[valid_group]]
        value = value * scale
        projected = torch.zeros_like(bias)
        projected.scatter_(1, class_id.unsqueeze(1), value.clamp_min(0.0).unsqueeze(1))
        return projected

    def forward(
        self,
        relation_features: torch.Tensor,
        union_features: torch.Tensor,
        original_logits: torch.Tensor,
    ) -> torch.Tensor:
        return original_logits + self.predict_bias(
            relation_features, union_features, original_logits
        )

    def training_losses(
        self,
        relation_features: torch.Tensor,
        union_features: torch.Tensor,
        original_logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        bias_weight: float = 1.0,
        margin_weight: float = 0.5,
        corrected_ce_weight: float = 0.075,
        preserve_weight: float = 0.5,
        distill_weight: float = 1.0,
        norm_weight: float = 1e-3,
        preserve_temperature: float = 1.0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | int]]:
        """Return fully weighted losses and detached audit diagnostics."""

        labels = labels.long()
        base = original_logits.detach()
        target_state = sparse_competition_targets(
            base,
            labels,
            target_margin=self.target_margin,
            max_correction=self.max_bias,
            background_class=self.background_class,
        )
        encoded = self.encode(relation_features, union_features, base)
        universal_bias = self.universal(encoded, self.max_bias)
        corrected = base + universal_bias
        rows = torch.arange(labels.numel(), device=labels.device)
        safe_labels = labels.clamp(min=0, max=self.num_classes - 1)
        competitor = target_state["competitor"]
        pair_prediction = torch.stack(
            [universal_bias[rows, safe_labels], universal_bias[rows, competitor]],
            dim=1,
        )
        pair_target = torch.stack(
            [
                target_state["target_bias"][rows, safe_labels],
                target_state["target_bias"][rows, competitor],
            ],
            dim=1,
        )
        bias_per_sample = F.smooth_l1_loss(
            pair_prediction, pair_target, reduction="none"
        ).mean(dim=1)
        active = target_state["active"]
        valid = target_state["valid"]
        stable = target_state["stable"]
        unlabeled_preserve = labels.lt(0)
        universal_bias_loss = _macro_class_mean(bias_per_sample, labels, active)

        corrected_margin = (
            corrected[rows, safe_labels] - corrected[rows, competitor]
        )
        margin_per_sample = F.softplus(self.target_margin - corrected_margin)
        universal_margin_loss = _macro_class_mean(
            margin_per_sample, labels, active
        )
        ce_per_sample = F.cross_entropy(corrected, safe_labels, reduction="none")
        corrected_ce = _macro_class_mean(ce_per_sample, labels, valid)

        temperature = float(preserve_temperature)
        if temperature <= 0.0:
            raise ValueError("preserve temperature must be positive")
        preserve_per_sample = F.kl_div(
            F.log_softmax(corrected / temperature, dim=1),
            F.softmax(base / temperature, dim=1),
            reduction="none",
        ).sum(dim=1) * (temperature * temperature)
        labeled_preserve_loss = _macro_class_mean(
            preserve_per_sample, labels, stable
        )
        unlabeled_preserve_loss = (
            preserve_per_sample[unlabeled_preserve].mean()
            if unlabeled_preserve.any()
            else preserve_per_sample.sum() * 0.0
        )
        preserve_loss = labeled_preserve_loss + unlabeled_preserve_loss
        universal_norm = universal_bias.square().mean(dim=1)
        foreground_norm_loss = _macro_class_mean(universal_norm, labels, valid)
        unlabeled_norm_loss = (
            universal_norm[unlabeled_preserve].mean()
            if unlabeled_preserve.any()
            else universal_norm.sum() * 0.0
        )
        norm_loss = foreground_norm_loss + unlabeled_norm_loss

        specialist_bias_terms: list[torch.Tensor] = []
        specialist_margin_terms: list[torch.Tensor] = []
        specialist_preserve_terms: list[torch.Tensor] = []
        specialist_norm_terms: list[torch.Tensor] = []
        distill_terms: list[torch.Tensor] = []
        observed_groups = 0
        group_ids = self.class_to_group.to(labels.device)[safe_labels]
        for group_id, specialist in enumerate(self.specialists):
            group_rows = valid & group_ids.eq(group_id)
            if not group_rows.any():
                continue
            observed_groups += 1
            group_bias = specialist(encoded[group_rows], self.max_bias)
            group_labels = labels[group_rows]
            group_base = base[group_rows]
            group_competitor = competitor[group_rows]
            group_target = target_state["target_bias"][group_rows]
            group_active = active[group_rows]
            group_stable = stable[group_rows]
            local_rows = torch.arange(group_labels.numel(), device=labels.device)
            local_labels = group_labels.clamp(min=0, max=self.num_classes - 1)
            group_pair_prediction = torch.stack(
                [
                    group_bias[local_rows, local_labels],
                    group_bias[local_rows, group_competitor],
                ],
                dim=1,
            )
            group_pair_target = torch.stack(
                [
                    group_target[local_rows, local_labels],
                    group_target[local_rows, group_competitor],
                ],
                dim=1,
            )
            group_bias_sample = F.smooth_l1_loss(
                group_pair_prediction, group_pair_target, reduction="none"
            ).mean(dim=1)
            specialist_bias_terms.append(
                _macro_class_mean(group_bias_sample, group_labels, group_active)
            )
            group_corrected = group_base + group_bias
            group_margin = (
                group_corrected[local_rows, local_labels]
                - group_corrected[local_rows, group_competitor]
            )
            specialist_margin_terms.append(
                _macro_class_mean(
                    F.softplus(self.target_margin - group_margin),
                    group_labels,
                    group_active,
                )
            )
            group_preserve = F.kl_div(
                F.log_softmax(group_corrected / temperature, dim=1),
                F.softmax(group_base / temperature, dim=1),
                reduction="none",
            ).sum(dim=1) * (temperature * temperature)
            specialist_preserve_terms.append(
                _macro_class_mean(group_preserve, group_labels, group_stable)
            )
            specialist_norm_terms.append(
                _macro_class_mean(
                    group_bias.square().mean(dim=1),
                    group_labels,
                    torch.ones_like(group_stable),
                )
            )
            # The universal calibrator learns each specialist's sample-level
            # correction, but specialists never chase the student.
            distill_sample = F.smooth_l1_loss(
                universal_bias[group_rows], group_bias.detach(), reduction="none"
            ).mean(dim=1)
            distill_terms.append(
                _macro_class_mean(
                    distill_sample,
                    group_labels,
                    torch.ones_like(group_stable),
                )
            )

        zero = universal_bias.sum() * 0.0
        specialist_bias_loss = (
            torch.stack(specialist_bias_terms).mean()
            if specialist_bias_terms
            else zero
        )
        specialist_margin_loss = (
            torch.stack(specialist_margin_terms).mean()
            if specialist_margin_terms
            else zero
        )
        specialist_preserve_loss = (
            torch.stack(specialist_preserve_terms).mean()
            if specialist_preserve_terms
            else zero
        )
        specialist_norm_loss = (
            torch.stack(specialist_norm_terms).mean()
            if specialist_norm_terms
            else zero
        )
        distill_loss = torch.stack(distill_terms).mean() if distill_terms else zero

        losses = {
            "loss_scc_bias": float(bias_weight)
            * (universal_bias_loss + specialist_bias_loss),
            "loss_scc_margin": float(margin_weight)
            * (universal_margin_loss + specialist_margin_loss),
            "loss_scc_corrected_ce": float(corrected_ce_weight) * corrected_ce,
            "loss_scc_preserve": float(preserve_weight)
            * (preserve_loss + specialist_preserve_loss),
            "loss_scc_distill": float(distill_weight) * distill_loss,
            "loss_scc_norm": float(norm_weight)
            * (norm_loss + specialist_norm_loss),
        }
        diagnostics: dict[str, float | int] = {
            "samples": int(labels.numel()),
            "valid_foreground": int(valid.sum()),
            "active_corrections": int(active.sum()),
            "stable_preserved": int(stable.sum()),
            "unlabeled_preserved": int(unlabeled_preserve.sum()),
            "observed_groups": int(observed_groups),
            "base_accuracy": float(
                (base.argmax(dim=1)[valid] == labels[valid]).float().mean()
                if valid.any()
                else 0.0
            ),
            "mean_base_margin": float(
                target_state["base_margin"][valid].mean() if valid.any() else 0.0
            ),
            "mean_required_correction": float(
                target_state["required_correction"][active].mean()
                if active.any()
                else 0.0
            ),
            "mean_abs_universal_bias": float(universal_bias.detach().abs().mean()),
            "max_abs_universal_bias": float(universal_bias.detach().abs().max()),
            "mean_abs_unlabeled_bias": float(
                universal_bias.detach()[unlabeled_preserve].abs().mean()
                if unlabeled_preserve.any()
                else 0.0
            ),
        }
        return losses, diagnostics

    def load_calibration_checkpoint(self, checkpoint: Mapping[str, object]) -> None:
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported sparse competition checkpoint")
        if str(checkpoint.get("group_manifest_hash")) != self.manifest_hash:
            raise RuntimeError("calibrator checkpoint/manifest hash mismatch")
        state = checkpoint.get("model")
        if not isinstance(state, Mapping):
            raise ValueError("calibrator checkpoint has no model state")
        self.load_state_dict(state, strict=True)


__all__ = [
    "CHECKPOINT_SCHEMA",
    "GROUP_MANIFEST_SCHEMA",
    "GroupDistilledCompetitionCalibrator",
    "build_group_manifest",
    "load_group_manifest",
    "save_group_manifest",
    "sparse_competition_targets",
    "train_frequency_bias",
]
