"""Bias-Aware Learning (BAL) paper reproduction components.

This module implements the two-stage method in Wang et al., *Bias-aware
learning for unbiased scene graph generation in remote sensing imagery*,
ISPRS JPRS 2026:

* train-only K-means grouping of relationship-category feature centroids;
* one SBP/BGAN model per group;
* the correction-bias construction inherited from SBP;
* five-layer 1-D-convolution G and three-layer D;
* group-by-group GCD (KLD for G, MSE for D); and
* inference with the final group's generator only.

The official BAL and SBP repositories do not publish BGAN/GCD source.  Channel
widths, Transformer token width, the optimizer for phi, and the Lipschitz
implementation required by the stated Wasserstein objective are therefore
fixed, explicit reproduction choices.  They are stored in the manifest and
checkpoint instead of being presented as paper-disclosed values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hpl_rpcm import HPLRPCM
from .rpcm_sgg_toolkit_original import RPCMSGGToolkitOriginal
from .roi_relation_predictors import _cfg_get


BAL_GROUP_SCHEMA = "bal-paper-groups-v3"


def _canonical_hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def relationship_global_bias(
    counts: torch.Tensor, *, exponent: float = 0.1, epsilon: float = 0.001
) -> torch.Tensor:
    """Equation (3) of BAL / the global-bias equation of SBP."""

    counts = counts.float().clamp_min(0.0)
    if counts.ndim != 1 or counts.numel() < 2:
        raise ValueError("relationship counts must be a 1-D class vector")
    if exponent <= 0.0 or epsilon <= 0.0:
        raise ValueError("BAL global-bias exponent and epsilon must be positive")
    proportions = counts / counts.sum().clamp_min(1.0)
    powered = proportions.clamp_min(1e-12).pow(float(exponent))
    return -(powered / powered.sum().clamp_min(1e-12) + float(epsilon)).log()


def build_bal_group_manifest(
    class_feature_centroids: torch.Tensor,
    class_counts: Sequence[int],
    *,
    class_feature_counts: Sequence[int] | None = None,
    num_groups: int = 3,
    seed: int = 1029,
    relation_names: Sequence[str] = (),
    source: Mapping[str, object] | None = None,
    conv_hidden_channels: int = 64,
    transformer_token_dim: int = 32,
    background_class: int = 0,
) -> dict[str, object]:
    """K-means relationship categories and order groups easy-to-difficult.

    The paper assigns smaller IDs to groups containing fewer categories.  Ties
    are resolved by the smallest predicate ID for deterministic distillation.
    """

    if class_feature_centroids.ndim != 2:
        raise ValueError("class_feature_centroids must be [classes, dim]")
    counts = [int(value) for value in class_counts]
    if len(counts) != class_feature_centroids.size(0):
        raise ValueError("class_counts length does not match centroids")
    feature_counts = (
        [int(value) for value in class_feature_counts]
        if class_feature_counts is not None
        else list(counts)
    )
    if len(feature_counts) != len(counts):
        raise ValueError("class_feature_counts length does not match centroids")
    if any(value < 0 for value in counts + feature_counts):
        raise ValueError("BAL class counts must be non-negative")
    background_class = int(background_class)
    if not 0 <= background_class < len(counts):
        raise ValueError("BAL background class is outside the class vector")
    frequency_valid = [
        index
        for index, count in enumerate(counts)
        if count > 0 and index != background_class
    ]
    valid = [index for index in frequency_valid if feature_counts[index] > 0]
    if len(valid) < int(num_groups):
        raise ValueError(
            "Not enough relationship classes with captured TRAIN features "
            "for BAL groups"
        )
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise RuntimeError("BAL group construction requires scikit-learn") from exc
    # BAL states ordinary K-means over relationship features.  Do not add a
    # cosine/spherical normalization that changes its Euclidean objective.
    features = class_feature_centroids[valid].float().cpu().numpy()
    raw = KMeans(
        n_clusters=int(num_groups),
        n_init=10,
        random_state=int(seed),
    ).fit_predict(features)
    raw_groups: list[list[int]] = [[] for _ in range(int(num_groups))]
    for class_id, cluster_id in zip(valid, raw.tolist()):
        raw_groups[int(cluster_id)].append(int(class_id))
    ordered = sorted(
        (sorted(group) for group in raw_groups),
        key=lambda group: (len(group), group[0]),
    )
    class_to_group = [-1] * len(counts)
    for group_id, group in enumerate(ordered):
        for class_id in group:
            class_to_group[class_id] = group_id
    # Classes absent from train data cannot be clustered without leakage.  The
    # final (most capable) SBP model handles them during inference. Background
    # is not a relationship category in the paper and is never assigned to a
    # training group, although the generator still emits its logit bias.
    for class_id, group_id in enumerate(class_to_group):
        if group_id < 0 and class_id != background_class:
            class_to_group[class_id] = int(num_groups) - 1

    foreground_bias = relationship_global_bias(
        torch.as_tensor([counts[index] for index in frequency_valid])
    )
    global_bias = [0.0] * len(counts)
    for class_id, bias in zip(frequency_valid, foreground_bias.tolist()):
        global_bias[class_id] = float(bias)
    payload: dict[str, object] = {
        "schema": BAL_GROUP_SCHEMA,
        "split": "train",
        "seed": int(seed),
        "num_groups": int(num_groups),
        "num_rel_classes": len(counts),
        "feature_dim": int(class_feature_centroids.size(1)),
        "groups": ordered,
        "class_to_group": class_to_group,
        "background_class": background_class,
        "class_counts": counts,
        "class_feature_counts": feature_counts,
        "relation_names": list(map(str, relation_names)),
        "global_bias": global_bias,
        "global_bias_exponent": 0.1,
        "global_bias_epsilon": 0.001,
        "global_bias_foreground_only": True,
        "generator_layers": 5,
        "discriminator_layers": 3,
        "conv_hidden_channels": int(conv_hidden_channels),
        "transformer_layers": 1,
        "transformer_token_dim": int(transformer_token_dim),
        # Necessary deterministic completions for the undisclosed BGAN
        # implementation.  The adversarial equations are Wasserstein losses;
        # clipping is the original WGAN Lipschitz constraint used with
        # RMSProp, and Algorithm 2 explicitly updates learning rates.
        "discriminator_weight_clip": 0.01,
        "lr_schedule": "linear_decay",
        "source": dict(source or {}),
        "paper_contract": {
            "validation_or_test_statistics": False,
            "group_order": "ascending category count",
            "inference_group": int(num_groups) - 1,
            "kmeans_input": "raw_class_feature_centroids",
            "global_bias_frequency": "all_train_gt_relationships",
            "feature_frequency": "captured_frozen_rpcm_train_relationships",
        },
    }
    payload["content_hash"] = _canonical_hash(payload)
    return payload


def save_bal_group_manifest(payload: Mapping[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_bal_group_manifest(
    path: str | Path, *, num_rel_classes: int
) -> dict[str, object]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != BAL_GROUP_SCHEMA:
        raise ValueError(f"Unsupported BAL group schema in {path}")
    expected = str(payload.get("content_hash", ""))
    unhashed = dict(payload)
    unhashed.pop("content_hash", None)
    if expected != _canonical_hash(unhashed):
        raise RuntimeError(f"BAL group manifest hash mismatch: {path}")
    if payload.get("split") != "train":
        raise ValueError("BAL grouping manifest must be TRAIN-only")
    if int(payload.get("num_rel_classes", -1)) != int(num_rel_classes):
        raise ValueError("BAL manifest class count does not match model")
    return payload


class BALCorrectionMapper(nn.Module):
    """Single-Transformer-layer mapping phi(f_union) used for b_true."""

    def __init__(
        self,
        union_dim: int,
        num_rel_classes: int,
        token_dim: int = 32,
        attention_heads: int = 4,
    ):
        super().__init__()
        if token_dim % attention_heads:
            raise ValueError("BAL token_dim must be divisible by attention_heads")
        self.num_rel_classes = int(num_rel_classes)
        self.feature_projection = nn.Linear(union_dim, token_dim)
        self.class_tokens = nn.Parameter(
            torch.empty(num_rel_classes, token_dim)
        )
        nn.init.normal_(self.class_tokens, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=attention_heads,
            dim_feedforward=token_dim * 4,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.output = nn.Linear(token_dim, 1)

    def forward(self, union_features: torch.Tensor) -> torch.Tensor:
        global_token = self.feature_projection(union_features).unsqueeze(1)
        tokens = global_token + self.class_tokens.unsqueeze(0)
        return self.output(self.transformer(tokens)).squeeze(-1)


def construct_correction_bias(
    intermediate_bias: torch.Tensor,
    original_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    epsilon: float = 0.0001,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the correction set ``S`` with the paper's intended margin.

    BAL states that the ground-truth entry is updated, but prints the margin
    with the opposite sign; SBP prints the complementary predicted-entry
    update with the opposite epsilon sign.  Either literal formula can remain
    misclassified.  The unique one-step update consistent with BAL's stated
    target is ``b_true += zhat[pred] - zhat[true] + xi``: the ground-truth
    score becomes the previous maximum plus ``xi``.
    """

    if intermediate_bias.shape != original_logits.shape:
        raise ValueError("intermediate bias and logits must have equal shape")
    labels = labels.long()
    if labels.shape != (original_logits.size(0),):
        raise ValueError("BAL labels must have shape [samples]")
    corrected_logits = original_logits + intermediate_bias
    predicted = corrected_logits.argmax(dim=1)
    wrong = predicted.ne(labels)
    target = intermediate_bias.clone()
    if wrong.any():
        rows = torch.nonzero(wrong, as_tuple=False).flatten()
        true = labels[rows]
        pred = predicted[rows]
        difference = corrected_logits[rows, pred] - corrected_logits[rows, true]
        target[rows, true] = target[rows, true] + difference + float(epsilon)
    final_prediction = (original_logits + target).argmax(dim=1)
    if not final_prediction.eq(labels).all():
        raise RuntimeError("BAL correction construction failed to make GT maximal")
    return target, final_prediction.eq(labels)


def _conv_stack(
    input_channels: int,
    hidden_channels: int,
    output_channels: int,
    layers: int,
) -> nn.Sequential:
    if layers < 2:
        raise ValueError("BAL Conv1d stack requires at least two layers")
    modules: list[nn.Module] = []
    for layer_id in range(layers):
        in_channels = input_channels if layer_id == 0 else hidden_channels
        out_channels = output_channels if layer_id == layers - 1 else hidden_channels
        modules.append(nn.Conv1d(in_channels, out_channels, 3, padding=1))
        if layer_id != layers - 1:
            modules.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*modules)


class BALGenerator(nn.Module):
    """Five-layer 1-D-convolution sample-specific bias generator."""

    def __init__(
        self,
        union_dim: int,
        num_rel_classes: int,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.union_to_relationship = nn.Linear(union_dim, num_rel_classes)
        self.network = _conv_stack(3, hidden_channels, 1, layers=5)

    def forward(
        self,
        union_features: torch.Tensor,
        global_bias: torch.Tensor,
        original_logits: torch.Tensor,
    ) -> torch.Tensor:
        if global_bias.ndim == 1:
            global_bias = global_bias.unsqueeze(0).expand(original_logits.size(0), -1)
        union_sequence = self.union_to_relationship(union_features)
        stacked = torch.stack((union_sequence, global_bias, original_logits), dim=1)
        return self.network(stacked).squeeze(1)


class BALDiscriminator(nn.Module):
    """Three-layer 1-D-convolution Wasserstein discriminator."""

    def __init__(self, hidden_channels: int = 64):
        super().__init__()
        self.network = _conv_stack(1, hidden_channels, 1, layers=3)

    def forward(self, bias: torch.Tensor) -> torch.Tensor:
        return self.network(bias.unsqueeze(1)).mean(dim=2).squeeze(1)


class BALGroupModel(nn.Module):
    def __init__(
        self,
        union_dim: int,
        num_rel_classes: int,
        *,
        hidden_channels: int,
        token_dim: int,
    ):
        super().__init__()
        self.correction_mapper = BALCorrectionMapper(
            union_dim, num_rel_classes, token_dim=token_dim
        )
        self.generator = BALGenerator(
            union_dim, num_rel_classes, hidden_channels=hidden_channels
        )
        self.discriminator = BALDiscriminator(hidden_channels=hidden_channels)


class BALPaperModule(nn.Module):
    """All grouped SBP models and paper losses; inference uses final G."""

    def __init__(
        self,
        union_dim: int,
        num_rel_classes: int,
        manifest: Mapping[str, object],
        *,
        alpha: float = 0.075,
        epsilon: float = 0.0001,
        distillation_weight: float = 1.0,
    ):
        super().__init__()
        self.union_dim = int(union_dim)
        self.num_rel_classes = int(num_rel_classes)
        self.manifest_hash = str(manifest["content_hash"])
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.distillation_weight = float(distillation_weight)
        self.num_groups = int(manifest["num_groups"])
        self.discriminator_weight_clip = float(
            manifest.get("discriminator_weight_clip", 0.01)
        )
        if self.discriminator_weight_clip <= 0.0:
            raise ValueError("BAL discriminator weight clip must be positive")
        hidden = int(manifest.get("conv_hidden_channels", 64))
        token_dim = int(manifest.get("transformer_token_dim", 32))
        self.groups = nn.ModuleList(
            [
                BALGroupModel(
                    union_dim,
                    num_rel_classes,
                    hidden_channels=hidden,
                    token_dim=token_dim,
                )
                for _ in range(self.num_groups)
            ]
        )
        self.register_buffer(
            "class_to_group",
            torch.as_tensor(manifest["class_to_group"], dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "global_bias",
            torch.as_tensor(manifest["global_bias"], dtype=torch.float32),
            persistent=True,
        )

    @property
    def inference_group(self) -> BALGroupModel:
        return self.groups[-1]

    def predict_bias(
        self, union_features: torch.Tensor, original_logits: torch.Tensor
    ) -> torch.Tensor:
        return self.inference_group.generator(
            union_features, self.global_bias, original_logits
        )

    def forward(
        self, union_features: torch.Tensor, original_logits: torch.Tensor
    ) -> torch.Tensor:
        return original_logits + self.predict_bias(union_features, original_logits)

    def correction_targets(
        self,
        group_id: int,
        union_features: torch.Tensor,
        original_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mapper = self.groups[int(group_id)].correction_mapper
        intermediate = mapper(union_features) + self.global_bias.unsqueeze(0)
        # S is a training target, not a route for optimizing phi through D.
        return construct_correction_bias(
            intermediate,
            original_logits,
            labels,
            epsilon=self.epsilon,
        )

    def correction_mapper_loss(
        self,
        group_id: int,
        union_features: torch.Tensor,
        original_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Operational completion for the paper's unspecified phi training.

        BAL states that ``phi`` is a one-layer Transformer used to construct
        correction targets, but Algorithms 1/2 only define optimization for G
        and D.  A short, separately recorded CE warm-up makes phi a meaningful
        mapper before its outputs are detached into the correction set.  This
        step can be set to zero for a literal algorithm-only ablation.
        """

        mapped = self.groups[int(group_id)].correction_mapper(union_features)
        return F.cross_entropy(
            original_logits.detach() + mapped + self.global_bias.unsqueeze(0),
            labels.long(),
        )

    def generator_loss(
        self,
        group_id: int,
        union_features: torch.Tensor,
        original_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model = self.groups[int(group_id)]
        predicted_bias = model.generator(
            union_features, self.global_bias, original_logits
        )
        score = model.discriminator(predicted_bias)
        corrected = original_logits + predicted_bias
        adversarial = -score.mean()
        corrected_ce = F.cross_entropy(corrected, labels.long())
        return adversarial + self.alpha * corrected_ce, {
            "adversarial": adversarial.detach(),
            "corrected_ce": corrected_ce.detach(),
        }

    def discriminator_loss(
        self,
        group_id: int,
        union_features: torch.Tensor,
        original_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model = self.groups[int(group_id)]
        with torch.no_grad():
            true_bias, corrected = self.correction_targets(
                group_id, union_features, original_logits, labels
            )
            predicted_bias = model.generator(
                union_features, self.global_bias, original_logits
            )
        true_score = model.discriminator(true_bias)
        generated_score = model.discriminator(predicted_bias)
        loss = -true_score.mean() + generated_score.mean()
        return loss, {
            "true_score": true_score.mean().detach(),
            "generated_score": generated_score.mean().detach(),
            "target_correction_rate": corrected.float().mean().detach(),
        }

    def generator_distillation_loss(
        self,
        student_group: int,
        teacher_group: int,
        union_features: torch.Tensor,
        original_logits: torch.Tensor,
    ) -> torch.Tensor:
        student = self.groups[int(student_group)].generator(
            union_features, self.global_bias, original_logits
        )
        with torch.no_grad():
            teacher = self.groups[int(teacher_group)].generator(
                union_features, self.global_bias, original_logits
            )
        return F.kl_div(
            F.log_softmax(student, dim=1),
            F.softmax(teacher, dim=1),
            reduction="batchmean",
        )

    def discriminator_distillation_loss(
        self,
        student_group: int,
        teacher_group: int,
        union_features: torch.Tensor,
        original_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        student_model = self.groups[int(student_group)]
        teacher_model = self.groups[int(teacher_group)]
        with torch.no_grad():
            teacher_true, _ = self.correction_targets(
                teacher_group, union_features, original_logits, labels
            )
            teacher_generated = teacher_model.generator(
                union_features, self.global_bias, original_logits
            )
            # Equation (10) defines the arrowed quantity as the output of
            # group n on the *same input data* used by group j.  It is not the
            # output of D_n applied to G_j's generated bias.
            student_generated = student_model.generator(
                union_features, self.global_bias, original_logits
            )
            teacher_true_score = teacher_model.discriminator(teacher_true)
            teacher_generated_score = teacher_model.discriminator(teacher_generated)
        student_true_score = student_model.discriminator(teacher_true)
        student_generated_score = student_model.discriminator(student_generated)
        return F.mse_loss(student_true_score, teacher_true_score) + F.mse_loss(
            student_generated_score, teacher_generated_score
        )


class BALInferenceModule(nn.Module):
    """The paper inference graph: global bias plus final-group G only."""

    def __init__(
        self,
        union_dim: int,
        num_rel_classes: int,
        manifest: Mapping[str, object],
    ):
        super().__init__()
        self.num_groups = int(manifest["num_groups"])
        self.manifest_hash = str(manifest["content_hash"])
        self.generator = BALGenerator(
            int(union_dim),
            int(num_rel_classes),
            hidden_channels=int(manifest.get("conv_hidden_channels", 64)),
        )
        self.register_buffer(
            "global_bias",
            torch.as_tensor(manifest["global_bias"], dtype=torch.float32),
            persistent=True,
        )

    def load_paper_checkpoint(self, checkpoint: Mapping[str, object]) -> None:
        schema = str(checkpoint.get("schema", ""))
        if schema and schema != "bal-paper-checkpoint-v3":
            raise ValueError(f"Unsupported BAL checkpoint schema: {schema}")
        checkpoint_hash = str(checkpoint.get("manifest_hash", ""))
        if checkpoint_hash and checkpoint_hash != self.manifest_hash:
            raise RuntimeError(
                "BAL checkpoint and group manifest hashes do not match"
            )
        state = checkpoint.get("bal_state_dict", checkpoint)
        if not isinstance(state, Mapping):
            raise TypeError("BAL checkpoint has no bal_state_dict")
        prefix = f"groups.{self.num_groups - 1}.generator."
        generator_state = {
            str(key)[len(prefix) :]: value
            for key, value in state.items()
            if str(key).startswith(prefix)
        }
        incompatible = self.generator.load_state_dict(generator_state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "BAL final generator is incompatible: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        checkpoint_bias = state.get("global_bias")
        if checkpoint_bias is not None and not torch.equal(
            torch.as_tensor(checkpoint_bias).cpu(), self.global_bias.cpu()
        ):
            raise RuntimeError("BAL checkpoint global bias differs from manifest")

    def forward(
        self, union_features: torch.Tensor, original_logits: torch.Tensor
    ) -> torch.Tensor:
        predicted_bias = self.generator(
            union_features, self.global_bias, original_logits
        )
        return original_logits + predicted_bias


class BALPaperPostTrainer:
    """RMSProp/5D:1G optimizer implementing BAL Algorithms 2 and GCD."""

    def __init__(
        self,
        module: BALPaperModule,
        *,
        generator_lr: float = 0.0001,
        discriminator_lr: float = 0.0005,
        discriminator_steps: int = 5,
        mapper_lr: float = 0.0001,
        total_iterations: int = 4000,
    ):
        self.module = module
        self.discriminator_steps = int(discriminator_steps)
        if self.discriminator_steps != 5:
            raise ValueError("Paper reproduction requires five D steps per G step")
        self.total_iterations = int(total_iterations)
        if self.total_iterations <= 0:
            raise ValueError("BAL total iterations must be positive")
        self.generator_optimizers = [
            torch.optim.RMSprop(group.generator.parameters(), lr=float(generator_lr))
            for group in module.groups
        ]
        self.discriminator_optimizers = [
            torch.optim.RMSprop(group.discriminator.parameters(), lr=float(discriminator_lr))
            for group in module.groups
        ]
        self.mapper_optimizers = [
            torch.optim.RMSprop(
                group.correction_mapper.parameters(), lr=float(mapper_lr)
            )
            for group in module.groups
        ]
        self._initial_generator_lrs = [
            float(optimizer.param_groups[0]["lr"])
            for optimizer in self.generator_optimizers
        ]
        self._initial_discriminator_lrs = [
            float(optimizer.param_groups[0]["lr"])
            for optimizer in self.discriminator_optimizers
        ]
        self._lr_step = 0

    def learning_rates(self) -> dict[str, list[float]]:
        return {
            "generator": [
                float(optimizer.param_groups[0]["lr"])
                for optimizer in self.generator_optimizers
            ],
            "discriminator": [
                float(optimizer.param_groups[0]["lr"])
                for optimizer in self.discriminator_optimizers
            ],
        }

    def warmup_mapper_step(
        self,
        group_id: int,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> float:
        union_features, original_logits, labels = batch
        group_id = int(group_id)
        optimizer = self.mapper_optimizers[group_id]
        optimizer.zero_grad(set_to_none=True)
        loss = self.module.correction_mapper_loss(
            group_id, union_features, original_logits, labels
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite BAL phi loss in group {group_id}")
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def step_learning_rates(self) -> None:
        """Algorithm 2 line 13: update G/D learning rates once per iteration."""

        self._lr_step += 1
        factor = max(
            0.0,
            1.0 - float(self._lr_step) / float(self.total_iterations),
        )
        for optimizer, initial_lr in zip(
            self.generator_optimizers, self._initial_generator_lrs
        ):
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = initial_lr * factor
        for optimizer, initial_lr in zip(
            self.discriminator_optimizers, self._initial_discriminator_lrs
        ):
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = initial_lr * factor

    @staticmethod
    def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def step(
        self,
        group_id: int,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        previous_group_batches: Mapping[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] | None = None,
    ) -> dict[str, float]:
        union_features, original_logits, labels = batch
        group_id = int(group_id)
        previous_group_batches = previous_group_batches or {}
        model = self.module.groups[group_id]
        d_optimizer = self.discriminator_optimizers[group_id]
        g_optimizer = self.generator_optimizers[group_id]
        last_d = None
        d_diag: dict[str, torch.Tensor] = {}
        for _ in range(self.discriminator_steps):
            self._set_requires_grad(model.discriminator, True)
            d_optimizer.zero_grad(set_to_none=True)
            d_loss, d_diag = self.module.discriminator_loss(
                group_id, union_features, original_logits, labels
            )
            for teacher_id in range(group_id):
                teacher_batch = previous_group_batches.get(teacher_id)
                if teacher_batch is None:
                    continue
                d_loss = d_loss + self.module.distillation_weight * (
                    self.module.discriminator_distillation_loss(
                        group_id, teacher_id, *teacher_batch
                    )
                )
            if not torch.isfinite(d_loss):
                raise FloatingPointError(
                    f"Non-finite BAL discriminator loss in group {group_id}"
                )
            d_loss.backward()
            d_optimizer.step()
            # Equations (6)/(7) are the Wasserstein objective.  With the
            # paper's RMSProp optimizer, the standard WGAN parameter clipping
            # supplies the otherwise-missing Lipschitz constraint.
            clip = self.module.discriminator_weight_clip
            with torch.no_grad():
                for parameter in model.discriminator.parameters():
                    parameter.clamp_(-clip, clip)
            last_d = d_loss.detach()

        self._set_requires_grad(model.discriminator, False)
        g_optimizer.zero_grad(set_to_none=True)
        g_loss, g_diag = self.module.generator_loss(
            group_id, union_features, original_logits, labels
        )
        for teacher_id in range(group_id):
            teacher_batch = previous_group_batches.get(teacher_id)
            if teacher_batch is None:
                continue
            teacher_union, teacher_logits, _ = teacher_batch
            g_loss = g_loss + self.module.distillation_weight * (
                self.module.generator_distillation_loss(
                    group_id, teacher_id, teacher_union, teacher_logits
                )
            )
        if not torch.isfinite(g_loss):
            raise FloatingPointError(
                f"Non-finite BAL generator loss in group {group_id}"
            )
        g_loss.backward()
        g_optimizer.step()
        self._set_requires_grad(model.discriminator, True)
        return {
            "group": float(group_id),
            "loss_g": float(g_loss.detach()),
            "loss_d": float(last_d if last_d is not None else 0.0),
            "lr_g": float(g_optimizer.param_groups[0]["lr"]),
            "lr_d": float(d_optimizer.param_groups[0]["lr"]),
            **{f"g_{key}": float(value) for key, value in g_diag.items()},
            **{f"d_{key}": float(value) for key, value in d_diag.items()},
        }


class _BALRPCMMixin:
    def _init_bal(self, cfg: dict) -> None:
        manifest_path = str(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "BAL",
                "GROUP_MANIFEST",
                default="",
            )
        ).strip()
        if not manifest_path:
            raise FileNotFoundError(
                "BAL predictor requires BAL.GROUP_MANIFEST built from a TRAIN cache"
            )
        manifest = load_bal_group_manifest(
            manifest_path, num_rel_classes=self.num_rel_classes
        )
        bal_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"].get("BAL", {})
        self.bal = BALInferenceModule(
            self.pooling_dim,
            self.num_rel_classes,
            manifest,
        )
        checkpoint_path = str(bal_cfg.get("CHECKPOINT", "")).strip()
        if not checkpoint_path:
            raise FileNotFoundError(
                "BAL.ENABLED requires BAL.CHECKPOINT from tools/train_bal.py"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.bal.load_paper_checkpoint(checkpoint)
        print(
            "[BAL] "
            f"groups={self.bal.num_groups}, manifest={self.bal.manifest_hash}, "
            f"checkpoint={checkpoint_path}, inference=final_group_generator",
            flush=True,
        )

    def _calibrate_relation_logits(
        self,
        relation_logits: torch.Tensor,
        *,
        union_features: torch.Tensor | tuple[torch.Tensor, ...] | None,
        relation_representation: torch.Tensor,
        rel_labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del relation_representation, rel_labels
        if self.training:
            raise RuntimeError(
                "BAL is a frozen post-training method; use tools/train_bal.py, "
                "not the ordinary SGG optimizer"
            )
        if not isinstance(union_features, torch.Tensor):
            raise TypeError("BAL requires the fused union feature tensor")
        return self.bal(union_features, relation_logits), {}


class BALRPCM(_BALRPCMMixin, RPCMSGGToolkitOriginal):
    """Frozen Original RPCM followed by the paper BAL generator."""

    def __init__(self, cfg: dict, in_channels: int):
        super().__init__(cfg, in_channels)
        self._init_bal(cfg)


class HPLBALRPCM(_BALRPCMMixin, HPLRPCM):
    """HPL-RPCM followed by the independent BAL post-training stage."""

    def __init__(self, cfg: dict, in_channels: int):
        super().__init__(cfg, in_channels)
        self._init_bal(cfg)


__all__ = [
    "BALCorrectionMapper",
    "BALDiscriminator",
    "BALGenerator",
    "BALGroupModel",
    "BALInferenceModule",
    "BALPaperModule",
    "BALPaperPostTrainer",
    "BALRPCM",
    "HPLBALRPCM",
    "build_bal_group_manifest",
    "construct_correction_bias",
    "load_bal_group_manifest",
    "relationship_global_bias",
    "save_bal_group_manifest",
]
