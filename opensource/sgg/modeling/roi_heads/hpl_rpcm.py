"""Paper-faithful HPL-Net components adapted to the Original RPCM backbone.

The implementation follows Wang et al., *Hierarchical Prototype Learning via
Aggregation-Decomposition for Fine-Grained Geospatial Scene Graph
Generation*, TGRS 2025.  Dataset-dependent hierarchy construction is kept in
an explicit train-only manifest; no validation/test statistics are read by the
model.

The paper does not publish hidden widths, the HC temperature, the exact
head/tail split point, or the test-time tail routing rule.  Runnable,
deterministic defaults for those necessary engineering choices are recorded in
the configuration and checkpoint.  No additional classifier, residual loss,
or independently learnable upper-layer prototype is introduced.  The FR
dictionary is projected to the initial-relation width by a semantic En(.)
branch, while a relation En(.) branch maps RPCM features into the same latent
space. All stated paper hyperparameters (three layers, 0.9/0.5 clustering
thresholds, Delta=100, beta=0.125, focal alpha=0.1/gamma=2) retain their
published values.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rpcm_sgg_toolkit_original import RPCMSGGToolkitOriginal
from .roi_relation_predictors import _cfg_get


HPL_HIERARCHY_SCHEMA = "hpl-paper-hierarchy-v3"
HPL_MODULE_SCHEMA = "hpl-paper-module-v4"


def _canonical_hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass
class _AgglomerativeCluster:
    members: list[int]
    center: torch.Tensor
    weight: int


def _iterative_cosine_merge(
    vectors: Sequence[torch.Tensor],
    weights: Sequence[int],
    threshold: float,
) -> tuple[list[int], list[torch.Tensor], list[int]]:
    """Deterministic implementation of the paper's iterative merging.

    At every step the most similar pair at or above ``threshold`` is merged.
    Lexicographic member IDs resolve exact ties.  This makes the hierarchy
    invariant to hash/dictionary iteration order and reproducible by seed.
    """

    if len(vectors) != len(weights):
        raise ValueError("vectors and weights must have the same length")
    if not vectors:
        return [], [], []
    clusters = [
        _AgglomerativeCluster([idx], vector.float().clone(), int(weight))
        for idx, (vector, weight) in enumerate(zip(vectors, weights))
    ]
    if any(cluster.weight <= 0 for cluster in clusters):
        raise ValueError("cluster weights must be positive")

    while len(clusters) > 1:
        best: tuple[float, tuple[int, ...], tuple[int, ...], int, int] | None = None
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                similarity = float(
                    F.cosine_similarity(
                        clusters[left].center,
                        clusters[right].center,
                        dim=0,
                        eps=1e-8,
                    )
                )
                candidate = (
                    similarity,
                    tuple(-value for value in clusters[left].members),
                    tuple(-value for value in clusters[right].members),
                    left,
                    right,
                )
                if best is None or candidate > best:
                    best = candidate
        assert best is not None
        similarity, _, _, left, right = best
        if similarity < float(threshold):
            break
        lhs, rhs = clusters[left], clusters[right]
        total_weight = lhs.weight + rhs.weight
        merged = _AgglomerativeCluster(
            members=sorted(lhs.members + rhs.members),
            center=(lhs.center * lhs.weight + rhs.center * rhs.weight)
            / float(total_weight),
            weight=total_weight,
        )
        clusters[left] = merged
        del clusters[right]
        clusters.sort(key=lambda cluster: tuple(cluster.members))

    assignments = [0] * len(vectors)
    for cluster_id, cluster in enumerate(clusters):
        for member in cluster.members:
            assignments[member] = cluster_id
    return (
        assignments,
        [cluster.center for cluster in clusters],
        [cluster.weight for cluster in clusters],
    )


def build_hpl_hierarchy_manifest(
    predicate_pairs: Iterable[tuple[int, int, int, int]],
    object_embeddings: torch.Tensor,
    *,
    num_rel_classes: int,
    predicate_counts: Sequence[int],
    object_names: Sequence[str] = (),
    relation_names: Sequence[str] = (),
    child_threshold: float = 0.9,
    parent_threshold: float = 0.5,
    tail_fraction: float = 0.5,
    tail_class_ids: Sequence[int] = (),
    seed: int = 1029,
    source: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Construct the child-predicate-parent hierarchy from TRAIN relations.

    ``predicate_pairs`` rows are ``(predicate, subject_class, object_class,
    count)``.  Background (predicate 0) is intentionally not part of the HPL
    hierarchy; it remains present in the ordinary SGG classifier.
    """

    if object_embeddings.ndim != 2:
        raise ValueError("object_embeddings must have shape [objects, dim]")
    if int(num_rel_classes) < 2:
        raise ValueError("HPL requires at least one foreground predicate")
    counts = [int(value) for value in predicate_counts]
    if len(counts) != int(num_rel_classes):
        raise ValueError("predicate_counts must include background and all predicates")
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError("tail_fraction must be in (0,1]")
    configured_tail_ids = [int(value) for value in tail_class_ids]
    if configured_tail_ids:
        if len(set(configured_tail_ids)) != len(configured_tail_ids):
            raise ValueError("tail_class_ids must not contain duplicates")
        if any(
            value <= 0 or value >= int(num_rel_classes)
            for value in configured_tail_ids
        ):
            raise ValueError("tail_class_ids must contain foreground predicates")
        resolved_tail_ids = configured_tail_ids
        tail_group_source = "explicit_train_protocol"
    else:
        foreground_order = sorted(
            range(1, int(num_rel_classes)),
            key=lambda predicate: (-counts[predicate], predicate),
        )
        tail_count = max(
            1,
            min(
                len(foreground_order),
                int(round(len(foreground_order) * float(tail_fraction))),
            ),
        )
        resolved_tail_ids = foreground_order[-tail_count:]
        tail_group_source = "train_predicate_frequency_rank"

    grouped: dict[int, list[tuple[int, int, int]]] = {
        predicate: [] for predicate in range(1, int(num_rel_classes))
    }
    for predicate, subject, object_, count in predicate_pairs:
        predicate, subject, object_, count = map(
            int, (predicate, subject, object_, count)
        )
        if predicate <= 0 or predicate >= int(num_rel_classes) or count <= 0:
            continue
        if not (0 <= subject < object_embeddings.size(0)) or not (
            0 <= object_ < object_embeddings.size(0)
        ):
            raise ValueError("object class in predicate_pairs is out of range")
        grouped[predicate].append((subject, object_, count))

    child_vectors: list[torch.Tensor] = []
    child_to_predicate: list[int] = []
    child_weights: list[int] = []
    child_instance_counts: list[int] = []
    pair_to_child: dict[str, int] = {}
    predicate_vectors: list[torch.Tensor] = []
    predicate_ids: list[int] = []

    pair_dim = int(object_embeddings.size(1) * 2)
    for predicate in range(1, int(num_rel_classes)):
        entries = sorted(grouped[predicate], key=lambda row: (row[0], row[1]))
        if not entries:
            # A class absent from TRAIN cannot be supervised without leakage.
            # Preserve a deterministic zero semantic prototype and mark it in
            # the manifest so audits can distinguish this case.
            predicate_ids.append(predicate)
            predicate_vectors.append(torch.zeros(pair_dim, dtype=torch.float32))
            continue
        vectors = [
            torch.cat(
                (
                    object_embeddings[subject].float(),
                    object_embeddings[object_].float(),
                )
            )
            for subject, object_, _ in entries
        ]
        # HPL decomposes the semantic variability of distinct subject-object
        # category combinations.  Occurrence frequency belongs to the CC
        # class margin, not semantic hierarchy geometry.
        weights = [1] * len(entries)
        assignments, centers, cluster_weights = _iterative_cosine_merge(
            vectors, weights, child_threshold
        )
        first_child = len(child_vectors)
        child_vectors.extend(centers)
        child_to_predicate.extend([predicate] * len(centers))
        child_weights.extend(cluster_weights)
        instance_counts = [0] * len(centers)
        for (subject, object_, _), cluster_id in zip(entries, assignments):
            pair_to_child[f"{predicate}:{subject}:{object_}"] = (
                first_child + int(cluster_id)
            )
        for (_, _, occurrence_count), cluster_id in zip(entries, assignments):
            instance_counts[int(cluster_id)] += int(occurrence_count)
        child_instance_counts.extend(instance_counts)
        predicate_vector = torch.stack(centers, dim=0).mean(dim=0)
        predicate_ids.append(predicate)
        predicate_vectors.append(predicate_vector)

    valid_parent_rows = [
        idx for idx, vector in enumerate(predicate_vectors) if vector.abs().sum() > 0
    ]
    if valid_parent_rows:
        valid_vectors = [predicate_vectors[idx] for idx in valid_parent_rows]
        valid_weights = [1 for _ in valid_parent_rows]
        assignments, parent_vectors, parent_weights = _iterative_cosine_merge(
            valid_vectors, valid_weights, parent_threshold
        )
    else:
        assignments, parent_vectors, parent_weights = [], [], []
    predicate_to_parent = [-1] * len(predicate_ids)
    for predicate_row, parent_id in zip(valid_parent_rows, assignments):
        predicate_to_parent[predicate_row] = int(parent_id)

    manifest: dict[str, object] = {
        "schema": HPL_HIERARCHY_SCHEMA,
        "seed": int(seed),
        "split": "train",
        "layers": 3,
        "num_object_classes": int(object_embeddings.size(0)),
        "num_rel_classes": int(num_rel_classes),
        "object_embedding_dim": int(object_embeddings.size(1)),
        "pair_embedding_dim": pair_dim,
        "child_threshold": float(child_threshold),
        "parent_threshold": float(parent_threshold),
        "predicate_counts": counts,
        "tail_group": {
            "source": tail_group_source,
            "fraction": float(tail_fraction),
            "predicate_ids": resolved_tail_ids,
        },
        "object_names": list(map(str, object_names)),
        "relation_names": list(map(str, relation_names)),
        "child_vectors": [vector.tolist() for vector in child_vectors],
        "child_to_predicate": child_to_predicate,
        "child_weights": child_weights,
        "child_instance_counts": child_instance_counts,
        "predicate_ids": predicate_ids,
        "predicate_vectors": [vector.tolist() for vector in predicate_vectors],
        "predicate_to_parent": predicate_to_parent,
        "parent_vectors": [vector.tolist() for vector in parent_vectors],
        "parent_weights": parent_weights,
        "pair_to_child": pair_to_child,
        "source": dict(source or {}),
        "paper_contract": {
            "construction": "iterative cosine aggregation-decomposition",
            "optimization": "bottom-up centroids over the initialized tree",
            "background_in_hierarchy": False,
            "validation_or_test_statistics": False,
        },
    }
    manifest["content_hash"] = _canonical_hash(manifest)
    return manifest


def load_hpl_hierarchy_manifest(
    path: str | Path, *, num_rel_classes: int
) -> dict[str, object]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != HPL_HIERARCHY_SCHEMA:
        raise ValueError(f"Unsupported HPL hierarchy schema in {path}")
    expected_hash = str(payload.get("content_hash", ""))
    unhashed = dict(payload)
    unhashed.pop("content_hash", None)
    if expected_hash != _canonical_hash(unhashed):
        raise RuntimeError(f"HPL hierarchy hash mismatch: {path}")
    if payload.get("split") != "train":
        raise ValueError("HPL hierarchy must be constructed from TRAIN only")
    if int(payload.get("layers", -1)) != 3:
        raise ValueError("The paper reproduction requires exactly three layers")
    if int(payload.get("num_rel_classes", -1)) != int(num_rel_classes):
        raise ValueError("HPL hierarchy relation-class count does not match model")
    return payload


def save_hpl_hierarchy_manifest(payload: Mapping[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class _TwoLayerEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class HPLPaperModule(nn.Module):
    """FR, hierarchical constraint (HC), and class-aware constraint (CC)."""

    def __init__(
        self,
        cfg: Mapping[str, object],
        feature_dim: int,
        num_rel_classes: int,
        manifest: Mapping[str, object],
    ):
        super().__init__()
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]  # type: ignore[index]
        hpl_cfg = rel_cfg.get("HPL", {})
        self.feature_dim = int(feature_dim)
        self.num_rel_classes = int(num_rel_classes)
        self.hierarchy_hash = str(manifest["content_hash"])
        self.tau = float(hpl_cfg.get("HC_TEMPERATURE", 0.1))
        self.delta = float(hpl_cfg.get("CC_DELTA", 100.0))
        self.beta = float(hpl_cfg.get("CC_BETA", 0.125))
        self.cc_normalize = bool(hpl_cfg.get("CC_NORMALIZE", True))
        self.cc_negative_mode = str(
            hpl_cfg.get("CC_NEGATIVE_MODE", "random")
        ).strip().lower()
        self.cc_negative_temperature = float(
            hpl_cfg.get("CC_NEGATIVE_TEMPERATURE", 0.1)
        )
        self.fr_enabled = bool(hpl_cfg.get("FR_ENABLED", True))
        self.hc_enabled = bool(hpl_cfg.get("HC_ENABLED", True))
        self.cc_enabled = bool(hpl_cfg.get("CC_ENABLED", True))
        self.tail_fraction = float(hpl_cfg.get("TAIL_FRACTION", 0.5))
        self.eval_tail_route = str(
            hpl_cfg.get("EVAL_TAIL_ROUTE", "base_prediction")
        ).lower()
        self.diagnostic_eval_route = str(
            hpl_cfg.get("DIAGNOSTIC_EVAL_ROUTE", "current")
        ).lower()
        self.matching_diagnostic_enabled = bool(
            hpl_cfg.get("MATCHING_DIAGNOSTIC_ENABLED", False)
        )
        if self.tau <= 0:
            raise ValueError("HPL.HC_TEMPERATURE must be positive")
        if not 0.0 < self.tail_fraction <= 1.0:
            raise ValueError("HPL.TAIL_FRACTION must be in (0,1]")
        if self.cc_negative_mode not in {
            "random",
            "all_mean",
            "logsumexp",
            "hardest",
        }:
            raise ValueError(
                "HPL.CC_NEGATIVE_MODE must be random, all_mean, logsumexp "
                f"or hardest, got {self.cc_negative_mode!r}"
            )
        if self.cc_negative_temperature <= 0:
            raise ValueError("HPL.CC_NEGATIVE_TEMPERATURE must be positive")
        valid_eval_tail_routes = {
            "base_prediction",
            "foreground_prediction",
            "soft_probability",
        }
        if self.eval_tail_route not in valid_eval_tail_routes:
            raise ValueError(
                "HPL.EVAL_TAIL_ROUTE must be one of "
                f"{sorted(valid_eval_tail_routes)}, got "
                f"{self.eval_tail_route!r}"
            )
        valid_diagnostic_routes = {"current", "pre_fr", "all", "gt_tail"}
        if self.diagnostic_eval_route not in valid_diagnostic_routes:
            raise ValueError(
                "HPL.DIAGNOSTIC_EVAL_ROUTE must be one of "
                f"{sorted(valid_diagnostic_routes)}, got "
                f"{self.diagnostic_eval_route!r}"
            )

        child = torch.as_tensor(manifest["child_vectors"], dtype=torch.float32)
        predicate = torch.as_tensor(
            manifest["predicate_vectors"], dtype=torch.float32
        )
        parent = torch.as_tensor(manifest["parent_vectors"], dtype=torch.float32)
        raw_dim = int(manifest["pair_embedding_dim"])
        if child.ndim != 2 or child.size(1) != raw_dim or child.size(0) == 0:
            raise ValueError("HPL manifest must contain foreground child prototypes")
        if predicate.shape != (self.num_rel_classes - 1, raw_dim):
            raise ValueError("HPL predicate prototype matrix has invalid shape")
        if parent.ndim != 2 or parent.size(1) != raw_dim or parent.size(0) == 0:
            raise ValueError("HPL manifest must contain parent prototypes")
        # The paper updates the hierarchy bottom-up and defines every upper
        # prototype as a cluster centroid.  Only child prototypes are free
        # parameters; predicate/parent prototypes are recomputed from the
        # current child parameters on every forward pass.
        self.raw_prototype_dim = raw_dim
        self.child_prototypes = nn.Parameter(child.clone())
        self.register_buffer(
            "child_to_predicate",
            torch.as_tensor(manifest["child_to_predicate"], dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "predicate_to_parent",
            torch.as_tensor(manifest["predicate_to_parent"], dtype=torch.long),
            persistent=True,
        )
        if (self.child_to_predicate < 1).any() or (
            self.child_to_predicate >= self.num_rel_classes
        ).any():
            raise ValueError("HPL child-to-predicate mapping is invalid")
        if (self.predicate_to_parent < 0).any() or (
            self.predicate_to_parent >= parent.size(0)
        ).any():
            raise ValueError("HPL predicate-to-parent mapping is invalid")
        self.num_parent_prototypes = int(parent.size(0))
        self.pair_to_child = {
            str(key): int(value)
            for key, value in dict(manifest.get("pair_to_child", {})).items()
        }

        configured_counts = torch.as_tensor(
            manifest["predicate_counts"], dtype=torch.float32
        ).clamp_min(1.0)
        self.register_buffer("predicate_counts", configured_counts, persistent=True)
        manifest_tail_group = dict(manifest.get("tail_group", {}))
        manifest_tail_fraction = float(
            manifest_tail_group.get("fraction", self.tail_fraction)
        )
        if abs(manifest_tail_fraction - self.tail_fraction) > 1e-12:
            raise ValueError(
                "HPL.TAIL_FRACTION must match the train-only hierarchy: "
                f"config={self.tail_fraction}, manifest={manifest_tail_fraction}"
            )
        manifest_tail_ids = [
            int(value) for value in manifest_tail_group.get("predicate_ids", [])
        ]
        if not manifest_tail_ids:
            raise ValueError("HPL v3 hierarchy must persist train-only tail IDs")
        configured_tail_ids = [
            int(value) for value in hpl_cfg.get("TAIL_CLASS_IDS", [])
        ]
        if configured_tail_ids and configured_tail_ids != manifest_tail_ids:
            raise ValueError(
                "HPL.TAIL_CLASS_IDS must exactly match the train-only hierarchy: "
                f"config={configured_tail_ids}, manifest={manifest_tail_ids}"
            )
        tail_mask = torch.zeros(self.num_rel_classes, dtype=torch.bool)
        tail_mask[torch.as_tensor(manifest_tail_ids, dtype=torch.long)] = True
        self.register_buffer("tail_class_mask", tail_mask, persistent=True)
        self.register_buffer(
            "tail_class_ids",
            torch.as_tensor(manifest_tail_ids, dtype=torch.long),
            persistent=True,
        )
        self.tail_group_source = str(manifest_tail_group.get("source", "unknown"))

        self.encoder_hidden_dim = int(
            hpl_cfg.get("ENCODER_HIDDEN_DIM", feature_dim)
        )
        configured_constraint_dim = int(hpl_cfg.get("CONSTRAINT_DIM", 0))
        self.constraint_dim = (
            configured_constraint_dim
            if configured_constraint_dim > 0
            else self.feature_dim
        )
        self.selector_hidden_dim = int(hpl_cfg.get("SELECTOR_HIDDEN_DIM", 512))
        if self.constraint_dim != self.feature_dim and self.fr_enabled:
            raise ValueError(
                "HPL.CONSTRAINT_DIM must equal the relation feature dimension "
                "when FR is enabled so M@D can be added to F_int"
            )
        # Semantic prototypes are 600-D on STAR while RPCM relations are
        # 2048-D.  Zero padding is not described by HPL and leaves most
        # prototype coordinates structurally dead.  Use two explicit En
        # input branches with a shared latent contract instead.
        self.prototype_encoder = _TwoLayerEncoder(
            self.raw_prototype_dim, self.encoder_hidden_dim, self.constraint_dim
        )
        self.relation_encoder = _TwoLayerEncoder(
            self.feature_dim, self.encoder_hidden_dim, self.constraint_dim
        )
        dictionary_size = (
            child.size(0) + (self.num_rel_classes - 1) + self.num_parent_prototypes
        )
        self.prototype_selector = nn.Sequential(
            nn.Linear(self.feature_dim, self.selector_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.selector_hidden_dim, dictionary_size),
        )
        self._last_diagnostics: dict[str, object] = {}
        self._last_edge_diagnostics: dict[str, torch.Tensor] = {}

    def distance_matched_logits(
        self,
        refined: torch.Tensor,
        original_logits: torch.Tensor,
        *,
        anchor_detach: bool = True,
    ) -> torch.Tensor:
        """Return CC-consistent logits with an explicit background contract.

        The HPL hierarchy contains foreground predicates only.  Negative
        squared distances in the shared ``En(.)`` space are therefore matched
        per edge to the mean and RMS spread of Original RPCM's foreground
        logits.  This preserves the distance argmax, supplies a calibrated
        foreground/background comparison, and uses no labels or split-level
        statistics.  The Original background logit is retained unchanged.
        """

        if original_logits.ndim != 2 or original_logits.size(1) != self.num_rel_classes:
            raise ValueError(
                "Original HPL anchor logits must have shape [E, num_rel_classes]"
            )
        if refined.ndim != 2 or refined.size(0) != original_logits.size(0):
            raise ValueError(
                "HPL refined relation features must align with anchor logits"
            )
        if refined.size(1) != self.feature_dim:
            raise ValueError(
                f"HPL refined feature dim must be {self.feature_dim}, "
                f"got {refined.size(1)}"
            )
        if refined.numel() == 0:
            return original_logits

        raw_layers, _ = self._raw_hierarchy()
        encoded_relation = self.relation_encoder(refined)
        encoded_predicate = self.prototype_encoder(raw_layers[1])
        if self.cc_normalize:
            encoded_relation = F.normalize(encoded_relation, dim=1, eps=1e-8)
            encoded_predicate = F.normalize(encoded_predicate, dim=1, eps=1e-8)
        distance_logits = -torch.cdist(
            encoded_relation, encoded_predicate, p=2
        ).square()

        anchor_foreground = original_logits[:, 1:]
        if anchor_detach:
            anchor_foreground = anchor_foreground.detach()
        distance_centered = distance_logits - distance_logits.mean(
            dim=1, keepdim=True
        )
        anchor_mean = anchor_foreground.mean(dim=1, keepdim=True)
        anchor_centered = anchor_foreground - anchor_mean
        distance_scale = distance_centered.square().mean(
            dim=1, keepdim=True
        ).sqrt().clamp_min(1e-6)
        anchor_scale = anchor_centered.square().mean(
            dim=1, keepdim=True
        ).sqrt().clamp_min(1e-6)
        matched_foreground = (
            distance_centered / distance_scale * anchor_scale + anchor_mean
        )
        return torch.cat([original_logits[:, :1], matched_foreground], dim=1)

    def get_extra_state(self) -> dict[str, object]:
        return {
            "schema": HPL_MODULE_SCHEMA,
            "hierarchy_hash": self.hierarchy_hash,
            "tau": self.tau,
            "delta": self.delta,
            "beta": self.beta,
            "cc_normalize": self.cc_normalize,
            "cc_negative_mode": self.cc_negative_mode,
            "cc_negative_temperature": self.cc_negative_temperature,
            "tail_fraction": self.tail_fraction,
            "tail_class_ids": self.tail_class_ids.tolist(),
            "tail_group_source": self.tail_group_source,
            "eval_tail_route": self.eval_tail_route,
            "fr_enabled": self.fr_enabled,
            "hc_enabled": self.hc_enabled,
            "cc_enabled": self.cc_enabled,
            "encoder_hidden_dim": self.encoder_hidden_dim,
            "constraint_dim": self.constraint_dim,
            "selector_hidden_dim": self.selector_hidden_dim,
            "prototype_initialization": "raw_semantic_trainable",
            "constraint_encoder": "separate_input_branches_shared_latent",
            "cc_distance": (
                "normalized_squared_euclidean_" + self.cc_negative_mode
                if self.cc_normalize
                else "unnormalized_squared_euclidean_" + self.cc_negative_mode
            ),
            "upper_layer_update": "deterministic_bottom_up_kmeans",
        }

    def set_extra_state(self, state: Mapping[str, object]) -> None:
        if state.get("schema") != HPL_MODULE_SCHEMA:
            raise RuntimeError(
                "HPL checkpoint schema is incompatible with the corrected "
                f"paper implementation: checkpoint={state.get('schema')!r}, "
                f"expected={HPL_MODULE_SCHEMA!r}"
            )
        checkpoint_hash = str(state.get("hierarchy_hash", ""))
        if checkpoint_hash != self.hierarchy_hash:
            raise RuntimeError(
                "HPL checkpoint hierarchy hash does not match HIERARCHY_PATH: "
                f"checkpoint={checkpoint_hash}, manifest={self.hierarchy_hash}"
            )
        expected = self.get_extra_state()
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected.items()
            # Test-time tail routing is deliberately switchable on one fixed
            # checkpoint.  It has no parameter or training-state consequence,
            # so it must not make otherwise identical weights incompatible.
            if key not in {"hierarchy_hash", "eval_tail_route"}
            and state.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"HPL checkpoint configuration mismatch: {mismatches}")

    @staticmethod
    def _group_centroids(
        values: torch.Tensor, assignments: torch.Tensor, num_groups: int
    ) -> torch.Tensor:
        if values.ndim != 2 or assignments.ndim != 1:
            raise ValueError("prototype values/assignments have invalid rank")
        if values.size(0) != assignments.numel():
            raise ValueError("prototype values/assignments have inconsistent size")
        if num_groups <= 0:
            raise ValueError("prototype group count must be positive")
        output = values.new_zeros((int(num_groups), values.size(1)))
        output.index_add_(0, assignments, values)
        counts = torch.bincount(assignments, minlength=int(num_groups)).to(
            device=values.device, dtype=values.dtype
        )
        if (counts <= 0).any():
            raise RuntimeError("HPL hierarchy contains an empty prototype cluster")
        return output / counts.unsqueeze(1)

    @classmethod
    def _bottom_up_kmeans(
        cls,
        values: torch.Tensor,
        initial_assignments: torch.Tensor,
        num_groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic bottom-up K-means update disclosed in Section III-A.

        Assignment decisions are discrete and detached, while the returned
        centroids remain differentiable averages of the assigned prototypes.
        Empty groups are repaired by moving the worst-fit point from a group
        that has more than one member, with tensor order resolving ties.
        """

        assignments = initial_assignments.to(device=values.device).clone()
        if assignments.shape != (values.size(0),):
            raise ValueError("HPL K-means initialization has invalid shape")
        rows = torch.arange(values.size(0), device=values.device)
        for _ in range(100):
            centers = cls._group_centroids(values, assignments, num_groups)
            with torch.no_grad():
                distances = torch.cdist(
                    values.detach().float(), centers.detach().float(), p=2
                ).square()
                updated = distances.argmin(dim=1)
                counts = torch.bincount(updated, minlength=num_groups)
                for empty_group in torch.where(counts == 0)[0].tolist():
                    movable = counts[updated] > 1
                    if not movable.any():
                        raise RuntimeError("HPL K-means cannot repair an empty group")
                    fit_error = distances[rows, updated].masked_fill(
                        ~movable, -1.0
                    )
                    moved_row = int(fit_error.argmax())
                    old_group = int(updated[moved_row])
                    updated[moved_row] = int(empty_group)
                    counts[old_group] -= 1
                    counts[int(empty_group)] += 1
            if torch.equal(updated, assignments):
                assignments = updated
                break
            assignments = updated
        return assignments, cls._group_centroids(values, assignments, num_groups)

    def _raw_hierarchy(
        self,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        predicate = self._group_centroids(
            self.child_prototypes,
            self.child_to_predicate - 1,
            self.num_rel_classes - 1,
        )
        active_predicate_to_parent, parent = self._bottom_up_kmeans(
            predicate,
            self.predicate_to_parent,
            self.num_parent_prototypes,
        )
        return (self.child_prototypes, predicate, parent), active_predicate_to_parent

    def _raw_layers(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layers, _ = self._raw_hierarchy()
        return layers

    def _tail_rows(
        self,
        labels: torch.Tensor | None,
        preliminary_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.training:
            if labels is None:
                raise ValueError("HPL training requires relationship labels for tail FR")
            valid = (labels >= 0) & (labels < self.num_rel_classes)
            rows = torch.zeros_like(valid)
            rows[valid] = self.tail_class_mask[labels[valid]]
            return rows
        if self.diagnostic_eval_route == "pre_fr":
            if preliminary_logits is not None:
                return torch.zeros(
                    preliminary_logits.size(0),
                    dtype=torch.bool,
                    device=preliminary_logits.device,
                )
            if labels is not None:
                return torch.zeros_like(labels, dtype=torch.bool)
            raise ValueError("HPL pre-FR diagnostic requires logits or labels")
        if self.diagnostic_eval_route == "all":
            if preliminary_logits is not None:
                return torch.ones(
                    preliminary_logits.size(0),
                    dtype=torch.bool,
                    device=preliminary_logits.device,
                )
            if labels is not None:
                return torch.ones_like(labels, dtype=torch.bool)
            raise ValueError("HPL FR-all diagnostic requires logits or labels")
        if self.diagnostic_eval_route == "gt_tail":
            if labels is None:
                raise ValueError(
                    "HPL GT-tail oracle requires diagnostic evaluation labels"
                )
            valid = (labels >= 0) & (labels < self.num_rel_classes)
            rows = torch.zeros_like(valid)
            rows[valid] = self.tail_class_mask[labels[valid]]
            return rows
        if preliminary_logits is None:
            raise ValueError("HPL evaluation requires preliminary relationship logits")
        # The paper does not disclose its oracle-free test-time implementation.
        # Keep each completion explicit so a single checkpoint can localize
        # whether including background in the routing argmax causes a
        # self-locking failure for tail relations.  None of these paths reads
        # evaluation targets.
        if self.eval_tail_route == "base_prediction":
            return self.tail_class_mask[preliminary_logits.argmax(dim=1)]
        if self.eval_tail_route == "foreground_prediction":
            foreground_prediction = preliminary_logits[:, 1:].argmax(dim=1) + 1
            return self.tail_class_mask[foreground_prediction]
        if self.eval_tail_route == "soft_probability":
            probabilities = preliminary_logits.softmax(dim=1)
            return probabilities[:, self.tail_class_mask].sum(dim=1)
        raise AssertionError(f"unreachable HPL tail route: {self.eval_tail_route}")

    def refine(
        self,
        features: torch.Tensor,
        *,
        labels: torch.Tensor | None,
        preliminary_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.numel() == 0:
            self._last_edge_diagnostics = {
                "hpl_refined_mask": features.new_zeros((0,), dtype=torch.bool),
                "hpl_tail_gate": features.new_zeros((0,)),
                "hpl_attention_entropy": features.new_zeros((0,)),
                "hpl_residual_norm_ratio": features.new_zeros((0,)),
            }
            return features, features.new_zeros((0,), dtype=torch.bool)
        if not self.fr_enabled:
            self._last_edge_diagnostics = {
                "hpl_refined_mask": features.new_zeros(
                    (features.size(0),), dtype=torch.bool
                ),
                "hpl_tail_gate": features.new_zeros((features.size(0),)),
                "hpl_attention_entropy": features.new_zeros((features.size(0),)),
                "hpl_residual_norm_ratio": features.new_zeros((features.size(0),)),
            }
            return features, features.new_zeros(
                (features.size(0),), dtype=torch.bool, device=features.device
            )
        # Equation in Section III-B: r_t = M D and F~_t = F_t + r_t.
        # D is the continuously updated hierarchy itself.  Encoding D here
        # would mix the later HC/CC metric space into FR and leave the residual
        # magnitude unidentified under cosine-normalized constraints.
        # Project the 600-D semantic hierarchy into F_int through the explicit
        # prototype branch of En(.); never manufacture 2048-D prototypes by
        # appending constant zeros.
        raw_layers, _ = self._raw_hierarchy()
        dictionary = torch.cat(
            tuple(self.prototype_encoder(layer) for layer in raw_layers), dim=0
        )
        attention = F.softmax(self.prototype_selector(features), dim=1)
        residual = attention @ dictionary
        tail_route = self._tail_rows(labels, preliminary_logits)
        tail_gate = tail_route.to(features.dtype)
        refined = features + residual * tail_gate.unsqueeze(1)
        # A hard route remains exactly boolean.  For the soft completion, the
        # mask is only a human-readable majority-tail diagnostic; the exact
        # continuous value is exported separately as ``hpl_tail_gate``.
        tail_rows = (
            tail_route
            if tail_route.dtype == torch.bool
            else tail_gate > 0.5
        )
        attention_entropy = -(
            attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()
        ).sum(dim=1)
        residual_norm_ratio = residual.norm(dim=1) / features.norm(
            dim=1
        ).clamp_min(1e-12)
        self._last_edge_diagnostics = {
            "hpl_refined_mask": tail_rows.detach(),
            "hpl_tail_gate": tail_gate.detach(),
            "hpl_attention_entropy": attention_entropy.detach(),
            "hpl_residual_norm_ratio": residual_norm_ratio.detach(),
        }
        if (not self.training) and self.matching_diagnostic_enabled:
            # The paper's workflow ends with relation/prototype matching, but
            # the runnable reproduction currently classifies with Original
            # RPCM's independent predicate prototypes.  Export both natural
            # metrics in the shared En(.) space without changing predictions:
            # cosine (the HC space) and negative squared distance (the CC
            # space).  Background is intentionally absent from the HPL tree,
            # so these are foreground-only diagnostic scores.
            raw_layers, _ = self._raw_hierarchy()
            encoded_relation = self.relation_encoder(refined)
            encoded_predicate = self.prototype_encoder(raw_layers[1])
            cosine = (
                F.normalize(encoded_relation, dim=1, eps=1e-8)
                @ F.normalize(encoded_predicate, dim=1, eps=1e-8).t()
            )
            negative_squared_distance = -torch.cdist(
                encoded_relation, encoded_predicate, p=2
            ).square()
            self._last_edge_diagnostics.update(
                {
                    "hpl_matching_cosine_labels": (
                        cosine.argmax(dim=1).add(1).detach()
                    ),
                    "hpl_matching_distance_labels": (
                        negative_squared_distance.argmax(dim=1).add(1).detach()
                    ),
                }
            )
        self._last_diagnostics = {
            "module_schema": HPL_MODULE_SCHEMA,
            "edges": int(features.size(0)),
            "refined_edges": int(tail_rows.sum().detach().cpu()),
            "mean_tail_gate": float(tail_gate.mean().detach().cpu()),
            "dictionary_size": int(dictionary.size(0)),
            "mean_feature_norm": float(features.norm(dim=1).mean().detach().cpu()),
            "mean_dictionary_norm": float(
                dictionary.norm(dim=1).mean().detach().cpu()
            ),
            "mean_residual_norm": float(residual.norm(dim=1).mean().detach().cpu()),
            "mean_attention_entropy": float(
                attention_entropy.mean().detach().cpu()
            ),
            "mean_attention_effective_prototypes": float(
                attention_entropy.exp().mean().detach().cpu()
            ),
            "mean_residual_norm_ratio": float(
                residual_norm_ratio.mean().detach().cpu()
            ),
            "diagnostic_eval_route": self.diagnostic_eval_route,
        }
        return refined, tail_rows

    def edge_diagnostics(self) -> dict[str, torch.Tensor]:
        """Return per-edge inference diagnostics without changing predictions."""

        return dict(self._last_edge_diagnostics)

    def _constraint_layers(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map every hierarchy layer with the shared paper En(.) network."""

        layers, _ = self._constraint_hierarchy()
        return layers

    def _constraint_hierarchy(
        self,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        raw_layers, active_predicate_to_parent = self._raw_hierarchy()
        encoded = tuple(self.prototype_encoder(layer) for layer in raw_layers)
        return encoded, active_predicate_to_parent

    def _cc_negative_distance(
        self,
        distances: torch.Tensor,
        negative_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select/reduce legal c' != c distances under an explicit policy."""

        has_negative = negative_rows.any(dim=1)
        if not has_negative.any():
            return distances.new_zeros((0,)), has_negative
        values = distances[has_negative]
        mask = negative_rows[has_negative]
        if self.cc_negative_mode == "random":
            sampled = torch.multinomial(mask.to(values.dtype), 1).squeeze(1)
            rows = torch.arange(values.size(0), device=values.device)
            return values[rows, sampled], has_negative
        if self.cc_negative_mode == "all_mean":
            return (
                (values * mask.to(values.dtype)).sum(dim=1)
                / mask.sum(dim=1).clamp_min(1).to(values.dtype),
                has_negative,
            )
        masked = values.masked_fill(~mask, torch.inf)
        if self.cc_negative_mode == "hardest":
            return masked.min(dim=1).values, has_negative
        temperature = self.cc_negative_temperature
        negative_logits = (-values / temperature).masked_fill(~mask, -torch.inf)
        # Smooth minimum normalized by the number of eligible prototypes, so
        # duplicating an identical negative does not shift the loss.
        log_count = mask.sum(dim=1).to(values.dtype).log()
        smooth_min = -temperature * (
            torch.logsumexp(negative_logits, dim=1) - log_count
        )
        return smooth_min, has_negative

    def _hc_loss(
        self,
        constraint_layers: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
        active_predicate_to_parent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.hc_enabled:
            return self.child_prototypes.sum() * 0.0
        if constraint_layers is None:
            constraint_layers, active_predicate_to_parent = (
                self._constraint_hierarchy()
            )
        if active_predicate_to_parent is None:
            raise ValueError("HPL HC requires active parent assignments")
        layers = tuple(
            F.normalize(layer, dim=1, eps=1e-8) for layer in constraint_layers
        )
        losses: list[torch.Tensor] = []
        child_parent_rows = self.child_to_predicate - 1
        for layer_id, layer in enumerate(layers):
            for prototype_id in range(layer.size(0)):
                same_mask = torch.ones(
                    layer.size(0), dtype=torch.bool, device=layer.device
                )
                same_mask[prototype_id] = False
                negatives = [layer[same_mask]]
                if layer_id == 0:
                    parent_id = int(child_parent_rows[prototype_id])
                    parent_mask = torch.ones(
                        layers[1].size(0), dtype=torch.bool, device=layer.device
                    )
                    if 0 <= parent_id < parent_mask.numel():
                        parent_mask[parent_id] = False
                    negatives.append(layers[1][parent_mask])
                elif layer_id == 1:
                    parent_id = int(active_predicate_to_parent[prototype_id])
                    parent_mask = torch.ones(
                        layers[2].size(0), dtype=torch.bool, device=layer.device
                    )
                    if 0 <= parent_id < parent_mask.numel():
                        parent_mask[parent_id] = False
                    negatives.append(layers[2][parent_mask])
                valid_negatives = [value for value in negatives if value.numel() > 0]
                if not valid_negatives:
                    continue
                negative = torch.cat(valid_negatives, dim=0)
                similarity = negative @ layer[prototype_id]
                losses.append(F.softplus(similarity / self.tau))
        if not losses:
            return self.child_prototypes.sum() * 0.0
        return torch.cat(losses, dim=0).mean()

    def _cc_loss(
        self,
        refined: torch.Tensor,
        labels: torch.Tensor | None,
        child_targets: torch.Tensor | None,
        constraint_layers: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
        active_predicate_to_parent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.cc_enabled or labels is None or labels.numel() == 0:
            return refined.sum() * 0.0
        valid = (labels > 0) & (labels < self.num_rel_classes)
        if not valid.any():
            return refined.sum() * 0.0
        # HPL repeatedly describes both relationship representations and
        # prototypes on a hypersphere.  Delta/sum_c n_c^-beta naturally has
        # the same order as squared distance in [0,4] only after normalization.
        relation_raw = self.relation_encoder(refined[valid])
        relation = (
            F.normalize(relation_raw, dim=1, eps=1e-8)
            if self.cc_normalize
            else relation_raw
        )
        valid_labels = labels[valid]
        predicate_targets = valid_labels - 1
        if child_targets is None:
            valid_child_targets = valid_labels.new_full(valid_labels.shape, -1)
        else:
            if child_targets.shape != labels.shape:
                raise ValueError("HPL child targets and labels must have the same shape")
            valid_child_targets = child_targets.to(labels.device)[valid]

        inverse = self.predicate_counts[1:].pow(-self.beta)
        margins = self.delta * inverse / inverse.sum().clamp_min(1e-12)
        sample_margins = margins[predicate_targets]
        if constraint_layers is None:
            constraint_layers, active_predicate_to_parent = (
                self._constraint_hierarchy()
            )
        if active_predicate_to_parent is None:
            raise ValueError("HPL CC requires active parent assignments")
        parent_targets = active_predicate_to_parent[predicate_targets]
        terms: list[torch.Tensor] = []
        positive_diagnostics: list[torch.Tensor] = []
        negative_diagnostics: list[torch.Tensor] = []
        hinge_diagnostics: list[torch.Tensor] = []
        prototype_norms: list[torch.Tensor] = []
        for layer_id, (prototypes, targets) in enumerate(
            zip(
                constraint_layers,
                (valid_child_targets, predicate_targets, parent_targets),
            )
        ):
            target_valid = (targets >= 0) & (targets < prototypes.size(0))
            if not target_valid.any():
                continue
            relation_layer = relation[target_valid]
            target_layer = targets[target_valid]
            prototype_norms.append(prototypes.norm(dim=1).detach())
            if self.cc_normalize:
                prototypes = F.normalize(prototypes, dim=1, eps=1e-8)
            distances = torch.cdist(relation_layer, prototypes, p=2).square()
            rows = torch.arange(target_layer.numel(), device=targets.device)
            positive = distances[rows, target_layer]
            if layer_id == 0:
                # Sibling children belong to the same relationship category;
                # Eq. (2) defines negatives only for c != c'.
                negative_mask = self.child_to_predicate.unsqueeze(0).eq(
                    valid_labels[target_valid].unsqueeze(1)
                )
            else:
                negative_mask = F.one_hot(
                    target_layer, num_classes=prototypes.size(0)
                ).bool()
            negative_rows = ~negative_mask
            negative, has_negative = self._cc_negative_distance(
                distances, negative_rows
            )
            if has_negative.any():
                hinge = (
                    positive[has_negative]
                    - negative
                    + sample_margins[target_valid][has_negative]
                )
                terms.append(F.relu(hinge))
                positive_diagnostics.append(positive[has_negative].detach())
                negative_diagnostics.append(negative.detach())
                hinge_diagnostics.append(hinge.detach())
        if not terms:
            return refined.sum() * 0.0
        positive_values = torch.cat(positive_diagnostics)
        negative_values = torch.cat(negative_diagnostics)
        hinge_values = torch.cat(hinge_diagnostics)
        prototype_norm_values = torch.cat(prototype_norms)
        margins_detached = sample_margins.detach()
        relation_norm_values = relation_raw.norm(dim=1).detach()

        def quantiles(values: torch.Tensor) -> list[float]:
            result = torch.quantile(
                values.float(),
                values.new_tensor([0.05, 0.5, 0.95], dtype=torch.float32),
            )
            return [float(value) for value in result.cpu().tolist()]

        self._last_diagnostics.update(
            {
                "cc_normalize": self.cc_normalize,
                "cc_negative_mode": self.cc_negative_mode,
                "cc_relation_norm_mean": float(
                    relation_norm_values.mean().cpu()
                ),
                "cc_relation_norm_p05_p50_p95": quantiles(
                    relation_norm_values
                ),
                "cc_prototype_norm_mean": float(
                    prototype_norm_values.mean().cpu()
                ),
                "cc_prototype_norm_p05_p50_p95": quantiles(
                    prototype_norm_values
                ),
                "cc_positive_distance_mean": float(
                    positive_values.mean().cpu()
                ),
                "cc_positive_distance_p05_p50_p95": quantiles(
                    positive_values
                ),
                "cc_negative_distance_mean": float(
                    negative_values.mean().cpu()
                ),
                "cc_negative_distance_p05_p50_p95": quantiles(
                    negative_values
                ),
                "cc_margin_min": float(margins_detached.min().cpu()),
                "cc_margin_mean": float(margins_detached.mean().cpu()),
                "cc_margin_max": float(margins_detached.max().cpu()),
                "cc_hinge_active_ratio": float(
                    (hinge_values > 0).float().mean().cpu()
                ),
            }
        )
        return torch.cat(terms, dim=0).mean()

    def child_targets_for_pairs(
        self,
        labels: torch.Tensor | None,
        proposals: Sequence,
        rel_pair_idxs: Sequence[torch.Tensor],
    ) -> torch.Tensor | None:
        """Map a supervised relation instance to its bottom child prototype."""

        if labels is None:
            return None
        if len(proposals) != len(rel_pair_idxs):
            raise ValueError("HPL proposals and relation-pair lists differ in length")
        targets: list[int] = []
        label_offset = 0
        for proposal, pairs in zip(proposals, rel_pair_idxs):
            if not proposal.has_field("labels"):
                raise ValueError("HPL requires object labels for child matching")
            object_labels = proposal.get_field("labels").long().detach().cpu()
            for subject_id, object_id in pairs.long().detach().cpu().tolist():
                if not (0 <= subject_id < len(object_labels)) or not (
                    0 <= object_id < len(object_labels)
                ):
                    raise ValueError("HPL relation pair contains an invalid endpoint")
                if label_offset >= labels.numel():
                    raise ValueError("HPL has fewer relation labels than pairs")
                predicate = int(labels[label_offset].detach().cpu())
                targets.append(
                    self.pair_to_child.get(
                        f"{predicate}:{int(object_labels[subject_id])}:"
                        f"{int(object_labels[object_id])}",
                        -1,
                    )
                )
                label_offset += 1
        if label_offset != labels.numel():
            raise ValueError("HPL relation labels and relation pairs differ in size")
        return torch.as_tensor(targets, dtype=torch.long, device=labels.device)

    def losses(
        self,
        refined: torch.Tensor,
        labels: torch.Tensor | None,
        child_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.hc_enabled or self.cc_enabled:
            constraint_layers, active_predicate_to_parent = (
                self._constraint_hierarchy()
            )
        else:
            constraint_layers, active_predicate_to_parent = None, None
        return {
            "loss_hpl_hc": self._hc_loss(
                constraint_layers, active_predicate_to_parent
            ),
            "loss_hpl_cc": self._cc_loss(
                refined,
                labels,
                child_targets,
                constraint_layers,
                active_predicate_to_parent,
            ),
        }

    def diagnostics(self) -> dict[str, object]:
        return {
            **self._last_diagnostics,
            "hierarchy_hash": self.hierarchy_hash,
            "children": int(self.child_prototypes.size(0)),
            "predicates": int(self.num_rel_classes - 1),
            "parents": int(self.num_parent_prototypes),
            "tail_classes": int(self.tail_class_mask.sum()),
            "upper_layers": "deterministic_bottom_up_kmeans",
            "fr_dictionary_space": "prototype_En_to_Fint_width",
            "constraint_space": "separate_input_En_branches_shared_latent",
            "cc_distance": (
                ("normalized_" if self.cc_normalize else "unnormalized_")
                + "squared_euclidean_"
                + self.cc_negative_mode
            ),
            "tail_group_source": self.tail_group_source,
            "tail_class_ids": self.tail_class_ids.tolist(),
            "eval_tail_route": self.eval_tail_route,
        }


class HPLRPCM(RPCMSGGToolkitOriginal):
    """Original RPCM plus the paper's HPL FR/HC/CC mechanisms."""

    def __init__(self, cfg: dict, in_channels: int):
        super().__init__(cfg, in_channels)
        hierarchy_path = str(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "HPL",
                "HIERARCHY_PATH",
                default="",
            )
        ).strip()
        if not hierarchy_path:
            raise FileNotFoundError(
                "HPL_RPCM requires MODEL.ROI_RELATION_HEAD.HPL.HIERARCHY_PATH; "
                "build it from STAR TRAIN with tools/build_hpl_hierarchy.py"
            )
        manifest = load_hpl_hierarchy_manifest(
            hierarchy_path, num_rel_classes=self.num_rel_classes
        )
        self.hpl = HPLPaperModule(
            cfg, self.mlp_dim, self.num_rel_classes, manifest
        )
        hpl_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]["HPL"]
        self.hpl_keep_rpcm_regularizers = bool(
            hpl_cfg.get("KEEP_RPCM_REGULARIZERS", False)
        )
        self.hpl_final_logit_mode = str(
            hpl_cfg.get("FINAL_LOGIT_MODE", "original")
        ).strip().lower()
        if self.hpl_final_logit_mode not in {
            "original",
            "distance",
            "residual_fusion",
        }:
            raise ValueError(
                "HPL.FINAL_LOGIT_MODE must be original, distance or "
                f"residual_fusion, got {self.hpl_final_logit_mode!r}"
            )
        self.hpl_final_fusion_weight = float(
            hpl_cfg.get("FINAL_FUSION_WEIGHT", 0.0)
        )
        if not 0.0 <= self.hpl_final_fusion_weight <= 1.0:
            raise ValueError("HPL.FINAL_FUSION_WEIGHT must be in [0, 1]")
        self.hpl_final_anchor_detach = bool(
            hpl_cfg.get("FINAL_ANCHOR_DETACH", True)
        )
        self.allow_eval_relation_labels = (
            self.hpl.diagnostic_eval_route == "gt_tail"
        )
        print(
            "[HPL_RPCM] "
            f"hierarchy={self.hpl.hierarchy_hash}, "
            f"children={self.hpl.child_prototypes.size(0)}, "
            f"parents={self.hpl.num_parent_prototypes}, "
            f"module={HPL_MODULE_SCHEMA}, "
            "FR=M@En(D)_in_Fint, HC=cosine, "
            f"CC={self.hpl.diagnostics()['cc_distance']}, "
            f"keep_rpcm_regularizers={self.hpl_keep_rpcm_regularizers}, "
            f"final_logits={self.hpl_final_logit_mode}, "
            f"fusion_weight={self.hpl_final_fusion_weight:g}",
            flush=True,
        )

    def _original_prototype_losses_enabled(self) -> bool:
        # Equation (5) replaces the baseline prototype regularizers with
        # focal + HC + CC.  The default preserves that paper contract; the
        # explicit A/B/C diagnostic can retain all five RPCM constraints to
        # localize whether removing them causes the reproduction collapse.
        return self.hpl_keep_rpcm_regularizers

    def _method_relation_logits(
        self,
        relation_logits: torch.Tensor,
        *,
        relation_representation: torch.Tensor,
        predicate_prototype: torch.Tensor,
    ) -> torch.Tensor:
        del predicate_prototype
        if self.hpl_final_logit_mode == "original":
            return relation_logits
        distance_logits = self.hpl.distance_matched_logits(
            relation_representation,
            relation_logits,
            anchor_detach=self.hpl_final_anchor_detach,
        )
        if self.hpl_final_logit_mode == "distance":
            return distance_logits
        weight = self.hpl_final_fusion_weight
        if weight == 0.0:
            return relation_logits
        if weight == 1.0:
            return distance_logits
        fused = relation_logits.clone()
        fused[:, 1:] = (
            relation_logits[:, 1:] * (1.0 - weight)
            + distance_logits[:, 1:] * weight
        )
        return fused

    def _preliminary_logits(
        self, rel_rep: torch.Tensor, predicate_proto: torch.Tensor
    ) -> torch.Tensor:
        # Called only when labels are unavailable (evaluation or explicit
        # predicted-tail training), so dropout follows the module's current
        # mode exactly.
        projected_rel = self.norm_rel_rep(
            self.dropout_rel_rep(F.relu(self.linear_rel_rep(rel_rep))) + rel_rep
        )
        projected_rel = self.project_head(
            self.dropout_rel(F.relu(projected_rel))
        )
        projected_proto = self.project_head(
            self.dropout_pred(F.relu(predicate_proto))
        )
        return (
            F.normalize(projected_rel, dim=1, eps=1e-8)
            @ F.normalize(projected_proto, dim=1, eps=1e-8).t()
        ) * self.logit_scale.exp()

    def _refine_relation_representation(
        self,
        rel_rep: torch.Tensor,
        predicate_proto: torch.Tensor,
        rel_labels: torch.Tensor | None,
        proposals: Sequence,
        rel_pair_idxs: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        needs_prediction = (
            not self.training
            and self.hpl.diagnostic_eval_route
            in {"current", "pre_fr", "all"}
        )
        preliminary = (
            self._preliminary_logits(rel_rep, predicate_proto)
            if needs_prediction
            else None
        )
        refined, _ = self.hpl.refine(
            rel_rep,
            labels=rel_labels,
            preliminary_logits=preliminary,
        )
        child_targets = (
            self.hpl.child_targets_for_pairs(rel_labels, proposals, rel_pair_idxs)
            if self.training
            else None
        )
        losses = (
            self.hpl.losses(refined, rel_labels, child_targets)
            if self.training
            else {}
        )
        return refined, losses

    def hpl_edge_diagnostics(self) -> dict[str, torch.Tensor]:
        return self.hpl.edge_diagnostics()


__all__ = [
    "HPL_HIERARCHY_SCHEMA",
    "HPL_MODULE_SCHEMA",
    "HPLPaperModule",
    "HPLRPCM",
    "build_hpl_hierarchy_manifest",
    "load_hpl_hierarchy_manifest",
    "save_hpl_hierarchy_manifest",
]
