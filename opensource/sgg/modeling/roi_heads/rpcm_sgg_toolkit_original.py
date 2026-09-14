"""Inference-faithful Original RPCM predictor and controlled graph ablations.

This module intentionally contains only the source-classifier path needed by
the frozen Original RPCM checkpoint.  It does not restore any of the archived
selector, HBA, or typed-residual development.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geo_ee import GeometryAwareEEAttention, OBBGeometryEncoder
from .relation_inference import obj_prediction_nms
from .roi_relation_predictors import (
    MLP,
    RPCMLegacy,
    _LegacyGCNLayer,
    _LegacyGraphConvolutionLayerCollect,
    _OriginalRPCMPairwiseFeatureExtractor,
    _SGGToolkitGraphConvolutionLayerCollect,
    _SGGToolkitGraphConvolutionLayerUpdate,
    _cfg_get,
    _load_rpcm_glove_table,
    _orig_encode_box_info,
    _orig_make_fc,
)


def _matrix_squared_euclidean_distance(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    left_sq = left.square().sum(dim=1, keepdim=True)
    right_sq = right.square().sum(dim=1).unsqueeze(0)
    return (left_sq + right_sq - 2.0 * (left @ right.t())).clamp_min(0.0)


def _build_sgg_toolkit_object_glove_init(
    names: Sequence[str],
    glove_path: str,
    embed_dim: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Reproduce SGG-ToolKit ``obj_edge_vectors`` literally."""

    table = _load_rpcm_glove_table(glove_path, embed_dim)
    vectors = torch.empty((len(names), embed_dim), dtype=torch.float32)
    vectors.normal_(0.0, 1.0)
    if table is None:
        return vectors, {"missing_classes": list(map(str, names))}
    word_to_idx, glove_vectors = table
    missing_classes: list[str] = []
    fallback_tokens: dict[str, str] = {}
    for idx, raw_name in enumerate(names):
        name = str(raw_name)
        word_idx = word_to_idx.get(name)
        if word_idx is not None:
            vectors[idx] = glove_vectors[word_idx]
            continue
        tokens = name.split(" ")
        longest = sorted(tokens, key=len, reverse=True)[0] if tokens else name
        fallback_tokens[name] = longest
        word_idx = word_to_idx.get(longest)
        if word_idx is not None:
            vectors[idx] = glove_vectors[word_idx]
        else:
            missing_classes.append(name)
    return vectors, {
        "missing_classes": missing_classes,
        "fallback_tokens": fallback_tokens,
    }


def _build_sgg_toolkit_relation_glove_init(
    names: Sequence[str],
    glove_path: str,
    embed_dim: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Reproduce SGG-ToolKit ``rel_vectors`` literally."""

    table = _load_rpcm_glove_table(glove_path, embed_dim)
    vectors = torch.empty((len(names), embed_dim), dtype=torch.float32)
    vectors.normal_(0.0, 1.0)
    if table is None:
        return vectors, {"missing_predicates": list(map(str, names[1:]))}
    word_to_idx, glove_vectors = table
    missing_predicates: list[str] = []
    missing_tokens: list[str] = []
    for idx, raw_name in enumerate(names):
        if idx == 0:
            continue
        name = str(raw_name)
        word_idx = word_to_idx.get(name)
        if word_idx is not None:
            vectors[idx] = glove_vectors[word_idx]
            continue
        token_vectors: list[torch.Tensor] = []
        for token in name.split(" "):
            token_idx = word_to_idx.get(token)
            if token_idx is None:
                missing_tokens.append(f"{name}:{token}")
            else:
                token_vectors.append(glove_vectors[token_idx])
        if token_vectors:
            vectors[idx] = torch.stack(token_vectors, dim=0).mean(dim=0)
        else:
            missing_predicates.append(name)
    return vectors, {
        "missing_predicates": missing_predicates,
        "missing_tokens": sorted(set(missing_tokens)),
    }


class GlobalEntitySourceFusion(nn.Module):
    """Three trainable global source logits with an exact Original identity."""

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def weights(self, target: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0).to(dtype=target.dtype).expand(
            target.size(0), -1
        )

    def forward(
        self,
        target: torch.Tensor,
        sources: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.weights(target)
        original = (sources[0] + sources[1] + sources[2]) / 3.0
        # At zero logits the correction is bitwise zero in target dtype, so
        # checkpoint initialization follows the exact Original arithmetic.
        correction = target.new_tensor(0.0)
        uniform = target.new_tensor(1.0 / 3.0)
        for source_id, source in enumerate(sources):
            correction = correction + (
                weights[:, source_id : source_id + 1] - uniform
            ) * source
        return original + correction, weights


class AdaptiveEntitySourceFusion(nn.Module):
    """Per-entity dynamic fusion over entity and two relation-role sources."""

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.input = nn.Linear(self.feature_dim * 4, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, 3)
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        # Required identity initialization.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def weights(
        self,
        target: torch.Tensor,
        sources: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        fusion_input = torch.cat((target, *sources), dim=-1)
        logits = self.output(F.relu(self.input(fusion_input)))
        return torch.softmax(logits, dim=-1)

    def forward(
        self,
        target: torch.Tensor,
        sources: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.weights(target, sources)
        original = (sources[0] + sources[1] + sources[2]) / 3.0
        correction = target.new_tensor(0.0)
        uniform = target.new_tensor(1.0 / 3.0)
        for source_id, source in enumerate(sources):
            correction = correction + (
                weights[:, source_id : source_id + 1] - uniform
            ) * source
        return original + correction, weights


class GlobalAnchoredAdaptiveEntitySourceFusion(nn.Module):
    """Per-entity residual routing constrained around a frozen global prior."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        anchor_weights: tuple[float, float, float],
        rho: float = 0.5,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.rho = float(rho)
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError(f"Anchored ASF-E rho must be in [0, 1], got {rho}")
        anchor = torch.as_tensor(anchor_weights, dtype=torch.float32)
        if anchor.shape != (3,) or not torch.isfinite(anchor).all() or (anchor <= 0).any():
            raise ValueError("Anchored ASF-E requires three finite positive anchor weights")
        anchor = anchor / anchor.sum()
        # The name intentionally matches GlobalEntitySourceFusion so the
        # Round-1 Global checkpoint loads its learned prior without remapping.
        self.logits = nn.Parameter(anchor.log(), requires_grad=False)
        self.input = nn.Linear(self.feature_dim * 4, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, 3)
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def anchor_weights(self, target: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0).to(dtype=target.dtype).expand(
            target.size(0), -1
        )

    def weights(
        self,
        target: torch.Tensor,
        sources: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        fusion_input = torch.cat((target, *sources), dim=-1)
        delta = self.output(F.relu(self.input(fusion_input)))
        anchor = self.anchor_weights(target)
        # q = softmax(log(w_global) + delta).  Subtracting the numerically
        # evaluated zero-residual q0 makes delta=0 exactly equal to the Global
        # checkpoint even when 1-D and 2-D softmax kernels round differently.
        log_anchor = anchor.clamp_min(1e-12).log()
        q = torch.softmax(log_anchor + delta, dim=-1)
        q0 = torch.softmax(log_anchor, dim=-1)
        return anchor + self.rho * (q - q0)

    def forward(
        self,
        target: torch.Tensor,
        sources: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.weights(target, sources)
        original = (sources[0] + sources[1] + sources[2]) / 3.0
        correction = target.new_tensor(0.0)
        uniform = target.new_tensor(1.0 / 3.0)
        for source_id, source in enumerate(sources):
            correction = correction + (
                weights[:, source_id : source_id + 1] - uniform
            ) * source
        return original + correction, weights

class RPCMSGGToolkitOriginal(RPCMLegacy):
    """Original SGG-ToolKit classifier with a selectable RPCM graph front end.

    ``RPCM_RELATION_GRAPH_MODE="sgg_toolkit"`` is the complete published
    implementation. ``unified`` and ``dual_view`` retain the current RCA graph
    front end while replacing only its post-GNN classifier with the source
    object/predicate 300-D GloVe embeddings, gated subject/object/union
    semantic fusion, projected cosine prototypes, per-forward KMeans coarse
    prototypes, and five auxiliary prototype losses.

    The current controlled Base experiment is PredCls-only.  Refusing other
    tasks here prevents the audit implementation from being mistaken for the
    separately maintained SGCls/SGDet compatibility route.
    """

    # R4 context-only relation rows are removed immediately before the
    # unchanged prototype/classifier head.  RelationHead uses this declaration
    # to avoid slicing the already query-only logits a second time.
    r4_filters_prediction_queries = True

    def __init__(self, cfg: dict, in_channels: int):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.in_channels = int(in_channels)
        self.num_obj_classes = int(cfg["MODEL"]["ROI_BOX_HEAD"]["NUM_CLASSES"])
        self.num_rel_classes = int(cfg["MODEL"]["ROI_RELATION_HEAD"]["NUM_CLASSES"])
        self.hidden_dim = int(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "CONTEXT_HIDDEN_DIM",
                default=512,
            )
        )
        self.pooling_dim = int(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "CONTEXT_POOLING_DIM",
                default=in_channels,
            )
        )
        self.mlp_dim = int(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RPCM_MLP_DIM",
                default=2048,
            )
        )
        self.embed_dim = int(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RPCM_PROTO_EMBED_DIM",
                default=300,
            )
        )
        self.feat_update_step = int(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RPCM_FEAT_UPDATE_STEP",
                default=4,
            )
        )
        self.Par = int(_cfg_get(cfg, "EXP_nums", default=30))
        self.relation_graph_mode = str(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RPCM_RELATION_GRAPH_MODE",
                default="sgg_toolkit",
            )
        ).lower()
        if self.relation_graph_mode not in {
            "sgg_toolkit",
            "unified",
            "dual_view",
        }:
            raise ValueError(
                "RPCM_SGG_TOOLKIT_ORIGINAL requires graph mode "
                "'sgg_toolkit', 'unified', or 'dual_view', got "
                f"{self.relation_graph_mode!r}."
            )
        self.context_ablation = str(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RPCM_CONTEXT_ABLATION",
                default="full",
            )
        ).lower()
        context_aliases = {
            "none": "full",
            "entity_mediated_only": "no_rel_rel",
            "no_rel_ent_loop": "rel_rel_only",
        }
        self.context_ablation = context_aliases.get(
            self.context_ablation, self.context_ablation
        )
        valid_context_ablations = {
            "full",
            "no_rel_rel",
            "no_rel_ent",
            "no_ent_rel",
            "no_ent_ent",
            "rel_rel_only",
        }
        if self.context_ablation not in valid_context_ablations:
            raise ValueError(
                "RPCM_CONTEXT_ABLATION must be one of "
                f"{sorted(valid_context_ablations)}, got "
                f"{self.context_ablation!r}."
            )
        self.relation_adjacency = str(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RPCM_ORIGINAL_REL_ADJACENCY",
                default="unified",
            )
        ).lower()
        adjacency_aliases = {
            "original": "unified",
            "all": "unified",
            "dual": "ss_oo",
            "four-role": "four_role",
            "fourrole": "four_role",
            "same-role": "rr_same_role",
            "same_role": "rr_same_role",
            "cross-role": "rr_cross_role",
            "cross_role": "rr_cross_role",
            "four-role-balanced": "rr_four_role_balanced",
            "four_role_balanced": "rr_four_role_balanced",
        }
        self.relation_adjacency = adjacency_aliases.get(
            self.relation_adjacency, self.relation_adjacency
        )
        if self.relation_adjacency not in {
            "unified",
            "ss_oo",
            "four_role",
            "rr_same_role",
            "rr_cross_role",
            "rr_four_role_balanced",
        }:
            raise ValueError(
                "RPCM_ORIGINAL_REL_ADJACENCY must be 'unified', 'ss_oo', "
                "'four_role', 'rr_same_role', 'rr_cross_role', or "
                f"'rr_four_role_balanced', got {self.relation_adjacency!r}."
            )
        self.entity_adjacency = str(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RPCM_ORIGINAL_ENTITY_ADJACENCY",
                default="complete",
            )
        ).strip().lower()
        entity_adjacency_aliases = {
            "original": "complete",
            "full": "complete",
            "relation-supported": "relation_induced",
            "relation_supported": "relation_induced",
            "relation-induced": "relation_induced",
            "off": "none",
            "no_ee": "none",
            "no-ee": "none",
        }
        self.entity_adjacency = entity_adjacency_aliases.get(
            self.entity_adjacency, self.entity_adjacency
        )
        if self.entity_adjacency not in {"complete", "relation_induced", "none"}:
            raise ValueError(
                "RPCM_ORIGINAL_ENTITY_ADJACENCY must be 'complete', "
                f"'relation_induced', or 'none', got {self.entity_adjacency!r}."
            )
        self.graph_backend = str(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RPCM_GRAPH_BACKEND",
                default="dense",
            )
        ).strip().lower()
        if self.graph_backend not in {"dense", "incidence"}:
            raise ValueError(
                "RPCM_GRAPH_BACKEND must be 'dense' or 'incidence', got "
                f"{self.graph_backend!r}."
            )
        self.rel_entity_semantic_alpha = float(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RPCM_REL_ENTITY_SEMANTIC_ALPHA",
                default=1.0,
            )
        )
        if not math.isfinite(self.rel_entity_semantic_alpha) or not (
            0.0 <= self.rel_entity_semantic_alpha <= 1.0
        ):
            raise ValueError(
                "RPCM_REL_ENTITY_SEMANTIC_ALPHA must be finite and in [0, 1], "
                f"got {self.rel_entity_semantic_alpha!r}."
            )
        self.rel_entity_semantic_gate_mode = str(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RPCM_REL_ENTITY_SEMANTIC_GATE_MODE",
                default="global",
            )
        ).strip().lower().replace("-", "_")
        gate_aliases = {
            "all": "global",
            "high": "high_only",
            "low": "low_only",
        }
        self.rel_entity_semantic_gate_mode = gate_aliases.get(
            self.rel_entity_semantic_gate_mode,
            self.rel_entity_semantic_gate_mode,
        )
        if self.rel_entity_semantic_gate_mode not in {
            "global",
            "high_only",
            "low_only",
        }:
            raise ValueError(
                "RPCM_REL_ENTITY_SEMANTIC_GATE_MODE must be 'global', "
                f"'high_only', or 'low_only', got "
                f"{self.rel_entity_semantic_gate_mode!r}."
            )
        self.rel_entity_high_hhi_threshold = float(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RPCM_REL_ENTITY_HIGH_HHI_THRESHOLD",
                default=0.588613406795225,
            )
        )
        if not math.isfinite(self.rel_entity_high_hhi_threshold) or not (
            0.0 <= self.rel_entity_high_hhi_threshold <= 1.0
        ):
            raise ValueError(
                "RPCM_REL_ENTITY_HIGH_HHI_THRESHOLD must be finite and in "
                f"[0, 1], got {self.rel_entity_high_hhi_threshold!r}."
            )
        self.entity_source_fusion_mode = str(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RPCM_ENTITY_SOURCE_FUSION",
                default="original",
            )
        ).strip().lower()
        fusion_aliases = {
            "none": "original",
            "fixed": "original",
            "global_weight": "global",
            "adaptive": "asf_e",
            "asf-e": "asf_e",
            "anchored": "anchored_asf_e",
            "anchored-asf-e": "anchored_asf_e",
            "global_anchored": "anchored_asf_e",
        }
        self.entity_source_fusion_mode = fusion_aliases.get(
            self.entity_source_fusion_mode, self.entity_source_fusion_mode
        )
        if self.entity_source_fusion_mode not in {
            "original", "global", "asf_e", "anchored_asf_e"
        }:
            raise ValueError(
                "RPCM_ENTITY_SOURCE_FUSION must be 'original', 'global', "
                f"'asf_e', or 'anchored_asf_e', got {self.entity_source_fusion_mode!r}."
            )
        self.source_fusion_record_stats = bool(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RPCM_SOURCE_FUSION_RECORD_STATS",
                default=False,
            )
        )
        self._source_weight_stats: dict[int, dict[str, object]] = {}
        self.relation_entity_probe_enabled = bool(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RPCM_RELATION_ENTITY_PROBE_ENABLED",
                default=False,
            )
        )
        self.qg_full_relation_context = bool(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RPCM_QG_FULL_RELATION_CONTEXT",
                default=False,
            )
        )
        self._relation_entity_probe_gate: torch.Tensor | None = None
        self._relation_entity_probe_num_rels: tuple[int, ...] = ()
        if self.relation_graph_mode != "sgg_toolkit" and (
            self.context_ablation != "full"
            or self.relation_adjacency != "unified"
        ):
            raise ValueError(
                "Original RPCM context diagnostics require "
                "RPCM_RELATION_GRAPH_MODE='sgg_toolkit'."
            )
        self.rel_subj_view_enabled = bool(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RPCM_REL_SUBJ_VIEW_ENABLED",
                default=True,
            )
        )
        self.rel_obj_view_enabled = bool(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RPCM_REL_OBJ_VIEW_ENABLED",
                default=True,
            )
        )
        if self.relation_graph_mode == "dual_view" and not (
            self.rel_subj_view_enabled or self.rel_obj_view_enabled
        ):
            raise ValueError(
                "dual_view requires at least one enabled relation view."
            )
        self.exact_6850 = bool(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RPCM_LEGACY_6850_EXACT",
                default=False,
            )
        )
        use_gt_box = bool(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "USE_GT_BOX",
                default=False,
            )
        )
        self.use_gt_object_label = bool(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "USE_GT_OBJECT_LABEL",
                default=False,
            )
        )
        if use_gt_box:
            self.mode = "predcls" if self.use_gt_object_label else "sgcls"
        else:
            self.mode = "sgdet"
        self.nms_thresh = float(
            _cfg_get(
                cfg,
                "TEST",
                "RELATION",
                "LATER_NMS_PREDICTION_THRES",
                default=0.3,
            )
        )
        graph_dim = self.mlp_dim * 2
        if self.pooling_dim != graph_dim or self.in_channels != graph_dim:
            raise ValueError(
                "Published RPCM requires in_channels == CONTEXT_POOLING_DIM "
                f"== 2 * RPCM_MLP_DIM, got {self.in_channels}, "
                f"{self.pooling_dim}, and {graph_dim}."
            )
        if not 2 <= self.Par <= self.num_rel_classes:
            raise ValueError(
                f"EXP_nums must be in [2, {self.num_rel_classes}], got {self.Par}."
            )

        self.obj_classes = list(
            _cfg_get(cfg, "MODEL", "ROI_BOX_HEAD", "CLASS_NAMES", default=[])
        )
        self.rel_classes = list(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RELATION_NAMES",
                default=[],
            )
        )
        if len(self.obj_classes) < self.num_obj_classes:
            self.obj_classes += [
                f"class_{idx}"
                for idx in range(len(self.obj_classes), self.num_obj_classes)
            ]
        if len(self.rel_classes) < self.num_rel_classes:
            self.rel_classes += [
                f"relation_{idx}"
                for idx in range(len(self.rel_classes), self.num_rel_classes)
            ]

        dropout_p = float(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RPCM_DROPOUT",
                default=0.2,
            )
        )
        self.post_emb = nn.Linear(self.in_channels, self.mlp_dim * 2)

        # Preserve the source constructor order because scratch experiments
        # depend on its random/GloVe initialization sequence.
        self.pairwise_feature_extractor = _OriginalRPCMPairwiseFeatureExtractor(
            cfg, in_channels
        )
        proto_glove_path = str(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "RPCM_PROTO_GLOVE_PATH",
                default="",
            )
        )
        obj_embed_vecs, obj_glove_diag = _build_sgg_toolkit_object_glove_init(
            self.obj_classes[: self.num_obj_classes],
            proto_glove_path,
            self.embed_dim,
        )
        rel_embed_vecs, rel_glove_diag = _build_sgg_toolkit_relation_glove_init(
            self.rel_classes[: self.num_rel_classes],
            proto_glove_path,
            self.embed_dim,
        )
        self.obj_embed = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs)
            self.rel_embed.weight.copy_(rel_embed_vecs)

        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.gate_sub = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_obj = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.vis2sem = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(self.mlp_dim * 2, self.mlp_dim),
        )

        self.project_head = MLP(
            self.mlp_dim, self.mlp_dim, self.mlp_dim * 2, 2
        )
        # These source modules are unused by the published forward but remain
        # registered so the module/state layout faithfully records the model.
        self.project_head2 = MLP(
            self.mlp_dim, self.mlp_dim, self.mlp_dim * 2, 2
        )
        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_pred = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep2 = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep2 = nn.LayerNorm(self.mlp_dim)
        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)
        self.dropout_rel_rep = nn.Dropout(dropout_p)
        self.dropout_rel_rep2 = nn.Dropout(dropout_p)
        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_rel2 = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)
        self.down_samp = MLP(
            self.pooling_dim, self.mlp_dim, self.mlp_dim, 2
        )
        self.logit_scale = nn.Parameter(
            torch.ones((), dtype=torch.float32) * math.log(1.0 / 0.07)
        )

        self.pos_embed = nn.Sequential(
            nn.Linear(9, 32),
            nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
        )
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs)
        self.out_obj = _orig_make_fc(self.hidden_dim, self.num_obj_classes)
        self.lin_obj_cyx = _orig_make_fc(
            self.in_channels + self.embed_dim + 128, self.hidden_dim
        )

        if self.feat_update_step > 0 and self.relation_graph_mode == "sgg_toolkit":
            self.gcn_collect_feat = _SGGToolkitGraphConvolutionLayerCollect(
                graph_dim, graph_dim
            )
            self.gcn_update_feat = _SGGToolkitGraphConvolutionLayerUpdate()
        elif self.feat_update_step > 0:
            self.gcn_ent2ent = nn.ModuleList()
            self.gcn_ent2rel = nn.ModuleList()
            self.gcn_rel2rel = nn.ModuleList()
            for _ in range(self.feat_update_step):
                self.gcn_ent2ent.append(
                    _LegacyGCNLayer(graph_dim, graph_dim, residual=True)
                )
                self.gcn_ent2rel.append(
                    _LegacyGraphConvolutionLayerCollect(graph_dim, graph_dim)
                )
                self.gcn_rel2rel.append(
                    _LegacyGCNLayer(graph_dim, graph_dim, residual=True)
                )

        self.use_geo_ee = bool(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "USE_GEO_EE",
                default=False,
            )
        )
        self.geo_ee_log_period = int(
            _cfg_get(
                cfg,
                "MODEL",
                "ROI_RELATION_HEAD",
                "GEO_EE_LOG_PERIOD",
                default=100,
            )
        )
        self._geo_ee_forward_counts = {"train": 0, "inference": 0}
        if self.use_geo_ee:
            if self.relation_graph_mode != "sgg_toolkit":
                raise ValueError(
                    "USE_GEO_EE is restricted to the Original SGG-ToolKit RPCM graph"
                )
            if str(_cfg_get(cfg, "MODEL", "BOX_MODE", default="")).lower() != "obb":
                raise ValueError("USE_GEO_EE requires MODEL.BOX_MODE='obb'")
            if self.feat_update_step <= 0:
                raise ValueError("USE_GEO_EE requires at least one RPCM update step")
            geo_heads = int(
                _cfg_get(
                    cfg,
                    "MODEL",
                    "ROI_RELATION_HEAD",
                    "GEO_EE_NUM_HEADS",
                    default=4,
                )
            )
            geo_head_dim = int(
                _cfg_get(
                    cfg,
                    "MODEL",
                    "ROI_RELATION_HEAD",
                    "GEO_EE_HEAD_DIM",
                    default=64,
                )
            )
            geo_hidden_dim = int(
                _cfg_get(
                    cfg,
                    "MODEL",
                    "ROI_RELATION_HEAD",
                    "GEO_EE_GEOMETRY_HIDDEN_DIM",
                    default=32,
                )
            )
            geo_query_chunk_size = int(
                _cfg_get(
                    cfg,
                    "MODEL",
                    "ROI_RELATION_HEAD",
                    "GEO_EE_QUERY_CHUNK_SIZE",
                    default=64,
                )
            )
            geo_gate_init = float(
                _cfg_get(
                    cfg,
                    "MODEL",
                    "ROI_RELATION_HEAD",
                    "GEO_EE_GATE_INIT",
                    default=0.1,
                )
            )
            if geo_heads != 4:
                raise ValueError(f"GeoEE experiment is fixed to 4 heads, got {geo_heads}")
            if not math.isfinite(geo_gate_init):
                raise ValueError("GEO_EE_GATE_INIT must be finite")
            self.geo_ee_attention = nn.ModuleList(
                GeometryAwareEEAttention(
                    graph_dim,
                    num_heads=geo_heads,
                    head_dim=geo_head_dim,
                    geometry_hidden_dim=geo_hidden_dim,
                    query_chunk_size=geo_query_chunk_size,
                )
                for _ in range(self.feat_update_step)
            )
            self.geo_gate = nn.Parameter(
                torch.full((self.feat_update_step,), geo_gate_init)
            )
        else:
            self.geo_ee_attention = nn.ModuleList()
            self.register_parameter("geo_gate", None)

        if self.entity_source_fusion_mode == "global":
            self.entity_source_fusion = GlobalEntitySourceFusion()
        elif self.entity_source_fusion_mode == "asf_e":
            fusion_hidden_dim = int(
                _cfg_get(
                    cfg,
                    "MODEL",
                    "ROI_RELATION_HEAD",
                    "RPCM_ASF_E_HIDDEN_DIM",
                    default=128,
                )
            )
            self.entity_source_fusion = AdaptiveEntitySourceFusion(
                graph_dim, fusion_hidden_dim
            )
        elif self.entity_source_fusion_mode == "anchored_asf_e":
            fusion_hidden_dim = int(
                _cfg_get(
                    cfg,
                    "MODEL",
                    "ROI_RELATION_HEAD",
                    "RPCM_ASF_E_HIDDEN_DIM",
                    default=128,
                )
            )
            anchor_weights = tuple(
                float(value)
                for value in _cfg_get(
                    cfg,
                    "MODEL",
                    "ROI_RELATION_HEAD",
                    "RPCM_ASF_E_ANCHOR_WEIGHTS",
                    default=(0.4046995342, 0.3421354294, 0.2531650066),
                )
            )
            residual_rho = float(
                _cfg_get(
                    cfg,
                    "MODEL",
                    "ROI_RELATION_HEAD",
                    "RPCM_ASF_E_RESIDUAL_RHO",
                    default=0.5,
                )
            )
            self.entity_source_fusion = GlobalAnchoredAdaptiveEntitySourceFusion(
                graph_dim,
                fusion_hidden_dim,
                anchor_weights=anchor_weights,
                rho=residual_rho,
            )
        else:
            self.entity_source_fusion = None

        self.round11_feature_capture_enabled = False
        self._round11_feature_state: dict[str, torch.Tensor] = {}
        self.round13_representation_capture_enabled = False
        self._round13_representation_state: dict[str, torch.Tensor] = {}

        print(
            "[RPCM_SGG_TOOLKIT_ORIGINAL] "
            f"graph_dim={graph_dim}, mlp_dim={self.mlp_dim}, "
            f"feat_update_step={self.feat_update_step}, "
            f"relation_graph={self.relation_graph_mode}, "
            f"graph_backend={self.graph_backend}, "
            f"context_ablation={self.context_ablation}, "
            f"relation_adjacency={self.relation_adjacency}, "
            f"use_geo_ee={self.use_geo_ee}, "
            f"geo_gate={self.geo_gate_values()}, "
            f"entity_source_fusion={self.entity_source_fusion_mode}, "
            f"rel_entity_semantic_alpha={self.rel_entity_semantic_alpha:g}, "
            f"rel_entity_semantic_gate={self.rel_entity_semantic_gate_mode}, "
            f"rel_entity_high_hhi={self.rel_entity_high_hhi_threshold:.12g}, "
            f"relation_entity_probe={self.relation_entity_probe_enabled}, "
            f"coarse_prototypes={self.Par}, "
            "distance_impl=matrix, "
            f"obj_glove_missing={obj_glove_diag.get('missing_classes', [])}, "
            f"rel_glove_missing={rel_glove_diag.get('missing_predicates', [])}",
            flush=True,
        )

    @staticmethod
    def _fusion_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.relu(x + y) - (x - y).pow(2)

    def geo_gate_values(self) -> tuple[float, ...]:
        """Return detached per-layer GeoEE lambda values for logging/audits."""

        if self.geo_gate is None:
            return ()
        return tuple(float(value) for value in self.geo_gate.detach().cpu())

    def _record_geo_gate_values(self) -> None:
        if not self.use_geo_ee:
            return
        mode = "train" if self.training else "inference"
        self._geo_ee_forward_counts[mode] += 1
        count = self._geo_ee_forward_counts[mode]
        period = self.geo_ee_log_period
        if count != 1 and (period <= 0 or count % period != 0):
            return
        values = ", ".join(
            f"layer_{layer}={value:.8g}"
            for layer, value in enumerate(self.geo_gate_values())
        )
        print(
            f"[GeoEE] mode={mode} forward={count} geo_gate/lambda: {values}",
            flush=True,
        )

    def _geo_ee_message(
        self,
        step: int,
        entity_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_geometry: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.use_geo_ee or not self._path_enabled(4):
            return entity_features.new_zeros(entity_features.shape)
        if node_geometry is None:
            raise RuntimeError("GeoEE geometry was not prepared")
        return self.geo_ee_attention[int(step)](
            entity_features,
            adjacency,
            node_geometry,
        )

    def set_round11_feature_capture(self, enabled: bool) -> None:
        """Enable read-only frozen feature capture for the Round 11 probe."""

        self.round11_feature_capture_enabled = bool(enabled)
        self.pairwise_feature_extractor.round11_feature_capture_enabled = bool(enabled)
        if not enabled:
            self._round11_feature_state = {}

    def round11_feature_state(self) -> dict[str, torch.Tensor]:
        return dict(self._round11_feature_state)

    def set_round13_representation_capture(self, enabled: bool) -> None:
        """Enable read-only capture of the frozen final-representation ladder.

        The hook is deliberately separate from the Round 11 evidence capture:
        it observes tensors that already exist in the Original forward and
        never changes the classifier input, parameters, or returned logits.
        """

        self.round13_representation_capture_enabled = bool(enabled)
        if not enabled:
            self._round13_representation_state = {}

    def round13_representation_state(self) -> dict[str, torch.Tensor]:
        return dict(self._round13_representation_state)

    def reset_source_weight_statistics(self) -> None:
        self._source_weight_stats = {}

    def _record_source_weights(self, step: int, weights: torch.Tensor) -> None:
        # Statistics are an inference diagnostic.  Avoid four GPU->CPU
        # synchronizations per training graph and keep epoch summaries from
        # mixing stochastic training batches with the fixed validation split.
        if self.training or not self.source_fusion_record_stats or weights.numel() == 0:
            return
        values = weights.detach().float()
        entropy = -(values.clamp_min(1e-12) * values.clamp_min(1e-12).log()).sum(dim=1)
        entry = self._source_weight_stats.setdefault(
            int(step),
            {
                "count": 0,
                "sum": torch.zeros(3, dtype=torch.float64),
                "sum_sq": torch.zeros(3, dtype=torch.float64),
                "entropy_sum": 0.0,
            },
        )
        entry["count"] = int(entry["count"]) + int(values.size(0))
        entry["sum"] = entry["sum"] + values.double().sum(dim=0).cpu()
        entry["sum_sq"] = entry["sum_sq"] + values.double().square().sum(dim=0).cpu()
        entry["entropy_sum"] = float(entry["entropy_sum"]) + float(entropy.double().sum().cpu())

    def source_weight_statistics(self) -> dict[str, object]:
        rows = []
        for step, entry in sorted(self._source_weight_stats.items()):
            count = max(int(entry["count"]), 1)
            mean = entry["sum"] / float(count)
            variance = (entry["sum_sq"] / float(count) - mean.square()).clamp_min(0.0)
            rows.append(
                {
                    "step": int(step),
                    "entities": int(entry["count"]),
                    "mean": [float(value) for value in mean],
                    "variance": [float(value) for value in variance],
                    "mean_entropy": float(entry["entropy_sum"]) / float(count),
                }
            )
        global_weights = None
        if isinstance(
            self.entity_source_fusion,
            (GlobalEntitySourceFusion, GlobalAnchoredAdaptiveEntitySourceFusion),
        ):
            global_weights = [
                float(value)
                for value in torch.softmax(
                    self.entity_source_fusion.logits.detach().float(), dim=0
                ).cpu()
            ]
        return {
            "mode": self.entity_source_fusion_mode,
            "source_order": ["entity_to_entity", "relation_subject_to_entity", "relation_object_to_entity"],
            "global_weights": global_weights,
            "residual_rho": (
                float(self.entity_source_fusion.rho)
                if isinstance(
                    self.entity_source_fusion,
                    GlobalAnchoredAdaptiveEntitySourceFusion,
                )
                else None
            ),
            "per_step": rows,
        }

    def relation_entity_probe_state(
        self,
    ) -> tuple[torch.Tensor | None, tuple[int, ...]]:
        """Return the current diagnostic gate and per-image relation counts."""

        return self._relation_entity_probe_gate, self._relation_entity_probe_num_rels

    def _fuse_entity_sources(
        self,
        step: int,
        target: torch.Tensor,
        source_obj: torch.Tensor,
        source_rel_sub: torch.Tensor,
        source_rel_obj: torch.Tensor,
    ) -> torch.Tensor:
        sources = (source_obj, source_rel_sub, source_rel_obj)
        if self.entity_source_fusion is None:
            fused = (source_obj + source_rel_sub + source_rel_obj) / 3.0
            weights = target.new_full((target.size(0), 3), 1.0 / 3.0)
        else:
            fused, weights = self.entity_source_fusion(target, sources)
        self._record_source_weights(step, weights)
        return fused

    @staticmethod
    def _role_relation_adjacencies(
        subj_pred_map: torch.Tensor,
        obj_pred_map: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build the four endpoint-role relation graphs.

        Rows are message targets and columns are message sources, matching
        ``_LegacyCollectionUnit``.  Cross-role matrices are consequently
        directional and transpose each other.
        """

        products = {
            "ss": (subj_pred_map.t() @ subj_pred_map) > 0,
            "oo": (obj_pred_map.t() @ obj_pred_map) > 0,
            "so": (subj_pred_map.t() @ obj_pred_map) > 0,
            "os": (obj_pred_map.t() @ subj_pred_map) > 0,
        }
        for adjacency in products.values():
            adjacency.fill_diagonal_(False)
        unified = products["ss"] | products["oo"] | products["so"] | products["os"]
        return {
            **{key: value.to(dtype=subj_pred_map.dtype) for key, value in products.items()},
            "unified": unified.to(dtype=subj_pred_map.dtype),
        }

    def _diagnostic_entity_adjacency(
        self,
        original: torch.Tensor,
        subj_pred_map: torch.Tensor,
        obj_pred_map: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the frozen Round-1 entity-topology intervention.

        The default is an exact identity.  ``relation_induced`` connects two
        entity nodes iff at least one selected directed relation joins them,
        and then symmetrizes that support.  This helper owns no parameters and
        is intentionally restricted to the Original SGG-ToolKit backend.
        """

        mode = self.entity_adjacency
        if mode == "complete":
            return original
        if mode == "none":
            return torch.zeros_like(original)
        directed = (subj_pred_map @ obj_pred_map.t()) > 0
        adjacency = directed | directed.t()
        adjacency.fill_diagonal_(False)
        return adjacency.to(dtype=original.dtype)

    def _path_enabled(self, unit_id: int) -> bool:
        disabled = {
            "full": set(),
            "no_rel_rel": {5},
            "no_rel_ent": {0, 1},
            "no_ent_rel": {2, 3},
            "no_ent_ent": {4},
            "rel_rel_only": {0, 1, 2, 3, 4},
        }[self.context_ablation]
        return int(unit_id) not in disabled

    def _collect_or_zero(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        attention: torch.Tensor,
        unit_id: int,
    ) -> torch.Tensor:
        if not self._path_enabled(unit_id):
            return target.new_zeros(target.shape)
        return self.gcn_collect_feat(target, source, attention, unit_id)

    def _relation_to_entity_message(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        attention: torch.Tensor,
        semantic_group_ids: torch.Tensor,
        unit_id: int,
    ) -> torch.Tensor:
        """Collect mode 0/1 messages with optional semantic tempering.

        The alpha=1 branch is delegated unchanged to the published collector.
        This helper owns no parameters and is active only for the Original
        SGG-ToolKit backend.
        """

        if int(unit_id) not in (0, 1):
            raise ValueError("Semantic relation->entity collection requires mode 0 or 1")
        if not self._path_enabled(unit_id):
            return target.new_zeros(target.shape)
        return self.gcn_collect_feat(
            target,
            source,
            attention,
            unit_id,
            semantic_group_ids,
            self.rel_entity_semantic_alpha,
            self.rel_entity_semantic_gate_mode,
            self.rel_entity_high_hhi_threshold,
        )

    def _relation_semantic_group_ids(
        self,
        proposals: Sequence,
        rel_pair_idxs: Sequence[torch.Tensor],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the train/test-label-free `(head_class, tail_class)` key."""

        groups: list[torch.Tensor] = []
        for proposal, pairs in zip(proposals, rel_pair_idxs):
            if not proposal.has_field("labels"):
                raise KeyError(
                    "Semantic multiplicity tempering requires proposal labels"
                )
            labels = proposal.get_field("labels").to(device=device, dtype=torch.long)
            pairs = pairs.to(device=device, dtype=torch.long)
            if pairs.numel() == 0:
                continue
            if bool((pairs < 0).any()) or bool((pairs >= labels.numel()).any()):
                raise ValueError("Relation pair contains an invalid endpoint")
            groups.append(
                labels[pairs[:, 0]] * int(self.num_obj_classes)
                + labels[pairs[:, 1]]
            )
        if not groups:
            return torch.zeros((0,), dtype=torch.long, device=device)
        return torch.cat(groups, dim=0)

    def _relation_to_relation_message(
        self,
        relation_features: torch.Tensor,
        role_adjacencies: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not self._path_enabled(5):
            return relation_features.new_zeros(relation_features.shape)
        if self.relation_adjacency == "unified":
            return self.gcn_collect_feat(
                relation_features,
                relation_features,
                role_adjacencies["unified"],
                5,
            )
        if self.relation_adjacency in {"rr_same_role", "rr_cross_role"}:
            roles = (
                ("ss", "oo")
                if self.relation_adjacency == "rr_same_role"
                else ("so", "os")
            )
            # Round-7 frozen interventions first form the logical union and
            # then apply the *same* published mode-5 neighborhood mean.  This
            # deliberately differs from the historical ``ss_oo`` switch,
            # which averages two already-normalized channel messages.
            adjacency = role_adjacencies[roles[0]].bool()
            adjacency = adjacency | role_adjacencies[roles[1]].bool()
            return self.gcn_collect_feat(
                relation_features,
                relation_features,
                adjacency.to(dtype=relation_features.dtype),
                5,
            )
        if self.relation_adjacency == "rr_four_role_balanced":
            roles = ("ss", "oo", "so", "os")
            messages = []
            active = []
            for role in roles:
                adjacency = role_adjacencies[role]
                messages.append(
                    self.gcn_collect_feat(
                        relation_features,
                        relation_features,
                        adjacency,
                        5,
                    )
                )
                active.append(adjacency.ne(0).any(dim=1))
            stacked = torch.stack(messages, dim=0)
            active_mask = torch.stack(active, dim=0)
            active_count = active_mask.sum(dim=0).clamp_min(1).to(
                dtype=stacked.dtype
            )
            # Empty role channels must not dilute a relation node's message.
            # Collector outputs are already zero for empty rows; the mask is
            # retained explicitly to make the per-node contract auditable.
            return (
                stacked * active_mask.unsqueeze(-1).to(stacked.dtype)
            ).sum(dim=0) / active_count.unsqueeze(-1)
        roles = ("ss", "oo") if self.relation_adjacency == "ss_oo" else (
            "ss",
            "oo",
            "so",
            "os",
        )
        messages = [
            self.gcn_collect_feat(
                relation_features,
                relation_features,
                role_adjacencies[role],
                5,
            )
            for role in roles
        ]
        combined = messages[0]
        for message in messages[1:]:
            combined = combined + message
        return combined / float(len(messages))

    @staticmethod
    def _to_onehot_logits(
        labels: torch.Tensor, num_classes: int, fill: float = 1000.0
    ) -> torch.Tensor:
        logits = torch.full(
            (labels.numel(), num_classes),
            -fill,
            dtype=torch.float32,
            device=labels.device,
        )
        if labels.numel() > 0:
            logits[
                torch.arange(labels.numel(), device=labels.device),
                labels.long(),
            ] = fill
        return logits

    def _refine_obj_labels(
        self, roi_features: torch.Tensor, proposals: Sequence
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Faithful SGG-Toolkit RPCM task routing. During SGCls/SGDet training,
        # proposal labels supervise out_obj; at inference the semantic input
        # comes only from detached detector logits. PredCls remains byte- and
        # numerically compatible with the former strict audit path.
        use_gt_label = self.training or self.use_gt_object_label
        obj_labels = (
            torch.cat(
                [
                    proposal.get_field("labels").long().to(roi_features.device)
                    for proposal in proposals
                ],
                dim=0,
            )
            if use_gt_label
            else None
        )

        # The source evaluates this branch twice.  Besides being redundant,
        # that updates BatchNorm running statistics twice during training.
        # Preserve it here for numerical/optimization fidelity.
        _ = self.pos_embed(_orig_encode_box_info(proposals).to(roi_features.device))
        if self.use_gt_object_label:
            if obj_labels is None:
                raise RuntimeError("PredCls object labels are unavailable")
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = torch.cat(
                [
                    proposal.get_field("predict_logits").to(roi_features.device)
                    for proposal in proposals
                ],
                dim=0,
            ).detach()
            obj_embed = (
                F.softmax(obj_logits[:, : self.num_obj_classes], dim=1)
                @ self.obj_embed1.weight
            )
        pos_embed = self.pos_embed(
            _orig_encode_box_info(proposals).to(roi_features.device)
        )
        obj_pre_rep_for_pred = self.lin_obj_cyx(
            torch.cat([roi_features, obj_embed, pos_embed], dim=-1)
        )
        if self.mode == "predcls":
            if obj_labels is None:
                raise RuntimeError("PredCls object labels are unavailable")
            obj_preds = obj_labels
            obj_dists = self._to_onehot_logits(
                obj_preds, self.num_obj_classes
            )
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)
            num_objs = [len(proposal) for proposal in proposals]
            if self.mode == "sgdet" and not self.training:
                dist_parts = obj_dists.split(num_objs, dim=0)
                pred_parts = []
                for proposal, dist in zip(proposals, dist_parts):
                    if not proposal.has_field("boxes_per_cls"):
                        raise ValueError(
                            "SGDet Original RPCM inference requires boxes_per_cls"
                        )
                    pred_parts.append(
                        obj_prediction_nms(
                            proposal,
                            proposal.get_field("boxes_per_cls"),
                            dist,
                            self.nms_thresh,
                        ).long()
                    )
                obj_preds = torch.cat(pred_parts, dim=0)
            else:
                obj_preds = (obj_dists[:, 1:].max(dim=1).indices + 1).long()
        labels_for_audit = obj_labels if obj_labels is not None else obj_preds
        return obj_dists, obj_preds, labels_for_audit

    def _coarse_predicate_prototypes(
        self, predicate_proto: torch.Tensor
    ) -> torch.Tensor:
        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:
            raise RuntimeError(
                "RPCM_SGG_TOOLKIT_ORIGINAL requires scikit-learn for the "
                "source per-forward KMeans prototype clustering."
            ) from exc

        foreground = predicate_proto[1:].detach().cpu().numpy()
        kmeans = KMeans(
            n_clusters=self.Par - 1,
            n_init=10,
            random_state=0,
        ).fit(foreground)
        centers = torch.as_tensor(
            kmeans.cluster_centers_,
            dtype=predicate_proto.dtype,
            device=predicate_proto.device,
        )
        return torch.cat([predicate_proto[:1].detach(), centers], dim=0)

    def _prototype_losses(
        self,
        rel_rep: torch.Tensor,
        predicate_proto1: torch.Tensor,
        predicate_proto2: torch.Tensor,
        predicate_proto_norm1: torch.Tensor,
        predicate_proto_norm2: torch.Tensor,
        rel_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_proto_norm1 = predicate_proto_norm1.detach()
        target_proto_norm2 = predicate_proto_norm2.detach()
        simil_mat1 = predicate_proto_norm1 @ target_proto_norm1.t()
        simil_mat2 = predicate_proto_norm2 @ target_proto_norm2.t()
        l21_1 = torch.norm(
            torch.norm(simil_mat1, p=2, dim=1), p=1
        ) / float(self.num_rel_classes * self.num_rel_classes)
        l21_2 = torch.norm(
            torch.norm(simil_mat2, p=2, dim=1), p=1
        ) / float(self.Par * self.Par)

        gamma2 = 7.0
        proto_dis_mat1 = (
            predicate_proto1.unsqueeze(1)
            - predicate_proto1.detach().unsqueeze(0)
        ).norm(dim=2).pow(2)
        proto_dis_mat2 = (
            predicate_proto2.unsqueeze(1)
            - predicate_proto2.detach().unsqueeze(0)
        ).norm(dim=2).pow(2)
        topk_proto_dis1 = torch.sort(proto_dis_mat1, dim=1).values[:, :2].sum(
            dim=1
        )
        topk_proto_dis2 = torch.sort(proto_dis_mat2, dim=1).values[:, :2].sum(
            dim=1
        )
        dist_loss1 = torch.maximum(
            predicate_proto1.new_zeros((self.num_rel_classes,)),
            -topk_proto_dis1 + gamma2,
        ).mean()
        dist_loss2 = torch.maximum(
            predicate_proto2.new_zeros((self.Par,)),
            -topk_proto_dis2 + gamma2,
        ).mean()

        if rel_labels.numel() == 0:
            loss_dis = rel_rep.sum() * 0.0
        else:
            gamma1 = 1.0
            distance_set = _matrix_squared_euclidean_distance(
                rel_rep,
                predicate_proto1,
            )
            mask_neg = distance_set.new_ones(distance_set.shape)
            row_idx = torch.arange(rel_labels.numel(), device=rel_labels.device)
            mask_neg[row_idx, rel_labels] = 0
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[row_idx, rel_labels]
            negative_count = min(11, self.num_rel_classes)
            topk_negative = torch.sort(distance_set_neg, dim=1).values[
                :, :negative_count
            ].sum(dim=1) / float(max(negative_count - 1, 1))
            loss_dis = torch.maximum(
                distance_set_pos.new_zeros(distance_set_pos.shape),
                distance_set_pos - topk_negative + gamma1,
            ).mean()

        return {
            "l21_1_loss": l21_1,
            "l21_2_loss": l21_2,
            "dist_loss2_1": dist_loss1,
            "dist_loss2_2": dist_loss2,
            "loss_dis": loss_dis,
        }

    def _refine_relation_representation(
        self,
        rel_rep: torch.Tensor,
        predicate_proto: torch.Tensor,
        rel_labels: torch.Tensor | None,
        proposals: Sequence,
        rel_pair_idxs: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Extension point immediately before the published predicate head.

        The Original predictor deliberately returns the input unchanged.  A
        separate paper-reproduction predictor may override this hook without
        copying (and consequently drifting from) the six-path RPCM forward.
        ``predicate_proto`` is the unprojected 2048-D predicate representation.
        """

        del predicate_proto, rel_labels, proposals, rel_pair_idxs
        return rel_rep, {}

    def _calibrate_relation_logits(
        self,
        relation_logits: torch.Tensor,
        *,
        union_features: torch.Tensor | tuple[torch.Tensor, ...] | None,
        relation_representation: torch.Tensor,
        rel_labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Extension point for frozen post-training calibration methods."""

        del union_features, relation_representation, rel_labels
        return relation_logits, {}

    def _method_relation_logits(
        self,
        relation_logits: torch.Tensor,
        *,
        relation_representation: torch.Tensor,
        predicate_prototype: torch.Tensor,
    ) -> torch.Tensor:
        """Extension point for a method-specific predicate decision space.

        ``relation_representation`` and ``predicate_prototype`` are the
        unprojected representations immediately after the optional method
        refinement.  The Original RPCM path is an exact identity.
        """

        del relation_representation, predicate_prototype
        return relation_logits

    def _original_prototype_losses_enabled(self) -> bool:
        """Whether to retain RPCM's own five prototype regularizers."""

        return True

    def set_runtime_feat_update_steps(self, steps: int | None) -> None:
        """Override the number of propagation steps for controlled inference.

        ``None`` restores the configured value.  This owns no parameter and is
        used by the ReCon1M R3 T0/T4 comparison so both variants share one
        checkpoint and an identical candidate graph.
        """

        if steps is not None and not 0 <= int(steps) <= int(self.feat_update_step):
            raise ValueError(
                "Runtime RPCM steps must be in [0, configured steps], got "
                f"{steps!r} for configured {self.feat_update_step}."
            )
        self._runtime_feat_update_steps = None if steps is None else int(steps)

    @staticmethod
    def _r4_relation_masks(
        proposals: Sequence,
        rel_pair_idxs: Sequence[torch.Tensor],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Read relation-aligned R4 source/query masks from entity proposals.

        Absence means the exact Original path.  The query mask controls only
        mode 5 and final output routing; the context mask controls only modes
        0/1.  Neither mask changes modes 2/3 or the complete entity graph.
        """

        context_parts: list[torch.Tensor] = []
        query_parts: list[torch.Tensor] = []
        for proposal, pairs in zip(proposals, rel_pair_idxs):
            count = int(pairs.size(0))
            if proposal.has_field("rpcm_context_source_mask"):
                context = proposal.get_field("rpcm_context_source_mask").to(
                    device=device, dtype=torch.bool
                )
            else:
                context = torch.ones((count,), device=device, dtype=torch.bool)
            if proposal.has_field("rpcm_prediction_query_mask"):
                query = proposal.get_field("rpcm_prediction_query_mask").to(
                    device=device, dtype=torch.bool
                )
            else:
                query = torch.ones((count,), device=device, dtype=torch.bool)
            if context.ndim != 1 or context.numel() != count:
                raise ValueError(
                    "RPCM context-source mask must contain one value per U relation"
                )
            if query.ndim != 1 or query.numel() != count:
                raise ValueError(
                    "RPCM prediction-query mask must contain one value per U relation"
                )
            context_parts.append(context)
            query_parts.append(query)
        empty = torch.zeros((0,), device=device, dtype=torch.bool)
        return (
            torch.cat(context_parts) if context_parts else empty,
            torch.cat(query_parts) if query_parts else empty,
            context_parts,
            query_parts,
        )

    @staticmethod
    def _incidence_scatter_mean(
        transformed: torch.Tensor,
        target_index: torch.Tensor,
        num_targets: int,
    ) -> torch.Tensor:
        output = transformed.new_zeros((int(num_targets), transformed.size(1)))
        if transformed.numel() == 0:
            return output
        output.index_add_(0, target_index, transformed)
        degree = torch.bincount(target_index, minlength=int(num_targets)).to(
            dtype=transformed.dtype
        )
        return output / (degree[:, None] + 1.0e-7)

    def _incidence_complete_entity_message(
        self,
        entity_features: torch.Tensor,
        num_objs: Sequence[int],
    ) -> torch.Tensor:
        transformed = F.relu(
            self.gcn_collect_feat.collect_units[4].fc(entity_features)
        )
        output = transformed.new_zeros(transformed.shape)
        offset = 0
        for count in map(int, num_objs):
            chunk = transformed[offset : offset + count]
            if count > 1:
                output[offset : offset + count] = (
                    chunk.sum(dim=0, keepdim=True) - chunk
                ) / (chunk.new_tensor(float(count - 1)) + 1.0e-7)
            offset += count
        return output

    def _incidence_unified_relation_message(
        self,
        relation_features: torch.Tensor,
        rel_pair_idxs: Sequence[torch.Tensor],
        num_objs: Sequence[int],
        subject_index: torch.Tensor,
        object_index: torch.Tensor,
        total_objects: int,
    ) -> torch.Tensor:
        transformed = F.relu(
            self.gcn_collect_feat.collect_units[5].fc(relation_features)
        )
        if transformed.numel() == 0:
            return transformed
        incident_sum = transformed.new_zeros((int(total_objects), transformed.size(1)))
        incident_sum.index_add_(0, subject_index, transformed)
        incident_sum.index_add_(0, object_index, transformed)
        incident_degree = torch.bincount(
            torch.cat((subject_index, object_index)), minlength=int(total_objects)
        ).to(dtype=transformed.dtype)

        reverse = transformed.new_zeros(transformed.shape)
        reverse_exists = torch.zeros(
            (transformed.size(0),), dtype=torch.bool, device=transformed.device
        )
        edge_offset = 0
        for pairs, count in zip(rel_pair_idxs, num_objs):
            edge_count = int(pairs.size(0))
            if edge_count == 0:
                continue
            local = pairs.to(device=transformed.device, dtype=torch.long)
            key = local[:, 0] * int(count) + local[:, 1]
            reverse_key = local[:, 1] * int(count) + local[:, 0]
            order = torch.argsort(key, stable=True)
            sorted_key = key[order]
            location = torch.searchsorted(sorted_key, reverse_key)
            safe_location = location.clamp(max=max(edge_count - 1, 0))
            matched = (location < edge_count) & (
                sorted_key[safe_location] == reverse_key
            )
            rows = torch.arange(edge_count, device=transformed.device) + edge_offset
            if matched.any():
                reverse_rows = order[safe_location[matched]] + edge_offset
                reverse[rows[matched]] = transformed[reverse_rows]
                reverse_exists[rows[matched]] = True
            edge_offset += edge_count

        numerator = (
            incident_sum[subject_index]
            + incident_sum[object_index]
            - 2.0 * transformed
            - reverse
        )
        degree = (
            incident_degree[subject_index]
            + incident_degree[object_index]
            - 2.0
            - reverse_exists.to(dtype=transformed.dtype)
        ).clamp_min(0.0)
        return numerator / (degree[:, None] + 1.0e-7)

    def _sgg_toolkit_incidence_propagation(
        self,
        object_features: torch.Tensor,
        relation_features: torch.Tensor,
        proposals: Sequence,
        rel_pair_idxs: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact sparse form of the six published RPCM message pathways.

        It replaces only dense incidence/adjacency materialisation.  The same
        six learned collection FCs, role definitions, neighborhood means,
        source fusion and residual updates are used.
        """

        unsupported = []
        if self.relation_graph_mode != "sgg_toolkit":
            unsupported.append("relation_graph_mode")
        if self.relation_adjacency != "unified":
            unsupported.append("relation_adjacency")
        if self.entity_adjacency != "complete":
            unsupported.append("entity_adjacency")
        if self.context_ablation != "full":
            unsupported.append("context_ablation")
        if float(self.rel_entity_semantic_alpha) != 1.0:
            unsupported.append("semantic_tempering")
        if self.relation_entity_probe_enabled:
            unsupported.append("relation_entity_probe")
        if unsupported:
            raise ValueError(
                "The incidence backend is the exact Original/full RPCM path; "
                f"unsupported interventions: {sorted(set(unsupported))}."
            )

        num_objs = [len(proposal) for proposal in proposals]
        subject_parts: list[torch.Tensor] = []
        object_parts: list[torch.Tensor] = []
        object_offset = 0
        for count, pairs in zip(num_objs, rel_pair_idxs):
            local = pairs.to(device=object_features.device, dtype=torch.long)
            if local.numel():
                if bool((local < 0).any()) or bool((local >= int(count)).any()):
                    raise ValueError("Relation endpoint outside its image")
                if bool((local[:, 0] == local[:, 1]).any()):
                    raise ValueError("Self relation is illegal")
                key = local[:, 0] * int(count) + local[:, 1]
                if torch.unique(key).numel() != key.numel():
                    raise ValueError("Duplicate relation pair is illegal")
                subject_parts.append(local[:, 0] + object_offset)
                object_parts.append(local[:, 1] + object_offset)
            object_offset += int(count)
        subject_index = (
            torch.cat(subject_parts)
            if subject_parts
            else torch.zeros((0,), dtype=torch.long, device=object_features.device)
        )
        object_index = (
            torch.cat(object_parts)
            if object_parts
            else torch.zeros((0,), dtype=torch.long, device=object_features.device)
        )
        if subject_index.numel() != relation_features.size(0):
            raise RuntimeError("Incidence endpoints do not align with relation features")

        context_source_mask, query_mask, _, query_parts = self._r4_relation_masks(
            proposals, rel_pair_idxs, device=object_features.device
        )
        if context_source_mask.numel() != relation_features.size(0):
            raise RuntimeError("R4 context mask does not align with relation features")
        if query_mask.numel() != relation_features.size(0):
            raise RuntimeError("R4 query mask does not align with relation features")
        all_context_sources_active = bool(context_source_mask.all())
        all_relations_are_queries = bool(query_mask.all()) or bool(
            self.qg_full_relation_context
        )

        runtime_steps = getattr(self, "_runtime_feat_update_steps", None)
        steps = self.feat_update_step if runtime_steps is None else int(runtime_steps)
        geo_ee_adjacency = None
        geo_ee_geometry = None
        if self.use_geo_ee:
            geo_ee_adjacency = OBBGeometryEncoder.complete_adjacency(
                proposals,
                device=object_features.device,
                dtype=object_features.dtype,
            )
            geo_ee_geometry = OBBGeometryEncoder.build_node_geometry(
                proposals,
                device=object_features.device,
                dtype=object_features.dtype,
            )
        current_obj = object_features
        current_rel = relation_features
        for step in range(steps):
            source_obj = self._incidence_complete_entity_message(
                current_obj, num_objs
            )
            if self.use_geo_ee:
                source_obj = source_obj + self.geo_gate[step] * self._geo_ee_message(
                    step,
                    current_obj,
                    geo_ee_adjacency,
                    geo_ee_geometry,
                )
            transformed_rel_sub = F.relu(
                self.gcn_collect_feat.collect_units[0].fc(current_rel)
            )
            transformed_rel_obj = F.relu(
                self.gcn_collect_feat.collect_units[1].fc(current_rel)
            )
            if all_context_sources_active:
                source_rel_sub = self._incidence_scatter_mean(
                    transformed_rel_sub, subject_index, current_obj.size(0)
                )
                source_rel_obj = self._incidence_scatter_mean(
                    transformed_rel_obj, object_index, current_obj.size(0)
                )
            else:
                source_rel_sub = self._incidence_scatter_mean(
                    transformed_rel_sub[context_source_mask],
                    subject_index[context_source_mask],
                    current_obj.size(0),
                )
                source_rel_obj = self._incidence_scatter_mean(
                    transformed_rel_obj[context_source_mask],
                    object_index[context_source_mask],
                    current_obj.size(0),
                )
            source2obj = self._fuse_entity_sources(
                step, current_obj, source_obj, source_rel_sub, source_rel_obj
            )
            next_obj = self.gcn_update_feat(current_obj, source2obj, 0)

            transformed_sub = F.relu(
                self.gcn_collect_feat.collect_units[2].fc(current_obj)
            )
            transformed_obj = F.relu(
                self.gcn_collect_feat.collect_units[3].fc(current_obj)
            )
            source_sub_rel = transformed_sub[subject_index] / 1.0000001
            source_obj_rel = transformed_obj[object_index] / 1.0000001
            if all_relations_are_queries:
                source_rel_rel = self._incidence_unified_relation_message(
                    current_rel,
                    rel_pair_idxs,
                    num_objs,
                    subject_index,
                    object_index,
                    current_obj.size(0),
                )
            else:
                query_pairs = [
                    pairs[mask.to(device=pairs.device)]
                    for pairs, mask in zip(rel_pair_idxs, query_parts)
                ]
                query_message = self._incidence_unified_relation_message(
                    current_rel[query_mask],
                    query_pairs,
                    num_objs,
                    subject_index[query_mask],
                    object_index[query_mask],
                    current_obj.size(0),
                )
                source_rel_rel = current_rel.new_zeros(current_rel.shape)
                source_rel_rel[query_mask] = query_message
            source2rel = (
                source_sub_rel + source_obj_rel + source_rel_rel
            ) / 3.0
            next_rel = self.gcn_update_feat(current_rel, source2rel, 1)
            current_obj, current_rel = next_obj, next_rel
        return current_obj, current_rel

    def _classify_propagated_features(
        self,
        *,
        obj_features: torch.Tensor,
        rel_features: torch.Tensor,
        proposals: Sequence,
        rel_pair_idxs: Sequence[torch.Tensor],
        rel_labels: Sequence[torch.Tensor] | None,
        union_features: torch.Tensor | tuple[torch.Tensor, ...] | None,
        add_losses: dict[str, torch.Tensor],
    ):
        """Run the unchanged Original predicate head after graph propagation."""

        num_rels = [pair_idx.shape[0] for pair_idx in rel_pair_idxs]
        entity_dists, entity_preds, _ = self._refine_obj_labels(
            obj_features, proposals
        )
        entity_rep = self.post_emb(obj_features).view(
            obj_features.size(0), 2, self.mlp_dim
        )
        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)
        entity_embeds = self.obj_embed(entity_preds)

        num_objs = [len(proposal) for proposal in proposals]
        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_embeds = entity_embeds.split(num_objs, dim=0)
        fusion_so_parts: list[torch.Tensor] = []
        for pair_idx, sub_i, obj_i, embed_i in zip(
            rel_pair_idxs, sub_reps, obj_reps, entity_embeds
        ):
            if pair_idx.numel() == 0:
                continue
            pair_idx = pair_idx.long()
            s_embed = self.W_sub(embed_i[pair_idx[:, 0]])
            o_embed = self.W_obj(embed_i[pair_idx[:, 1]])
            sem_sub = self.vis2sem(sub_i[pair_idx[:, 0]])
            sem_obj = self.vis2sem(obj_i[pair_idx[:, 1]])
            gate_sem_sub = torch.sigmoid(
                self.gate_sub(torch.cat([s_embed, sem_sub], dim=-1))
            )
            gate_sem_obj = torch.sigmoid(
                self.gate_obj(torch.cat([o_embed, sem_obj], dim=-1))
            )
            sub = s_embed + sem_sub * gate_sem_sub
            obj = o_embed + sem_obj * gate_sem_obj
            sub = self.norm_sub(
                self.dropout_sub(F.relu(self.linear_sub(sub))) + sub
            )
            obj = self.norm_obj(
                self.dropout_obj(F.relu(self.linear_obj(obj))) + obj
            )
            fusion_so_parts.append(self._fusion_func(sub, obj))

        if fusion_so_parts:
            fusion_so = torch.cat(fusion_so_parts, dim=0)
        else:
            fusion_so = obj_features.new_zeros((0, self.mlp_dim))
        if self.round11_feature_capture_enabled:
            pair_geometry = (
                self.pairwise_feature_extractor.round11_pair_bbox_geo_info()
            )
            if pair_geometry is None:
                raise RuntimeError("Round 11 pair geometry was not captured")
            self._round11_feature_state = {
                "rel_pair_idxs": (
                    torch.cat(rel_pair_idxs, dim=0).detach()
                    if rel_pair_idxs
                    else fusion_so.new_zeros((0, 2), dtype=torch.long)
                ),
                "fusion_so": fusion_so.detach(),
                "pair_bbox_geo_info": pair_geometry,
            }
        else:
            self._round11_feature_state = {}
        sem_pred = self.vis2sem(self.down_samp(rel_features))
        gate_sem_pred = torch.sigmoid(
            self.gate_pred(torch.cat([fusion_so, sem_pred], dim=-1))
        )
        rel_rep = fusion_so - sem_pred * gate_sem_pred

        flat_rel_labels = (
            torch.cat(rel_labels, dim=0).long().to(rel_rep.device)
            if (
                (self.training or bool(getattr(self, "allow_eval_relation_labels", False)))
                and rel_labels
            )
            else None
        )
        predicate_proto1 = self.W_pred(self.rel_embed.weight)
        rel_rep, method_losses = self._refine_relation_representation(
            rel_rep,
            predicate_proto1,
            flat_rel_labels,
            proposals,
            rel_pair_idxs,
        )
        add_losses.update(method_losses)
        method_relation_representation = rel_rep
        method_predicate_prototype = predicate_proto1
        predicate_proto2 = self._coarse_predicate_prototypes(predicate_proto1)
        rel_rep = self.norm_rel_rep(
            self.dropout_rel_rep(F.relu(self.linear_rel_rep(rel_rep))) + rel_rep
        )
        rel_rep = self.project_head(self.dropout_rel(F.relu(rel_rep)))
        predicate_proto1 = self.project_head(
            self.dropout_pred(F.relu(predicate_proto1))
        )
        predicate_proto2 = self.project_head(
            self.dropout_pred(F.relu(predicate_proto2))
        )

        if self.round13_representation_capture_enabled:
            self._round13_representation_state = {
                "rel_pair_idxs": (
                    torch.cat(rel_pair_idxs, dim=0).detach()
                    if rel_pair_idxs
                    else rel_rep.new_zeros((0, 2), dtype=torch.long)
                ),
                "rel_features": rel_features.detach(),
                "sem_pred": sem_pred.detach(),
                "rel_rep": rel_rep.detach(),
                "predicate_proto1": predicate_proto1.detach(),
                "logit_scale": self.logit_scale.detach(),
            }
        else:
            self._round13_representation_state = {}

        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)
        predicate_proto_norm1 = predicate_proto1 / predicate_proto1.norm(
            dim=1, keepdim=True
        )
        predicate_proto_norm2 = predicate_proto2 / predicate_proto2.norm(
            dim=1, keepdim=True
        )
        relation_logits = (
            rel_rep_norm @ predicate_proto_norm1.t()
        ) * self.logit_scale.exp()
        relation_logits = self._method_relation_logits(
            relation_logits,
            relation_representation=method_relation_representation,
            predicate_prototype=method_predicate_prototype,
        )
        if self.round13_representation_capture_enabled:
            if not isinstance(union_features, torch.Tensor):
                raise TypeError(
                    "Paper-reproduction capture requires fused union features"
                )
            self._round13_representation_state.update(
                {
                    "union_features": union_features.detach(),
                    "base_relation_logits": relation_logits.detach(),
                }
            )

        relation_logits, calibration_losses = self._calibrate_relation_logits(
            relation_logits,
            union_features=union_features,
            relation_representation=rel_rep,
            rel_labels=flat_rel_labels,
        )
        add_losses.update(calibration_losses)
        if self.training and self._original_prototype_losses_enabled():
            prototype_loss_labels = (
                flat_rel_labels
                if flat_rel_labels is not None
                else relation_logits.new_zeros((0,), dtype=torch.long)
            )
            add_losses.update(
                self._prototype_losses(
                    rel_rep,
                    predicate_proto1,
                    predicate_proto2,
                    predicate_proto_norm1,
                    predicate_proto_norm2,
                    prototype_loss_labels,
                )
            )
        relation_logits = list(relation_logits.split(num_rels, dim=0))
        refine_logits = list(entity_dists.split(num_objs, dim=0))
        return relation_logits, refine_logits, add_losses

    def forward(
        self,
        proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
    ):
        del rel_binarys, logger
        add_losses: dict[str, torch.Tensor] = {}
        self._record_geo_gate_values()
        _, initial_query_mask, _, initial_query_parts = self._r4_relation_masks(
            proposals, rel_pair_idxs, device=roi_features.device
        )
        if bool(initial_query_mask.all()):
            augment_obj_feat, rel_feats = self.pairwise_feature_extractor(
                roi_features,
                union_features,
                proposals,
                rel_pair_idxs,
            )
        else:
            if not torch.is_tensor(union_features):
                raise TypeError(
                    "R4 query/context pairwise isolation requires fused union features"
                )
            query_pairs = [
                pairs[mask.to(device=pairs.device)]
                for pairs, mask in zip(rel_pair_idxs, initial_query_parts)
            ]
            context_pairs = [
                pairs[~mask.to(device=pairs.device)]
                for pairs, mask in zip(rel_pair_idxs, initial_query_parts)
            ]
            augment_obj_feat, query_rel_feats = self.pairwise_feature_extractor(
                roi_features,
                union_features[initial_query_mask],
                proposals,
                query_pairs,
            )
            _, context_rel_feats = self.pairwise_feature_extractor(
                roi_features,
                union_features[~initial_query_mask],
                proposals,
                context_pairs,
            )
            rel_feats = union_features.new_empty(
                (initial_query_mask.numel(), query_rel_feats.size(1))
            )
            rel_feats[initial_query_mask] = query_rel_feats
            rel_feats[~initial_query_mask] = context_rel_feats
        num_rels = [pair_idx.shape[0] for pair_idx in rel_pair_idxs]
        if self.relation_entity_probe_enabled:
            if not torch.is_grad_enabled():
                raise RuntimeError(
                    "relation->entity sensitivity probe requires autograd enabled"
                )
            self._relation_entity_probe_gate = torch.ones(
                (rel_feats.size(0),),
                device=rel_feats.device,
                dtype=rel_feats.dtype,
                requires_grad=True,
            )
            self._relation_entity_probe_num_rels = tuple(map(int, num_rels))
        else:
            self._relation_entity_probe_gate = None
            self._relation_entity_probe_num_rels = tuple(map(int, num_rels))
        if self.graph_backend == "incidence":
            obj_features, rel_features = self._sgg_toolkit_incidence_propagation(
                augment_obj_feat,
                rel_feats,
                proposals,
                rel_pair_idxs,
            )
            _, query_mask, _, query_parts = self._r4_relation_masks(
                proposals, rel_pair_idxs, device=rel_features.device
            )
            if not bool(query_mask.all()):
                rel_features = rel_features[query_mask]
                rel_pair_idxs = [
                    pairs[mask.to(device=pairs.device)]
                    for pairs, mask in zip(rel_pair_idxs, query_parts)
                ]
                if rel_labels is not None:
                    rel_labels = [
                        labels[mask.to(device=labels.device)]
                        for labels, mask in zip(rel_labels, query_parts)
                    ]
                if torch.is_tensor(union_features):
                    union_features = union_features[query_mask]
                elif isinstance(union_features, (list, tuple)):
                    union_features = type(union_features)(
                        value[mask.to(device=value.device)]
                        for value, mask in zip(union_features, query_parts)
                    )
            return self._classify_propagated_features(
                obj_features=obj_features,
                rel_features=rel_features,
                proposals=proposals,
                rel_pair_idxs=rel_pair_idxs,
                rel_labels=rel_labels,
                union_features=union_features,
                add_losses=add_losses,
            )
        (
            subj_pred_map,
            obj_pred_map,
            pred_pred_subj,
            pred_pred_obj,
            obj_obj_map,
        ) = self._get_map_idxs(
            proposals, [pair_idx.clone() for pair_idx in rel_pair_idxs]
        )
        if self.relation_graph_mode == "sgg_toolkit":
            obj_obj_map = self._diagnostic_entity_adjacency(
                obj_obj_map, subj_pred_map, obj_pred_map
            )
        geo_ee_geometry = None
        if self.use_geo_ee:
            geo_ee_geometry = OBBGeometryEncoder.build_node_geometry(
                proposals,
                device=augment_obj_feat.device,
                dtype=augment_obj_feat.dtype,
            )
        role_adjacencies = self._role_relation_adjacencies(
            subj_pred_map, obj_pred_map
        )
        relation_semantic_group_ids = self._relation_semantic_group_ids(
            proposals, rel_pair_idxs, device=rel_feats.device
        )
        if relation_semantic_group_ids.numel() != rel_feats.size(0):
            raise RuntimeError(
                "Semantic relation groups do not align with relation features"
            )
        context_source_masks: list[torch.Tensor] = []
        for proposal, pair_idx in zip(proposals, rel_pair_idxs):
            if proposal.has_field("rpcm_context_source_mask"):
                mask = proposal.get_field("rpcm_context_source_mask").to(
                    device=rel_feats.device, dtype=torch.bool
                )
            else:
                mask = torch.ones(
                    (pair_idx.size(0),),
                    device=rel_feats.device,
                    dtype=torch.bool,
                )
            if mask.ndim != 1 or mask.numel() != pair_idx.size(0):
                raise ValueError(
                    "RPCM context-source mask must contain one value per "
                    f"relation query, got {tuple(mask.shape)} for "
                    f"{pair_idx.size(0)} pairs"
                )
            context_source_masks.append(mask)
        context_source_mask = (
            torch.cat(context_source_masks, dim=0)
            if context_source_masks
            else torch.ones(
                (rel_feats.size(0),), device=rel_feats.device, dtype=torch.bool
            )
        )
        if context_source_mask.numel() != rel_feats.size(0):
            raise RuntimeError(
                "RPCM context-source masks do not align with relation features"
            )
        all_context_sources_active = bool(context_source_mask.all())
        _, prediction_query_mask, _, _ = self._r4_relation_masks(
            proposals, rel_pair_idxs, device=rel_feats.device
        )
        if prediction_query_mask.numel() != rel_feats.size(0):
            raise RuntimeError(
                "RPCM prediction-query masks do not align with relation features"
            )
        all_relations_are_queries = bool(prediction_query_mask.all()) or bool(
            self.qg_full_relation_context
        )
        if all_relations_are_queries:
            query_role_adjacencies = role_adjacencies
        else:
            query_role_adjacencies = self._role_relation_adjacencies(
                subj_pred_map[:, prediction_query_mask],
                obj_pred_map[:, prediction_query_mask],
            )
        if self.relation_graph_mode == "sgg_toolkit" and not torch.equal(
            role_adjacencies["unified"].to(dtype=pred_pred_subj.dtype),
            pred_pred_subj,
        ):
            raise RuntimeError(
                "Four-role adjacency union does not reproduce Original RPCM "
                "unified adjacency."
            )

        obj_feats = [augment_obj_feat]
        pred_feats = [rel_feats]
        runtime_steps = getattr(self, "_runtime_feat_update_steps", None)
        propagation_steps = (
            self.feat_update_step if runtime_steps is None else int(runtime_steps)
        )
        if self.relation_graph_mode == "sgg_toolkit":
            for step in range(propagation_steps):
                t = len(obj_feats) - 1
                source_obj = self._collect_or_zero(
                    obj_feats[t], obj_feats[t], obj_obj_map, 4
                )
                if self.use_geo_ee:
                    source_obj = source_obj + self.geo_gate[t] * self._geo_ee_message(
                        t,
                        obj_feats[t],
                        obj_obj_map,
                        geo_ee_geometry,
                    )
                relation_source = pred_feats[t]
                if self._relation_entity_probe_gate is not None:
                    relation_source = relation_source * self._relation_entity_probe_gate[:, None]
                # Query/context-role decoupling: every relation remains in
                # pred_feats and is classified normally.  The mask applies
                # only when the same relation acts as a source for modes 0/1.
                # Slice both features and incidence columns so the published
                # neighborhood mean is normalized by the number of *active*
                # sources.  The all-true branch passes the original tensors
                # directly and therefore remains bitwise identical.
                if all_context_sources_active:
                    context_relation_source = relation_source
                    context_subj_map = subj_pred_map
                    context_obj_map = obj_pred_map
                    context_semantic_groups = relation_semantic_group_ids
                else:
                    context_relation_source = relation_source[
                        context_source_mask
                    ]
                    context_subj_map = subj_pred_map[:, context_source_mask]
                    context_obj_map = obj_pred_map[:, context_source_mask]
                    context_semantic_groups = relation_semantic_group_ids[
                        context_source_mask
                    ]
                source_rel_sub = self._relation_to_entity_message(
                    obj_feats[t], context_relation_source, context_subj_map,
                    context_semantic_groups, 0
                )
                source_rel_obj = self._relation_to_entity_message(
                    obj_feats[t], context_relation_source, context_obj_map,
                    context_semantic_groups, 1
                )
                source2obj_all = self._fuse_entity_sources(
                    t,
                    obj_feats[t],
                    source_obj,
                    source_rel_sub,
                    source_rel_obj,
                )
                obj_feats.append(
                    self.gcn_update_feat(obj_feats[t], source2obj_all, 0)
                )

                source_sub_rel = self._collect_or_zero(
                    pred_feats[t], obj_feats[t], subj_pred_map.t(), 2
                )
                source_obj_rel = self._collect_or_zero(
                    pred_feats[t], obj_feats[t], obj_pred_map.t(), 3
                )
                if all_relations_are_queries:
                    source_rel_rel = self._relation_to_relation_message(
                        pred_feats[t], role_adjacencies
                    )
                else:
                    query_rel_rel = self._relation_to_relation_message(
                        pred_feats[t][prediction_query_mask],
                        query_role_adjacencies,
                    )
                    source_rel_rel = pred_feats[t].new_zeros(pred_feats[t].shape)
                    source_rel_rel[prediction_query_mask] = query_rel_rel
                source2rel_all = (
                    source_sub_rel + source_obj_rel + source_rel_rel
                ) / 3.0
                pred_feats.append(
                    self.gcn_update_feat(pred_feats[t], source2rel_all, 1)
                )
            obj_features = obj_feats[-1]
            rel_features = pred_feats[-1]
        else:
            for t in range(propagation_steps):
                obj_feats.append(
                    self.gcn_ent2ent[t](obj_feats[t], obj_obj_map)
                )
                if pred_feats[t].numel() == 0:
                    pred_feats.append(pred_feats[t])
                    continue
                source_sub_rel = self.gcn_ent2rel[t](
                    pred_feats[t], obj_feats[t], subj_pred_map.t(), 0
                )
                source_obj_rel = self.gcn_ent2rel[t](
                    pred_feats[t], obj_feats[t], obj_pred_map.t(), 1
                )
                relation_sources = [source_sub_rel, source_obj_rel]
                if self.relation_graph_mode == "unified":
                    relation_sources.append(
                        self.gcn_rel2rel[t](
                            pred_feats[t], pred_pred_subj
                        )
                    )
                else:
                    if self.rel_subj_view_enabled:
                        relation_sources.append(
                            self.gcn_rel2rel[t](
                                pred_feats[t], pred_pred_subj
                            )
                        )
                    if self.rel_obj_view_enabled:
                        relation_sources.append(
                            self.gcn_rel2rel[t](
                                pred_feats[t], pred_pred_obj
                            )
                        )
                combined_source = relation_sources[0]
                for source in relation_sources[1:]:
                    combined_source = combined_source + source
                pred_feats.append(
                    combined_source / float(len(relation_sources))
                )

            rel_features = pred_feats[0]
            for rel_state in pred_feats[1:]:
                rel_features = rel_features + rel_state
            rel_features = rel_features / float(len(pred_feats))
            if self.exact_6850 and self.relation_graph_mode == "dual_view":
                obj_features = obj_feats[0]
                for obj_state in obj_feats[1:]:
                    obj_features = obj_features + obj_state
                obj_features = obj_features / float(len(obj_feats))
            else:
                obj_features = obj_feats[-1]

        entity_dists, entity_preds, _ = self._refine_obj_labels(
            obj_features, proposals
        )

        entity_rep = self.post_emb(obj_features).view(
            obj_features.size(0), 2, self.mlp_dim
        )
        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)
        entity_embeds = self.obj_embed(entity_preds)

        num_objs = [len(proposal) for proposal in proposals]
        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_embeds = entity_embeds.split(num_objs, dim=0)

        fusion_so_parts: list[torch.Tensor] = []
        for pair_idx, sub_i, obj_i, embed_i in zip(
            rel_pair_idxs, sub_reps, obj_reps, entity_embeds
        ):
            if pair_idx.numel() == 0:
                continue
            pair_idx = pair_idx.long()
            s_embed = self.W_sub(embed_i[pair_idx[:, 0]])
            o_embed = self.W_obj(embed_i[pair_idx[:, 1]])
            sem_sub = self.vis2sem(sub_i[pair_idx[:, 0]])
            sem_obj = self.vis2sem(obj_i[pair_idx[:, 1]])
            gate_sem_sub = torch.sigmoid(
                self.gate_sub(torch.cat([s_embed, sem_sub], dim=-1))
            )
            gate_sem_obj = torch.sigmoid(
                self.gate_obj(torch.cat([o_embed, sem_obj], dim=-1))
            )
            sub = s_embed + sem_sub * gate_sem_sub
            obj = o_embed + sem_obj * gate_sem_obj
            sub = self.norm_sub(
                self.dropout_sub(F.relu(self.linear_sub(sub))) + sub
            )
            obj = self.norm_obj(
                self.dropout_obj(F.relu(self.linear_obj(obj))) + obj
            )
            fusion_so_parts.append(self._fusion_func(sub, obj))

        if fusion_so_parts:
            fusion_so = torch.cat(fusion_so_parts, dim=0)
        else:
            fusion_so = obj_features.new_zeros((0, self.mlp_dim))
        if self.round11_feature_capture_enabled:
            pair_geometry = self.pairwise_feature_extractor.round11_pair_bbox_geo_info()
            if pair_geometry is None:
                raise RuntimeError("Round 11 pair geometry was not captured")
            self._round11_feature_state = {
                "rel_pair_idxs": (
                    torch.cat(rel_pair_idxs, dim=0).detach()
                    if rel_pair_idxs
                    else fusion_so.new_zeros((0, 2), dtype=torch.long)
                ),
                "fusion_so": fusion_so.detach(),
                "pair_bbox_geo_info": pair_geometry,
            }
        else:
            self._round11_feature_state = {}
        sem_pred = self.vis2sem(self.down_samp(rel_features))
        gate_sem_pred = torch.sigmoid(
            self.gate_pred(torch.cat([fusion_so, sem_pred], dim=-1))
        )
        rel_rep = fusion_so - sem_pred * gate_sem_pred

        flat_rel_labels = (
            torch.cat(rel_labels, dim=0).long().to(rel_rep.device)
            if (
                (self.training or bool(getattr(self, "allow_eval_relation_labels", False)))
                and rel_labels
            )
            else None
        )
        predicate_proto1 = self.W_pred(self.rel_embed.weight)
        rel_rep, method_losses = self._refine_relation_representation(
            rel_rep,
            predicate_proto1,
            flat_rel_labels,
            proposals,
            rel_pair_idxs,
        )
        add_losses.update(method_losses)
        method_relation_representation = rel_rep
        method_predicate_prototype = predicate_proto1
        predicate_proto2 = self._coarse_predicate_prototypes(
            predicate_proto1
        )
        rel_rep = self.norm_rel_rep(
            self.dropout_rel_rep(F.relu(self.linear_rel_rep(rel_rep))) + rel_rep
        )
        rel_rep = self.project_head(self.dropout_rel(F.relu(rel_rep)))
        predicate_proto1 = self.project_head(
            self.dropout_pred(F.relu(predicate_proto1))
        )
        predicate_proto2 = self.project_head(
            self.dropout_pred(F.relu(predicate_proto2))
        )

        if self.round13_representation_capture_enabled:
            self._round13_representation_state = {
                "rel_pair_idxs": (
                    torch.cat(rel_pair_idxs, dim=0).detach()
                    if rel_pair_idxs
                    else rel_rep.new_zeros((0, 2), dtype=torch.long)
                ),
                "rel_features": rel_features.detach(),
                "sem_pred": sem_pred.detach(),
                "rel_rep": rel_rep.detach(),
                "predicate_proto1": predicate_proto1.detach(),
                "logit_scale": self.logit_scale.detach(),
            }
        else:
            self._round13_representation_state = {}

        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)
        predicate_proto_norm1 = predicate_proto1 / predicate_proto1.norm(
            dim=1, keepdim=True
        )
        predicate_proto_norm2 = predicate_proto2 / predicate_proto2.norm(
            dim=1, keepdim=True
        )
        relation_logits = (
            rel_rep_norm @ predicate_proto_norm1.t()
        ) * self.logit_scale.exp()
        relation_logits = self._method_relation_logits(
            relation_logits,
            relation_representation=method_relation_representation,
            predicate_prototype=method_predicate_prototype,
        )

        # The optional capture is read by the frozen BAL cache exporter.  It
        # does not register parameters or alter the ordinary Original path.
        # Store the pre-calibration logits because BAL is defined as an
        # additive correction of the common SGG model's original prediction.
        if self.round13_representation_capture_enabled:
            if not isinstance(union_features, torch.Tensor):
                raise TypeError(
                    "Paper-reproduction capture requires fused union features"
                )
            self._round13_representation_state.update(
                {
                    "union_features": union_features.detach(),
                    "base_relation_logits": relation_logits.detach(),
                }
            )

        relation_logits, calibration_losses = self._calibrate_relation_logits(
            relation_logits,
            union_features=union_features,
            relation_representation=rel_rep,
            rel_labels=flat_rel_labels,
        )
        add_losses.update(calibration_losses)

        if self.training and self._original_prototype_losses_enabled():
            prototype_loss_labels = (
                flat_rel_labels
                if flat_rel_labels is not None
                else relation_logits.new_zeros((0,), dtype=torch.long)
            )
            add_losses.update(
                self._prototype_losses(
                    rel_rep,
                    predicate_proto1,
                    predicate_proto2,
                    predicate_proto_norm1,
                    predicate_proto_norm2,
                    prototype_loss_labels,
                )
            )

        relation_logits = list(relation_logits.split(num_rels, dim=0))
        refine_logits = list(entity_dists.split(num_objs, dim=0))
        return relation_logits, refine_logits, add_losses
