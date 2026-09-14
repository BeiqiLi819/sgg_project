from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from sgg.modeling.roi_heads.ppg import PairProposalGenerator
from sgg.modeling.roi_heads.pair_proposal_network import PairProposalNetworkFilter
from sgg.modeling.roi_heads.rsgp import RemoteSensingGraphProposalFilter
from sgg.modeling.roi_heads.q5_filter import Q5PairProposalFilter
from sgg.modeling.roi_heads.relation_inference import make_roi_relation_post_processor
from sgg.modeling.roi_heads.relation_loss import make_roi_relation_loss_evaluator
from sgg.modeling.roi_heads.relation_sampling import make_roi_relation_samp_processor
from sgg.modeling.roi_heads.sema_filter import SemanticPairFilter
from sgg.modeling.roi_heads.roi_relation_feature_extractors import (
    make_roi_box_feature_extractor,
    make_roi_relation_feature_extractor,
)
from sgg.modeling.roi_heads.roi_relation_predictors import make_roi_relation_predictor


def _relation_feature_dim(cfg: dict, default: int) -> int:
    return int(cfg["MODEL"].get("ROI_BOX_HEAD", {}).get("MLP_HEAD_DIM", default))


def _to_onehot_logits(labels: torch.Tensor, num_classes: int, fill: float = 1000.0) -> torch.Tensor:
    logits = labels.new_full((labels.numel(), num_classes), -fill, dtype=torch.float32)
    if labels.numel() > 0:
        row_idx = torch.arange(labels.numel(), device=labels.device)
        logits[row_idx, labels.long()] = fill
    return logits


def _normalize_sgcls_filter_label_source(value: object) -> str:
    source = str(value).strip().lower()
    if source not in {"pred", "gt"}:
        raise ValueError(
            "MODEL.ROI_RELATION_HEAD.SGCLS_FILTER_LABEL_SOURCE must be 'pred' or 'gt', "
            f"got {value!r}."
        )
    return source


class ROIRelationHead(nn.Module):
    """
    Generic relation head scaffold.

    This mirrors the reference architecture layout:
        sampling -> object/union feature extraction -> predictor ->
        postprocess/loss

    Predictor internals remain intentionally separate.
    """

    def __init__(self, cfg: dict, in_channels: int):
        super().__init__()
        self.cfg = cfg
        self.union_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)
        self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        feat_dim = _relation_feature_dim(cfg, self.box_feature_extractor.out_channels)
        self.obj_feature_dim = feat_dim
        self.local_box_feature_proj = (
            nn.Linear(self.box_feature_extractor.out_channels, feat_dim)
            if self.box_feature_extractor.out_channels != feat_dim
            else nn.Identity()
        )
        self.predictor = make_roi_relation_predictor(cfg, feat_dim)
        self.post_processor = make_roi_relation_post_processor(cfg)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)
        self.samp_processor = make_roi_relation_samp_processor(cfg)
        filter_method = str(cfg["MODEL"]["ROI_RELATION_HEAD"].get("TEST_FILTER_METHOD", "PPG")).upper()
        supported_filters = {"PPG", "PPN", "RSGP", "Q5", "ABS_PRD"}
        if filter_method not in supported_filters:
            raise ValueError(
                "MODEL.ROI_RELATION_HEAD.TEST_FILTER_METHOD must be one of "
                f"{sorted(supported_filters)} for STAR; got {filter_method!r}. "
                "The unfiltered all-pairs graph is disabled to prevent OOM."
            )
        self.filter_method = filter_method
        if filter_method == "Q5":
            self.ppg = Q5PairProposalFilter(cfg)
        elif filter_method == "RSGP":
            self.ppg = RemoteSensingGraphProposalFilter(cfg)
        elif filter_method == "PPN":
            self.ppg = PairProposalNetworkFilter(cfg)
        elif filter_method == "ABS_PRD":
            # AUG's dataset-native ABS graph is fitted offline using only the
            # outer-fold training images and serialized with each record.
            # It has no learned module and no global Top-K stage.
            self.ppg = None
        else:
            self.ppg = PairProposalGenerator(cfg)
        self.sema_filter = SemanticPairFilter(cfg)
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        self.type = cfg.get("TYPE", "CV")
        self.task = str(cfg["MODEL"].get("TASK", "sgdet")).lower()
        self.sgcls_filter_label_source = _normalize_sgcls_filter_label_source(
            rel_cfg.get("SGCLS_FILTER_LABEL_SOURCE", "pred")
        )
        self.use_union_box = bool(rel_cfg.get("PREDICT_USE_VISION", True))
        self.use_gt_box = bool(rel_cfg.get("USE_GT_BOX", False))
        self.use_gt_object_label = bool(rel_cfg.get("USE_GT_OBJECT_LABEL", False))
        self.predictor_name = str(rel_cfg.get("PREDICTOR", ""))
        self.legacy_filter_flow = bool(rel_cfg.get("RPCM_LEGACY_FILTER_FLOW", False))
        self.num_obj_classes = int(cfg["MODEL"]["ROI_BOX_HEAD"]["NUM_CLASSES"])
        self.train_pair_filter_enabled = bool(
            rel_cfg.get("TRAIN_PAIR_FILTER_ENABLED", False)
        )
        if self.train_pair_filter_enabled and not self.use_gt_box:
            raise ValueError(
                "TRAIN_PAIR_FILTER_ENABLED currently supports aligned GT-box "
                "PredCls/SGCls training only."
            )
        self.train_reasoning_graph_enabled = bool(
            rel_cfg.get("TRAIN_REASONING_GRAPH_ENABLED", False)
        )
        self.train_reasoning_pair_dir = str(
            rel_cfg.get("TRAIN_REASONING_PAIR_DIR", "")
        ).strip()
        self._train_reasoning_files: dict[int, Path] = {}
        self._last_train_qg_diagnostics: list[dict[str, int]] = []
        if self.train_reasoning_graph_enabled:
            if not self.use_gt_box:
                raise ValueError(
                    "TRAIN_REASONING_GRAPH_ENABLED currently requires aligned GT boxes"
                )
            if not self.train_reasoning_pair_dir:
                raise ValueError(
                    "TRAIN_REASONING_GRAPH_ENABLED requires TRAIN_REASONING_PAIR_DIR"
                )
            root = Path(self.train_reasoning_pair_dir)
            manifest_path = root / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Missing frozen train reasoning manifest: {manifest_path}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("split") != "train":
                raise ValueError("Training reasoning graph manifest must use train split")
            expected_digest = str(
                rel_cfg.get("TRAIN_REASONING_PAIR_DIGEST", "")
            ).strip()
            actual_digest = str(manifest.get("pair_digest", ""))
            if expected_digest and actual_digest != expected_digest:
                raise ValueError(
                    "Frozen train reasoning pair digest mismatch: "
                    f"expected={expected_digest}, actual={actual_digest}"
                )
            if bool(manifest.get("hard_quota_training_signal", True)):
                raise ValueError("Q6 reasoning graph must be HardQuota-free")
            for row in manifest.get("images", []):
                image_id = int(row["image_id"])
                if image_id in self._train_reasoning_files:
                    raise ValueError(
                        f"Duplicate train reasoning image_id={image_id}"
                    )
                self._train_reasoning_files[image_id] = root / Path(row["path"]).name
            print(
                "[TRAIN_QG] frozen reasoning graph: "
                f"{root} ({len(self._train_reasoning_files)} images, digest={actual_digest})",
                flush=True,
            )
            if str(getattr(self.predictor, "graph_backend", "dense")) != "incidence":
                raise ValueError(
                    "Q6 train reasoning graphs require RPCM_GRAPH_BACKEND='incidence' "
                    "so G-only logits/labels are sliced before supervision"
                )
            if not bool(getattr(self.predictor, "qg_full_relation_context", False)):
                raise ValueError(
                    "Q6 train reasoning graphs require RPCM_QG_FULL_RELATION_CONTEXT=True"
                )
        self.pair_override_artifact_dir = str(
            rel_cfg.get("PAIR_OVERRIDE_ARTIFACT_DIR", "")
        ).strip()
        self._pair_override_files: dict[int, Path] = {}
        if self.pair_override_artifact_dir:
            override_root = Path(self.pair_override_artifact_dir)
            manifest_path = override_root / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Missing frozen-pair manifest: {manifest_path}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for row in manifest.get("images", []):
                image_id = int(row["image_id"])
                artifact_path = override_root / Path(row["path"]).name
                if image_id in self._pair_override_files:
                    raise ValueError(
                        f"Duplicate image_id={image_id} in {manifest_path}"
                    )
                self._pair_override_files[image_id] = artifact_path
            print(
                "[PAIR_OVERRIDE] using frozen inference pair cache: "
                f"{override_root} ({len(self._pair_override_files)} images)",
                flush=True,
            )
        self.context_source_mode = str(
            rel_cfg.get("RPCM_CONTEXT_SOURCE_MODE", "all")
        ).strip().lower()
        if self.context_source_mode not in {"all", "annotated", "random"}:
            raise ValueError(
                "RPCM_CONTEXT_SOURCE_MODE must be all, annotated, or random; "
                f"got {self.context_source_mode!r}"
            )
        self.context_source_random_seed = int(
            rel_cfg.get("RPCM_CONTEXT_SOURCE_RANDOM_SEED", 0)
        )

    def set_training_sampling_context(self, *, global_step: int, epoch: int) -> None:
        """Set the local deterministic sampling key for this training batch."""

        setter = getattr(self.samp_processor, "set_training_sampling_context", None)
        if callable(setter):
            setter(global_step=int(global_step), epoch=int(epoch))

    def matched_sampling_diagnostics(self) -> dict[str, object]:
        getter = getattr(self.samp_processor, "matched_sampling_diagnostics", None)
        return getter() if callable(getter) else {"enabled": False}

    def train_qg_diagnostics(self) -> list[dict[str, int]]:
        return [dict(row) for row in self._last_train_qg_diagnostics]

    def _feature_device(self, features) -> torch.device:
        if isinstance(features, torch.Tensor):
            return features.device
        if isinstance(features, (list, tuple)):
            if not features:
                return torch.device("cpu")
            return features[0].device
        if isinstance(features, dict):
            if not features:
                return torch.device("cpu")
            first_key = next(iter(features))
            return features[first_key].device
        return torch.device("cpu")

    def _collect_refine_logits_from_proposals(self, proposals):
        refine_logits = []
        for proposal in proposals:
            if proposal.has_field("predict_logits"):
                refine_logits.append(proposal.get_field("predict_logits"))
            elif proposal.has_field("pred_logits"):
                refine_logits.append(proposal.get_field("pred_logits"))
        return refine_logits if refine_logits else None

    def _filter_labels_for_proposal(self, proposal):
        """Select semantic/proposal-filter labels without changing predictor inputs.

        The original SGG-Toolkit sgcls detector leaves ``proposal.labels`` as
        GT labels even though it attaches predicted labels separately.  The
        current project keeps predicted labels in that field for standard
        sgcls.  This method supports both filtering protocols while retaining
        ``pred_labels``/``predict_logits`` for the relation predictor.
        """
        if not proposal.has_field("labels"):
            return None, "missing"
        predicted_or_default = proposal.get_field("labels").long()
        if self.task != "sgcls" or self.sgcls_filter_label_source == "pred":
            return predicted_or_default, "pred"
        if proposal.has_field("gt_labels"):
            return proposal.get_field("gt_labels").long(), "gt"
        # Keep evaluation robust for externally constructed proposals while
        # making the missing legacy field visible in result diagnostics.
        return predicted_or_default, "pred_fallback_missing_gt"

    def _filter_test_pairs_for_proposal(self, proposal, pair_idx: torch.Tensor) -> torch.Tensor:
        """Apply semantic and learned pair filters under the selected label source."""
        if self.filter_method == "ABS_PRD":
            return self._dataset_native_pairs_for_proposal(proposal, pair_idx)
        filter_labels, resolved_source = self._filter_labels_for_proposal(proposal)
        proposal.add_field("filter_label_source", resolved_source)
        if filter_labels is None:
            proposal.add_field("sema_rel_pair_idxs", pair_idx)
            proposal.add_field("final_rel_pair_idxs", pair_idx)
            proposal.add_field("pruned_rel_pair_idxs", pair_idx)
            return pair_idx

        proposal.add_field("filter_labels", filter_labels)
        if self.sema_filter.enabled:
            pair_idx = self.sema_filter.filter_pairs(pair_idx, filter_labels)
        sema_pair_idx = pair_idx
        proposal.add_field("sema_rel_pair_idxs", sema_pair_idx)

        # PPG, PPN and RSGP all consume proposal.labels.  Temporarily expose
        # the selected filter labels only for their scoring path, then restore
        # the standard sgcls predicted labels used by the predictor/postprocess.
        original_labels = proposal.get_field("labels")
        swap_labels = filter_labels.data_ptr() != original_labels.data_ptr()
        if swap_labels:
            proposal.add_field("labels", filter_labels)
        try:
            if (
                self.legacy_filter_flow
                and self.filter_method == "RANDOM_FILTER"
                and sema_pair_idx.size(0) > self.ppg.threshold
            ):
                rand_idx = torch.randperm(sema_pair_idx.size(0), device=sema_pair_idx.device)
                filtered_pair_idx = sema_pair_idx[rand_idx[: self.ppg.topk]]
            elif self.ppg.filter_method in {"PPG", "PPN", "RSGP", "Q5"}:
                filtered_pair_idx = self.ppg.filter_pairs(proposal, sema_pair_idx)
            else:  # Constructor validates this; keep failure local if a filter mutates itself.
                raise RuntimeError(f"Unsupported active pair filter {self.ppg.filter_method!r}")
        finally:
            if swap_labels:
                proposal.add_field("labels", original_labels)

        proposal.add_field("final_rel_pair_idxs", filtered_pair_idx)
        proposal.add_field("pruned_rel_pair_idxs", filtered_pair_idx)
        return filtered_pair_idx

    def _dataset_native_pairs_for_proposal(
        self, proposal, legal_pair_idx: torch.Tensor
    ) -> torch.Tensor:
        """Read a leakage-safe dataset-native candidate graph.

        The field contains pair endpoints only.  It is produced from boxes,
        object classes and a train-only ABS dictionary; no relationship label
        or predicate answer is serialized into this path.
        """

        if not proposal.has_field("candidate_pairs"):
            raise KeyError(
                "ABS_PRD requires proposal.candidate_pairs generated from the "
                "outer-fold training dictionary"
            )
        pair_idx = proposal.get_field("candidate_pairs").to(
            device=legal_pair_idx.device, dtype=torch.long
        )
        if pair_idx.ndim != 2 or pair_idx.size(1) != 2:
            raise ValueError(
                f"Invalid dataset-native pair shape {tuple(pair_idx.shape)}"
            )
        if pair_idx.numel() > 0:
            n = len(proposal)
            if bool((pair_idx < 0).any()) or bool((pair_idx >= n).any()):
                raise ValueError("Dataset-native graph has an illegal endpoint")
            if bool((pair_idx[:, 0] == pair_idx[:, 1]).any()):
                raise ValueError("Dataset-native graph has a self pair")
            encoded = pair_idx[:, 0] * max(n, 1) + pair_idx[:, 1]
            if torch.unique(encoded).numel() != encoded.numel():
                raise ValueError("Dataset-native graph has duplicate pairs")
            legal_encoded = (
                legal_pair_idx[:, 0] * max(n, 1) + legal_pair_idx[:, 1]
            )
            if not torch.isin(encoded, legal_encoded).all():
                raise ValueError(
                    "Dataset-native graph is not a subset of legal candidates"
                )
            pair_idx = pair_idx[torch.argsort(encoded, stable=True)]
        proposal.add_field("sema_rel_pair_idxs", pair_idx)
        proposal.add_field("final_rel_pair_idxs", pair_idx)
        proposal.add_field("pruned_rel_pair_idxs", pair_idx)
        proposal.add_field("filter_label_source", "train_only_abs_prd")
        return pair_idx

    @staticmethod
    def _proposal_image_id(proposal, fallback: int) -> int:
        if proposal.has_field("image_id"):
            value = proposal.get_field("image_id")
            if torch.is_tensor(value) and value.numel() > 0:
                return int(value.reshape(-1)[0].item())
            return int(value)
        return int(fallback)

    def _attach_context_source_masks(
        self,
        proposals,
        targets,
        rel_pair_idxs,
    ) -> None:
        """Attach diagnostic mode-0/1 source masks without changing queries."""

        if self.context_source_mode != "all" and self.training:
            raise RuntimeError(
                "annotated/random context-source masking is frozen inference only"
            )
        if self.context_source_mode != "all" and targets is None:
            raise ValueError(
                "annotated/random context-source diagnosis requires PredCls targets"
            )
        for batch_index, (proposal, pair_idx) in enumerate(
            zip(proposals, rel_pair_idxs)
        ):
            edge_count = int(pair_idx.size(0))
            if proposal.has_field("rpcm_context_source_mask"):
                if self.context_source_mode != "all":
                    raise RuntimeError(
                        "Frozen R4 context masks cannot be combined with the "
                        "annotated/random diagnostic source modes"
                    )
                mask = proposal.get_field("rpcm_context_source_mask").to(
                    device=pair_idx.device, dtype=torch.bool
                )
                if mask.ndim != 1 or mask.numel() != edge_count:
                    raise ValueError(
                        "Frozen R4 context mask must align with U relations"
                    )
                proposal.add_field("rpcm_context_source_mask", mask)
                proposal.add_field(
                    "rpcm_context_source_count",
                    pair_idx.new_tensor([int(mask.sum().item())]),
                )
                proposal.add_field(
                    "rpcm_annotated_context_count",
                    pair_idx.new_tensor([-1]),
                )
                continue
            mask = torch.ones(
                (edge_count,), dtype=torch.bool, device=pair_idx.device
            )
            positive_count = edge_count
            if self.context_source_mode != "all":
                target = targets[batch_index]
                relation_matrix = self.samp_processor._get_relation_matrix(
                    target
                ).to(device=pair_idx.device)
                if edge_count:
                    positive_mask = relation_matrix[
                        pair_idx[:, 0].long(), pair_idx[:, 1].long()
                    ].gt(0)
                else:
                    positive_mask = mask
                positive_count = int(positive_mask.sum().item())
                if self.context_source_mode == "annotated":
                    mask = positive_mask
                else:
                    mask = torch.zeros_like(positive_mask)
                    if positive_count:
                        image_id = self._proposal_image_id(
                            proposal, batch_index
                        )
                        # Keep the sampling independent of batch order/device.
                        # The constants are fixed odd integers, not Python's
                        # process-randomized hash().
                        mixed_seed = (
                            int(self.context_source_random_seed)
                            + 1_000_003 * (int(image_id) + 1)
                            + 97_409
                        ) % (2**63 - 1)
                        generator = torch.Generator(device="cpu")
                        generator.manual_seed(mixed_seed)
                        selected = torch.randperm(
                            edge_count, generator=generator
                        )[:positive_count]
                        mask[selected.to(device=mask.device)] = True
            proposal.add_field("rpcm_context_source_mask", mask)
            proposal.add_field(
                "rpcm_context_source_count",
                pair_idx.new_tensor([int(mask.sum().item())]),
            )
            proposal.add_field(
                "rpcm_annotated_context_count",
                pair_idx.new_tensor([positive_count]),
            )

    def _training_query_reasoning_union(
        self,
        proposal,
        query_pairs: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return U=Q∪G while retaining labels only for Q.

        Q order and labels remain exactly as produced by the ordinary sampler.
        Frozen G-only rows are appended in canonical endpoint order.  Masks
        route all U features through reasoning but let the predictor return
        only Q logits before any supervised loss is constructed.
        """

        image_id = self._proposal_image_id(proposal, -1)
        path = self._train_reasoning_files.get(image_id)
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"Missing Q6 frozen train graph for image_id={image_id}"
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        reasoning_pairs = payload["prediction"]["rel_pair_idxs"].long().to(
            query_pairs.device
        )
        num_objects = len(proposal)
        for name, pairs in (("Q", query_pairs), ("G", reasoning_pairs)):
            if pairs.ndim != 2 or pairs.size(1) != 2:
                raise ValueError(f"Q6 {name} pairs must have shape [E,2]")
            if pairs.numel() and (
                bool((pairs < 0).any())
                or bool((pairs >= num_objects).any())
                or bool((pairs[:, 0] == pairs[:, 1]).any())
            ):
                raise ValueError(f"Q6 {name} graph has an illegal endpoint/self pair")
        width = max(num_objects, 1)
        query_codes = query_pairs[:, 0] * width + query_pairs[:, 1]
        reasoning_codes = reasoning_pairs[:, 0] * width + reasoning_pairs[:, 1]
        if torch.unique(query_codes).numel() != query_codes.numel():
            raise ValueError("Q6 sampled query graph contains duplicate pairs")
        if torch.unique(reasoning_codes).numel() != reasoning_codes.numel():
            raise ValueError("Q6 frozen reasoning graph contains duplicate pairs")
        reasoning_order = torch.argsort(reasoning_codes, stable=True)
        reasoning_pairs = reasoning_pairs[reasoning_order]
        reasoning_codes = reasoning_codes[reasoning_order]
        g_only_mask = ~torch.isin(reasoning_codes, query_codes)
        g_only_pairs = reasoning_pairs[g_only_mask]
        union_pairs = torch.cat((query_pairs, g_only_pairs), dim=0)
        query_mask = torch.zeros(
            (union_pairs.size(0),), dtype=torch.bool, device=query_pairs.device
        )
        query_mask[: query_pairs.size(0)] = True
        context_mask = torch.cat(
            (
                torch.isin(query_codes, reasoning_codes),
                torch.ones(
                    (g_only_pairs.size(0),),
                    dtype=torch.bool,
                    device=query_pairs.device,
                ),
            )
        )
        # G-only labels are inert placeholders.  The incidence Original
        # predictor slices logits and labels by query_mask before every
        # supervised classifier/prototype loss.
        union_labels = torch.cat(
            (
                query_labels,
                torch.zeros(
                    (g_only_pairs.size(0),),
                    dtype=query_labels.dtype,
                    device=query_labels.device,
                ),
            )
        )
        proposal.add_field("rpcm_prediction_query_mask", query_mask)
        proposal.add_field("rpcm_context_source_mask", context_mask)
        proposal.add_field(
            "rpcm_q_count", query_pairs.new_tensor([query_pairs.size(0)])
        )
        proposal.add_field(
            "rpcm_g_count", query_pairs.new_tensor([reasoning_pairs.size(0)])
        )
        proposal.add_field(
            "rpcm_u_count", query_pairs.new_tensor([union_pairs.size(0)])
        )
        self._last_train_qg_diagnostics.append(
            {
                "image_id": int(image_id),
                "Q": int(query_pairs.size(0)),
                "G": int(reasoning_pairs.size(0)),
                "U": int(union_pairs.size(0)),
                "Q_intersect_G": int(context_mask[: query_pairs.size(0)].sum()),
                "G_only": int(g_only_pairs.size(0)),
            }
        )
        return union_pairs, union_labels

    def _frozen_pairs_for_proposal(
        self, proposal, legal_pair_idx: torch.Tensor
    ) -> torch.Tensor:
        if not proposal.has_field("image_id"):
            raise ValueError("Frozen pair override requires proposal.image_id")
        image_id_value = proposal.get_field("image_id")
        image_id = int(image_id_value.reshape(-1)[0].item())
        artifact_path = self._pair_override_files.get(image_id)
        if artifact_path is None or not artifact_path.is_file():
            raise FileNotFoundError(
                f"No frozen pair artifact for validation image_id={image_id}"
            )
        # Only prediction.rel_pair_idxs is read.  Target/GT fields stored in a
        # qualitative artifact are deliberately ignored.
        payload = torch.load(artifact_path, map_location="cpu")
        prediction_payload = payload["prediction"]
        pair_idx = prediction_payload["rel_pair_idxs"].long()
        query_mask = prediction_payload.get("rpcm_prediction_query_mask")
        context_mask = prediction_payload.get("rpcm_context_source_mask")
        if query_mask is None:
            query_mask = torch.ones((pair_idx.size(0),), dtype=torch.bool)
        else:
            query_mask = torch.as_tensor(query_mask, dtype=torch.bool)
        if context_mask is None:
            context_mask = torch.ones((pair_idx.size(0),), dtype=torch.bool)
        else:
            context_mask = torch.as_tensor(context_mask, dtype=torch.bool)
        for name, mask in (
            ("prediction query", query_mask),
            ("context source", context_mask),
        ):
            if mask.ndim != 1 or mask.numel() != pair_idx.size(0):
                raise ValueError(
                    f"Frozen R4 {name} mask has shape {tuple(mask.shape)} for "
                    f"{pair_idx.size(0)} U relations"
                )
        if pair_idx.ndim != 2 or pair_idx.size(1) != 2:
            raise ValueError(
                f"Invalid frozen pair shape for image_id={image_id}: "
                f"{tuple(pair_idx.shape)}"
            )
        pair_idx = pair_idx.to(device=legal_pair_idx.device)
        query_mask = query_mask.to(device=legal_pair_idx.device)
        context_mask = context_mask.to(device=legal_pair_idx.device)
        if pair_idx.numel() > 0:
            if (pair_idx < 0).any() or (pair_idx >= len(proposal)).any():
                raise ValueError(
                    f"Frozen pair cache has illegal endpoints for image_id={image_id}"
                )
            if (pair_idx[:, 0] == pair_idx[:, 1]).any():
                raise ValueError(
                    f"Frozen pair cache has self-loops for image_id={image_id}"
                )
            encoded = pair_idx[:, 0] * max(len(proposal), 1) + pair_idx[:, 1]
            if torch.unique(encoded).numel() != encoded.numel():
                raise ValueError(
                    f"Frozen pair cache has duplicates for image_id={image_id}"
                )
            legal_encoded = (
                legal_pair_idx[:, 0] * max(len(proposal), 1)
                + legal_pair_idx[:, 1]
            )
            if not torch.isin(encoded, legal_encoded).all():
                raise ValueError(
                    f"Frozen pair cache is not a subset of legal candidates for "
                    f"image_id={image_id}"
                )
            # Candidate graphs are sets.  Canonicalizing their tensor order
            # prevents GPU reduction round-off from being mistaken for a
            # context effect when two cache producers serialized the same set
            # in different ranking orders.
            if not bool(payload.get("r4_preserve_relation_order", False)):
                canonical_order = torch.argsort(encoded, stable=True)
                pair_idx = pair_idx[canonical_order]
                query_mask = query_mask[canonical_order]
                context_mask = context_mask[canonical_order]
        if int(query_mask.sum()) != int(context_mask.sum()):
            raise ValueError(
                "Frozen R4 requires equal prediction-query and context budgets"
            )
        proposal.add_field(
            "rpcm_prediction_query_mask",
            query_mask.to(device=legal_pair_idx.device),
        )
        proposal.add_field(
            "rpcm_context_source_mask",
            context_mask.to(device=legal_pair_idx.device),
        )
        proposal.add_field("sema_rel_pair_idxs", pair_idx)
        proposal.add_field("final_rel_pair_idxs", pair_idx)
        proposal.add_field("pruned_rel_pair_idxs", pair_idx)
        proposal.add_field("filter_label_source", "frozen_pair_cache")
        return pair_idx

    def forward(
        self,
        features,
        proposals: Sequence,
        targets=None,
        logger=None,
        OBj=None,
        s_f=None,
        **kwargs,
    ):
        del s_f, kwargs
        if self.training:
            with torch.no_grad():
                if self.train_pair_filter_enabled:
                    if targets is None:
                        raise ValueError(
                            "Filtered relation training requires targets for labels."
                        )
                    rel_pair_idxs = self.samp_processor.prepare_test_pairs(
                        self._feature_device(features), proposals
                    )
                    for proposal, pair_idx in zip(proposals, rel_pair_idxs):
                        proposal.add_field("base_rel_pair_idxs", pair_idx)
                    rel_pair_idxs = [
                        self._filter_test_pairs_for_proposal(proposal, pair_idx)
                        for proposal, pair_idx in zip(proposals, rel_pair_idxs)
                    ]
                    labeled = [
                        self.samp_processor.label_gtbox_pairs(
                            proposal, target, pair_idx
                        )
                        for proposal, target, pair_idx in zip(
                            proposals, targets, rel_pair_idxs
                        )
                    ]
                    rel_labels = [item[0] for item in labeled]
                    rel_binarys = [item[1] for item in labeled]
                elif self.use_gt_box:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.gtbox_relsample(proposals, targets)
                else:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.detect_relsample(proposals, targets)
                if self.train_reasoning_graph_enabled:
                    self._last_train_qg_diagnostics = []
                    union = [
                        self._training_query_reasoning_union(
                            proposal, pair_idx, labels
                        )
                        for proposal, pair_idx, labels in zip(
                            proposals, rel_pair_idxs, rel_labels
                        )
                    ]
                    rel_pair_idxs = [item[0] for item in union]
                    rel_labels = [item[1] for item in union]
        else:
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(self._feature_device(features), proposals)
            for proposal, pair_idx in zip(proposals, rel_pair_idxs):
                proposal.add_field("base_rel_pair_idxs", pair_idx)
            if self._pair_override_files:
                rel_pair_idxs = [
                    self._frozen_pairs_for_proposal(proposal, pair_idx)
                    for proposal, pair_idx in zip(proposals, rel_pair_idxs)
                ]
            else:
                rel_pair_idxs = [
                    self._filter_test_pairs_for_proposal(proposal, pair_idx)
                    for proposal, pair_idx in zip(proposals, rel_pair_idxs)
                ]
            if bool(getattr(self.predictor, "allow_eval_relation_labels", False)):
                if targets is None:
                    raise ValueError(
                        "HPL GT-tail oracle is diagnostic-only and requires targets"
                    )
                if not self.use_gt_box:
                    raise ValueError(
                        "HPL GT-tail oracle currently supports aligned GT-box tasks only"
                    )
                labeled = [
                    self.samp_processor.label_gtbox_pairs(
                        proposal, target, pair_idx
                    )
                    for proposal, target, pair_idx in zip(
                        proposals, targets, rel_pair_idxs
                    )
                ]
                rel_labels = [item[0] for item in labeled]
                rel_binarys = [item[1] for item in labeled]

        if self.use_gt_box and self.use_gt_object_label and (
            self.predictor_name in {"RPCM", "RPCM_LEGACY", "LEGACY_RPCM"}
        ):
            for proposal in proposals:
                labels = proposal.get_field("labels").long().clamp(min=0, max=self.num_obj_classes - 1)
                predict_logits = _to_onehot_logits(labels, num_classes=self.num_obj_classes)
                proposal.add_field("predict_logits", predict_logits.to(proposal.bbox.device))
                proposal.add_field("pred_scores", torch.ones(len(labels), device=proposal.bbox.device))
                proposal.add_field("pred_labels", labels.to(proposal.bbox.device))

        if OBj is not None and hasattr(OBj, "bbox_roi_extractor") and hasattr(OBj, "bbox_head"):
            roi_feats = OBj.bbox_roi_extractor(features, list(proposals))
            roi_features = OBj.bbox_head(roi_feats)
        else:
            roi_features = self.box_feature_extractor(features, proposals)
            roi_features = self.local_box_feature_proj(roi_features)

        union_features = (
            self.union_feature_extractor(features, proposals, rel_pair_idxs, OBj=OBj)
            if self.use_union_box
            else None
        )

        self._attach_context_source_masks(
            proposals, targets, rel_pair_idxs
        )

        predictor_output = self.predictor(
            proposals,
            rel_pair_idxs,
            rel_labels,
            rel_binarys,
            roi_features,
            union_features,
            logger,
        )
        if not isinstance(predictor_output, tuple):
            raise TypeError("Relation predictor must return a tuple.")
        if len(predictor_output) == 2:
            relation_logits, add_losses = predictor_output
            refine_logits = self._collect_refine_logits_from_proposals(proposals)
        elif len(predictor_output) == 3:
            relation_logits, refine_logits, add_losses = predictor_output
        else:
            raise ValueError(
                "Relation predictor must return (relation_logits, add_losses) "
                "or (relation_logits, refine_logits, add_losses)."
            )

        if not self.training:
            if hasattr(self.predictor, "hpl_edge_diagnostics"):
                edge_diagnostics = self.predictor.hpl_edge_diagnostics()
                expected_edges = sum(int(pair.size(0)) for pair in rel_pair_idxs)
                for field, flat_value in edge_diagnostics.items():
                    if flat_value.size(0) != expected_edges:
                        raise RuntimeError(
                            f"HPL edge diagnostic {field!r} has "
                            f"{flat_value.size(0)} rows for {expected_edges} relations"
                        )
                    for proposal, value in zip(
                        proposals,
                        flat_value.split(
                            [int(pair.size(0)) for pair in rel_pair_idxs], dim=0
                        ),
                    ):
                        proposal.add_field(field, value)
            # R4 may append G\Q context-only relations to the graph.  They
            # participate in modes 0/1 and receive modes 2/3 updates, but must
            # never enter final prediction/ranking.  Slice only after the
            # unchanged classifier has produced U-aligned logits.
            if any(
                proposal.has_field("rpcm_prediction_query_mask")
                for proposal in proposals
            ):
                predictor_filtered = bool(
                    getattr(self.predictor, "r4_filters_prediction_queries", False)
                )
                if not predictor_filtered:
                    relation_logits = self.post_processor._split_relation_logits(
                        relation_logits, rel_pair_idxs
                    )
                query_pairs = []
                query_logits = []
                for batch_index, (proposal, pair_idx, rel_logit) in enumerate(
                    zip(proposals, rel_pair_idxs, relation_logits)
                ):
                    if proposal.has_field("rpcm_prediction_query_mask"):
                        query_mask = proposal.get_field(
                            "rpcm_prediction_query_mask"
                        ).to(device=pair_idx.device, dtype=torch.bool)
                    else:
                        query_mask = torch.ones(
                            (pair_idx.size(0),),
                            device=pair_idx.device,
                            dtype=torch.bool,
                        )
                    query_pairs.append(pair_idx[query_mask])
                    query_logits.append(
                        rel_logit if predictor_filtered else rel_logit[query_mask]
                    )
                    if proposal.has_field("rpcm_context_source_mask"):
                        context_mask = proposal.get_field(
                            "rpcm_context_source_mask"
                        ).to(device=pair_idx.device, dtype=torch.bool)
                        proposal.add_field(
                            "rpcm_context_source_mask", context_mask[query_mask]
                        )
                    proposal.add_field(
                        "rpcm_prediction_query_mask",
                        torch.ones(
                            (int(query_mask.sum()),),
                            device=pair_idx.device,
                            dtype=torch.bool,
                        ),
                    )
                rel_pair_idxs = query_pairs
                relation_logits = query_logits
            result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals)
            return roi_features, result, {}

        if self.training and self.train_reasoning_graph_enabled:
            rel_labels = [
                labels[
                    proposal.get_field("rpcm_prediction_query_mask").to(
                        device=labels.device, dtype=torch.bool
                    )
                ]
                for proposal, labels in zip(proposals, rel_labels)
            ]
        loss_relation, loss_refine_obj = self.loss_evaluator(
            proposals,
            rel_labels,
            relation_logits,
            refine_logits=refine_logits,
        )
        output_losses = {"loss_rel": loss_relation}
        if loss_refine_obj is not None:
            output_losses["loss_refine_obj"] = loss_refine_obj
        output_losses.update(add_losses)
        return roi_features, proposals, output_losses


def build_roi_relation_head(cfg: dict, in_channels: int):
    return ROIRelationHead(cfg, in_channels)
