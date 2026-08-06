from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Sequence

import torch
from torch import nn

from sgg.data.rsgp_statistics import load_rsgp_structural_prior
from sgg.modeling.core.obb_ops import angle_to_radians, get_boxlist_angle_unit
from sgg.modeling.roi_heads.pair_proposal_network import PairProposalNetworkFilter
from sgg.modeling.roi_heads.ppg import PairProposalGenerator
from sgg.structures.boxes import BoxList


def _cfg_get(cfg: dict, *keys, default=None):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _norm_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _as_float(rel_cfg: dict, key: str, default: float) -> float:
    return float(rel_cfg.get(key, default))


def _as_int(rel_cfg: dict, key: str, default: int) -> int:
    return int(rel_cfg.get(key, default))


def _as_bool(rel_cfg: dict, key: str, default: bool) -> bool:
    value = rel_cfg.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _zscore(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if values.numel() == 0:
        return values
    out = values.float().clone()
    if mask is not None and mask.any():
        active = out[mask]
    else:
        active = out
    if active.numel() <= 1:
        return out.zero_()
    mean = active.mean()
    std = active.std(unbiased=False).clamp(min=1e-6)
    out = (out - mean) / std
    if mask is not None:
        out = torch.where(mask, out, out.new_zeros(()))
    return out


class RemoteSensingGraphProposalFilter(nn.Module):
    """Remote-sensing graph-aware pair proposal used only at inference time.

    RSGP deliberately optimizes a candidate relation graph rather than a pure
    pairness top-k.  It can reuse PPG as a high-precision protected pool, PPN as
    a recall-completion pool, and light-weight OBB/topology priors as graph
    scoring signals.  The final greedy stage controls in/out degree and label
    pair density before RPCM's dense relation-to-relation GCN sees the graph.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        self.filter_method = str(rel_cfg.get("TEST_FILTER_METHOD", "PPG")).upper()
        # TEST_FILTER_METHOD is the single source of truth.  RSGP_ENABLED is a
        # legacy/diagnostic mirror and is not required to activate this filter.
        self.enabled = self.filter_method == "RSGP"
        self.mode = str(rel_cfg.get("RSGP_MODE", "HYBRID")).upper()
        self.threshold = _as_int(rel_cfg, "RSGP_THRESHOLD", _as_int(rel_cfg, "PPG_PAIR_THRESHOLD", 10000))
        self.topk = _as_int(rel_cfg, "RSGP_TOPK", int(rel_cfg.get("TEST_FILTER_TOPK", 10000)))
        self.chunk_size = max(1, _as_int(rel_cfg, "RSGP_CHUNK_SIZE", 200000))
        self.ppg_protected_topk = _as_int(rel_cfg, "RSGP_PPG_PROTECTED_TOPK", 9000)
        self.ppn_pool_topk = _as_int(rel_cfg, "RSGP_PPN_POOL_TOPK", 12000)
        self.rs_pool_topk = _as_int(rel_cfg, "RSGP_RS_POOL_TOPK", 12000)
        self.max_out_degree = _as_int(rel_cfg, "RSGP_MAX_OUT_DEGREE", 96)
        self.max_in_degree = _as_int(rel_cfg, "RSGP_MAX_IN_DEGREE", 96)
        self.relaxed_max_degree = _as_int(rel_cfg, "RSGP_RELAXED_MAX_DEGREE", 128)
        self.label_pair_quota = _as_int(rel_cfg, "RSGP_LABEL_PAIR_QUOTA", 800)
        self.relaxed_label_pair_quota = _as_int(rel_cfg, "RSGP_RELAXED_LABEL_PAIR_QUOTA", 1200)

        self.w_ppg = _as_float(rel_cfg, "RSGP_W_PPG", 1.0)
        self.w_ppn = _as_float(rel_cfg, "RSGP_W_PPN", 0.35)
        self.w_geom = _as_float(rel_cfg, "RSGP_W_GEOM", 0.35)
        self.w_anchor = _as_float(rel_cfg, "RSGP_W_ANCHOR", 0.25)
        self.w_topo = _as_float(rel_cfg, "RSGP_W_TOPO", 0.20)
        self.w_tail = _as_float(rel_cfg, "RSGP_W_TAIL", 0.15)
        self.w_degree = _as_float(rel_cfg, "RSGP_W_DEGREE", 0.15)
        self.use_ppn_completion = _as_bool(rel_cfg, "RSGP_USE_PPN_COMPLETION", True)
        self.use_geometry = _as_bool(rel_cfg, "RSGP_USE_GEOMETRY", True)
        self.use_anchor = _as_bool(rel_cfg, "RSGP_USE_ANCHOR", True)
        self.use_topology = _as_bool(rel_cfg, "RSGP_USE_TOPOLOGY", True)
        self.use_tail_prior = _as_bool(rel_cfg, "RSGP_USE_TAIL_PRIOR", True)
        self.use_degree_score = _as_bool(rel_cfg, "RSGP_USE_DEGREE_SCORE", True)
        self.enforce_degree_cap = _as_bool(rel_cfg, "RSGP_ENFORCE_DEGREE_CAP", True)
        self.enforce_label_quota = _as_bool(rel_cfg, "RSGP_ENFORCE_LABEL_QUOTA", True)

        self.class_names = list(_cfg_get(cfg, "MODEL", "ROI_BOX_HEAD", "CLASS_NAMES", default=[]))
        self.predicate_names = list(
            rel_cfg.get("RELATION_NAMES", _cfg_get(cfg, "MODEL", "RELATION_NAMES", default=[]))
        )
        self.role_mode = str(rel_cfg.get("RSGP_ROLE_MODE", "statistical")).strip().lower()
        if self.role_mode not in {"statistical", "legacy_manual"}:
            raise ValueError(
                "RSGP_ROLE_MODE must be 'statistical' or 'legacy_manual', "
                f"got {self.role_mode!r}."
            )
        self.use_context_role = _as_bool(
            rel_cfg,
            "RSGP_USE_CONTEXT_ROLE",
            _as_bool(rel_cfg, "RSGP_USE_ANCHOR", True),
        )
        self.use_alignment_role = _as_bool(
            rel_cfg,
            "RSGP_USE_ALIGNMENT_ROLE",
            _as_bool(rel_cfg, "RSGP_USE_TOPOLOGY", True),
        )
        self.use_connectivity_role = _as_bool(
            rel_cfg,
            "RSGP_USE_CONNECTIVITY_ROLE",
            _as_bool(rel_cfg, "RSGP_USE_TOPOLOGY", True),
        )
        self.use_rarity_prior = _as_bool(
            rel_cfg,
            "RSGP_USE_RARITY_PRIOR",
            _as_bool(rel_cfg, "RSGP_USE_TAIL_PRIOR", True),
        )
        self.context_carrier_min = max(1, _as_int(rel_cfg, "RSGP_CONTEXT_CARRIER_MIN", 16))
        self.context_carrier_max = max(
            self.context_carrier_min,
            _as_int(rel_cfg, "RSGP_CONTEXT_CARRIER_MAX", 128),
        )
        self.context_carrier_scale = max(
            0.0,
            _as_float(rel_cfg, "RSGP_CONTEXT_CARRIER_SCALE", 2.0),
        )
        self.w_context = _as_float(rel_cfg, "RSGP_W_CONTEXT", 0.25)
        self.w_alignment = _as_float(rel_cfg, "RSGP_W_ALIGNMENT", 0.10)
        self.w_connectivity = _as_float(rel_cfg, "RSGP_W_CONNECTIVITY", 0.10)
        self.w_rarity = _as_float(rel_cfg, "RSGP_W_RARITY", 0.15)

        self.anchor_class_ids: tuple[int, ...] = ()
        self.vehicle_class_ids: tuple[int, ...] = ()
        self.network_class_ids: tuple[int, ...] = ()
        self.tail_predicates: tuple[int, ...] = ()
        self.register_buffer("tail_pair_support", torch.zeros((0, 0), dtype=torch.bool), persistent=False)
        self.register_buffer(
            "class_context_profile",
            torch.zeros((0,), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "class_alignment_profile",
            torch.zeros((0,), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "class_connectivity_profile",
            torch.zeros((0,), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "rarity_pair_support",
            torch.zeros((0, 0), dtype=torch.float32),
            persistent=False,
        )
        self.structural_prior_hash = ""
        if self.role_mode == "statistical":
            self._load_structural_prior(rel_cfg)
        else:
            self._initialize_legacy_manual_roles(rel_cfg)

        self.ppg = self._make_ppg(cfg) if self.mode in {"HYBRID"} else None
        self.ppn = (
            self._make_ppn(cfg)
            if self.use_ppn_completion and self.mode in {"HYBRID", "PPN_GRAPH"}
            else None
        )
        print(
            "[RSGP] "
            f"mode={self.mode}, topk={self.topk}, ppg_protected={self.ppg_protected_topk}, "
            f"ppn_pool={self.ppn_pool_topk}, rs_pool={self.rs_pool_topk}, "
            f"degree={self.max_out_degree}/{self.max_in_degree}, "
            f"components=ppn:{self.use_ppn_completion},geom:{self.use_geometry},"
            f"context:{self.use_context_role},alignment:{self.use_alignment_role},"
            f"connectivity:{self.use_connectivity_role},rarity:{self.use_rarity_prior},"
            f"degree_cap:{self.enforce_degree_cap},quota:{self.enforce_label_quota}, "
            f"role_mode={self.role_mode}, prior_hash={self.structural_prior_hash or '<legacy>'}",
            flush=True,
        )

    def _initialize_legacy_manual_roles(self, rel_cfg: dict) -> None:
        self.anchor_class_ids = self._find_class_ids(
            rel_cfg.get(
                "RSGP_ANCHOR_CLASSES",
                "apron,truck_parking,car_parking,dock,runway,taxiway,breakwater,goods_yard",
            )
        )
        self.vehicle_class_ids = self._find_class_ids(
            rel_cfg.get(
                "RSGP_VEHICLE_CLASSES",
                "airplane,aircraft,vehicle,car,truck,ship,boat,bus,van",
            )
        )
        self.network_class_ids = self._find_class_ids(
            rel_cfg.get(
                "RSGP_NETWORK_CLASSES",
                "tower,lattice_tower,substation,genset,transmission_line,power_line,line,pole",
            )
        )
        self.tail_predicates = tuple(
            int(v)
            for v in rel_cfg.get(
                "RSGP_TAIL_PREDICATES",
                (7, 14, 20, 24, 25, 28, 31, 33, 36, 38, 39, 41, 53, 56, 58),
            )
        )
        self._load_tail_support(rel_cfg)

    def _load_structural_prior(self, rel_cfg: dict) -> None:
        if not self.class_names or not self.predicate_names:
            raise RuntimeError(
                "Statistical RSGP requires resolved class and predicate metadata before "
                "the relation head is constructed."
            )
        path = str(
            rel_cfg.get(
                "RSGP_STRUCTURAL_PRIOR_PATH",
                "pretrained/rsgp_structural_prior.json",
            )
        )
        payload = load_rsgp_structural_prior(
            path,
            class_names=self.class_names,
            predicate_names=self.predicate_names,
            expected_hash=str(rel_cfg.get("RSGP_STRUCTURAL_PRIOR_HASH", "")),
        )
        profiles = payload["class_profiles"]
        self.class_context_profile = torch.tensor(
            profiles["contextual_region"],
            dtype=torch.float32,
        )
        self.class_alignment_profile = torch.tensor(
            profiles["directional_alignment"],
            dtype=torch.float32,
        )
        self.class_connectivity_profile = torch.tensor(
            profiles["relational_connectivity"],
            dtype=torch.float32,
        )
        self.rarity_pair_support = torch.tensor(
            payload["rarity_pair_support"],
            dtype=torch.float32,
        )
        self.structural_prior_hash = str(payload["payload_hash"])

    def _find_class_ids(self, names: str | Sequence[str]) -> tuple[int, ...]:
        if not self.class_names:
            return ()
        tokens = names.split(",") if isinstance(names, str) else list(names)
        patterns = [_norm_name(token) for token in tokens if str(token).strip()]
        ids = []
        for idx, class_name in enumerate(self.class_names):
            if idx == 0:
                continue
            norm = _norm_name(class_name)
            if any(pattern and pattern in norm for pattern in patterns):
                ids.append(idx)
        return tuple(sorted(set(ids)))

    def _load_tail_support(self, rel_cfg: dict) -> None:
        path = Path(str(rel_cfg.get("SEMA_F_PATH", "")))
        if not path.is_file() or not self.tail_predicates:
            return
        try:
            prior = torch.tensor(json.loads(path.read_text(encoding="utf-8")), dtype=torch.float32)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return
        if prior.ndim != 3:
            return
        predicate_ids = [p for p in self.tail_predicates if 0 <= p < prior.size(2)]
        if not predicate_ids:
            return
        support = prior[:, :, predicate_ids].sum(dim=2) > 0
        self.tail_pair_support = support.bool()

    def _make_ppg(self, cfg: dict) -> PairProposalGenerator:
        ppg_cfg = copy.deepcopy(cfg)
        rel_cfg = ppg_cfg["MODEL"]["ROI_RELATION_HEAD"]
        rel_cfg["TEST_FILTER_METHOD"] = "PPG"
        rel_cfg["PPG_TOPK"] = max(self.topk, self.ppg_protected_topk)
        return PairProposalGenerator(ppg_cfg)

    def _make_ppn(self, cfg: dict) -> PairProposalNetworkFilter:
        ppn_cfg = copy.deepcopy(cfg)
        rel_cfg = ppn_cfg["MODEL"]["ROI_RELATION_HEAD"]
        rel_cfg["TEST_FILTER_METHOD"] = "PPN"
        rel_cfg["PPN_TOPK"] = max(self.topk, self.ppn_pool_topk)
        return PairProposalNetworkFilter(ppn_cfg)

    def should_filter(self, pair_idx: torch.Tensor) -> bool:
        return (
            self.filter_method == "RSGP"
            and pair_idx.numel() > 0
            and pair_idx.size(0) > self.threshold
            and self.topk > 0
        )

    @torch.no_grad()
    def filter_pairs(self, proposal: BoxList, pair_idx: torch.Tensor) -> torch.Tensor:
        if not self.should_filter(pair_idx):
            self._add_stage_fields(proposal, pair_idx, pair_idx, pair_idx)
            return pair_idx

        if self.role_mode == "statistical":
            self._prepare_statistical_roles(proposal, pair_idx)
        ppg_pairs = self._ppg_pairs(proposal, pair_idx)
        ppn_pairs, ppn_scores = self._ppn_pairs_and_scores(proposal, pair_idx)
        rs_pairs, _ = self._topk_by_rs_score(proposal, pair_idx, self.rs_pool_topk)

        candidate_pool = self._unique_pairs(
            [pairs for pairs in (ppg_pairs, ppn_pairs, rs_pairs) if pairs.numel() > 0],
            device=pair_idx.device,
        )
        if candidate_pool.numel() == 0:
            candidate_pool = pair_idx

        hybrid_scores = self._hybrid_scores(
            proposal=proposal,
            candidate_pairs=candidate_pool,
            ppg_pairs=ppg_pairs,
            ppn_pairs=ppn_pairs,
            ppn_pair_scores=ppn_scores,
        )
        order = hybrid_scores.argsort(descending=True)
        ranked_pool = candidate_pool[order]
        ranked_scores = hybrid_scores[order]

        protected = ppg_pairs[: min(self.ppg_protected_topk, ppg_pairs.size(0))]
        selected, degree_capped = self._graph_greedy_select(
            proposal,
            ranked_pool,
            protected_pairs=protected,
            scores=ranked_scores,
        )
        self._add_stage_fields(proposal, ranked_pool, degree_capped, selected)
        return selected

    def _add_stage_fields(
        self,
        proposal: BoxList,
        candidate_pool: torch.Tensor,
        degree_capped: torch.Tensor,
        selected: torch.Tensor,
    ) -> None:
        proposal.add_field("ppn_ranked_pair_idxs", candidate_pool.detach())
        proposal.add_field("degree_capped_pair_idxs", degree_capped.detach())
        proposal.add_field(
            "rsgp_stats",
            {
                "candidate_pool": int(candidate_pool.size(0)),
                "degree_capped": int(degree_capped.size(0)),
                "selected": int(selected.size(0)),
                "topk": int(self.topk),
                "mode": self.mode,
                "role_mode": self.role_mode,
                "structural_prior_hash": self.structural_prior_hash,
            },
        )

    def _ppg_pairs(self, proposal: BoxList, pair_idx: torch.Tensor) -> torch.Tensor:
        if self.ppg is None or not self.ppg.loaded:
            return pair_idx.new_zeros((0, 2))
        pairs = self.ppg.filter_pairs(proposal, pair_idx)
        return pairs[: min(max(self.topk, self.ppg_protected_topk), pairs.size(0))]

    def _ppn_pairs_and_scores(
        self, proposal: BoxList, pair_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ppn is None or not self.ppn.loaded or self.ppn.model is None:
            return pair_idx.new_zeros((0, 2)), proposal.bbox.new_zeros((0,))
        model = self.ppn.model
        model_device = next(model.parameters()).device
        if model_device != proposal.bbox.device:
            model.to(proposal.bbox.device)
        encoded = model.encode_entities(proposal)
        keep_pairs = pair_idx.new_zeros((0, 2))
        keep_scores = proposal.bbox.new_zeros((0,))
        pool_k = max(self.topk, self.ppn_pool_topk)
        for start in range(0, pair_idx.size(0), self.chunk_size):
            pairs = pair_idx[start:start + self.chunk_size]
            scores = model.pair_outputs_from_encoded(encoded, pairs)[0]
            all_pairs = torch.cat((keep_pairs, pairs))
            all_scores = torch.cat((keep_scores, scores))
            k = min(pool_k, all_scores.numel())
            keep_scores, selected = all_scores.topk(k, sorted=False)
            keep_pairs = all_pairs[selected]
        order = keep_scores.argsort(descending=True)
        keep_pairs = keep_pairs[order]
        keep_scores = keep_scores[order]
        return keep_pairs[: self.ppn_pool_topk], keep_scores[: self.ppn_pool_topk]

    def _topk_by_rs_score(
        self, proposal: BoxList, pair_idx: torch.Tensor, topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if topk <= 0 or pair_idx.numel() == 0:
            return pair_idx.new_zeros((0, 2)), proposal.bbox.new_zeros((0,))
        keep_pairs = pair_idx.new_zeros((0, 2))
        keep_scores = proposal.bbox.new_zeros((0,))
        for start in range(0, pair_idx.size(0), self.chunk_size):
            pairs = pair_idx[start:start + self.chunk_size]
            scores = self._rs_scores_for_pairs(proposal, pairs)["rs_total"]
            all_pairs = torch.cat((keep_pairs, pairs))
            all_scores = torch.cat((keep_scores, scores))
            k = min(topk, all_scores.numel())
            keep_scores, selected = all_scores.topk(k, sorted=False)
            keep_pairs = all_pairs[selected]
        order = keep_scores.argsort(descending=True)
        return keep_pairs[order], keep_scores[order]

    def _hybrid_scores(
        self,
        proposal: BoxList,
        candidate_pairs: torch.Tensor,
        ppg_pairs: torch.Tensor,
        ppn_pairs: torch.Tensor,
        ppn_pair_scores: torch.Tensor,
    ) -> torch.Tensor:
        parts = self._rs_scores_for_pairs(proposal, candidate_pairs)
        geom = _zscore(parts["geom"])
        degree = (
            _zscore(self._degree_balance_scores(proposal, candidate_pairs))
            if self.use_degree_score
            else candidate_pairs.new_zeros((candidate_pairs.size(0),), dtype=torch.float32)
        )
        ppg = self._rank_score(candidate_pairs, ppg_pairs, high_is_good=True)
        ppn = self._source_score(candidate_pairs, ppn_pairs, ppn_pair_scores)
        ppg = _zscore(ppg, ppg.ne(0))
        ppn = _zscore(ppn, ppn.ne(0))
        if self.role_mode == "statistical":
            context = _zscore(parts["context"], parts["context"].ne(0))
            alignment = _zscore(parts["alignment"], parts["alignment"].ne(0))
            connectivity = _zscore(parts["connectivity"], parts["connectivity"].ne(0))
            rarity = parts["rarity"]
            return (
                self.w_ppg * ppg
                + self.w_ppn * ppn
                + self.w_geom * geom
                + self.w_context * context
                + self.w_alignment * alignment
                + self.w_connectivity * connectivity
                + self.w_rarity * rarity
                + self.w_degree * degree
            )

        anchor = _zscore(parts["anchor"], parts["anchor"].ne(0))
        topo = _zscore(parts["topo"], parts["topo"].ne(0))
        tail = parts["tail"]
        return (
            self.w_ppg * ppg
            + self.w_ppn * ppn
            + self.w_geom * geom
            + self.w_anchor * anchor
            + self.w_topo * topo
            + self.w_tail * tail
            + self.w_degree * degree
        )

    def _degree_balance_scores(self, proposal: BoxList, pairs: torch.Tensor) -> torch.Tensor:
        if pairs.numel() == 0:
            return proposal.bbox.new_zeros((0,))
        num_nodes = len(proposal)
        h = pairs[:, 0].long()
        t = pairs[:, 1].long()
        out_counts = torch.bincount(h, minlength=num_nodes).float().to(pairs.device)
        in_counts = torch.bincount(t, minlength=num_nodes).float().to(pairs.device)
        # Lower preliminary endpoint congestion is better for RPCM's dense
        # relation graph, where very high-degree nodes create noisy rel-rel maps.
        return -torch.log1p(out_counts[h] + in_counts[t])

    def _prepare_statistical_roles(
        self,
        proposal: BoxList,
        semantic_pairs: torch.Tensor,
    ) -> None:
        num_entities = len(proposal)
        device = proposal.bbox.device
        if num_entities == 0:
            empty = proposal.bbox.new_zeros((0,))
            proposal.add_field("rsgp_context_role", empty)
            proposal.add_field("rsgp_alignment_role", empty)
            proposal.add_field("rsgp_connectivity_role", empty)
            proposal.add_field("rsgp_context_assignment", empty.long())
            proposal.add_field("rsgp_context_confidence", empty)
            proposal.add_field("rsgp_context_carrier_mask", empty.bool())
            return

        labels = proposal.get_field("labels").long().to(device)
        num_profile_classes = int(self.class_context_profile.numel())
        if labels.numel() and (
            int(labels.min()) < 0 or int(labels.max()) >= num_profile_classes
        ):
            raise RuntimeError(
                "RSGP proposal label is outside the structural-prior class range: "
                f"valid=[0,{num_profile_classes - 1}], "
                f"actual=[{int(labels.min())},{int(labels.max())}]."
            )
        context_prior = self.class_context_profile.to(device)[labels]
        alignment_prior = self.class_alignment_profile.to(device)[labels]
        connectivity_prior = self.class_connectivity_profile.to(device)[labels]
        centers, sizes, angles = self._box_state(proposal)
        centers, sizes, angles = centers.to(device), sizes.to(device), angles.to(device)

        areas = sizes.prod(dim=1)
        order = areas.argsort(stable=True)
        area_rank = areas.new_empty((num_entities,))
        area_rank[order] = (
            torch.arange(num_entities, device=device, dtype=areas.dtype) + 0.5
        ) / max(num_entities, 1)
        elongation = 1.0 - (
            sizes.min(dim=1).values / sizes.max(dim=1).values.clamp(min=1e-6)
        )

        if semantic_pairs.numel() > 0:
            heads = semantic_pairs[:, 0].long()
            tails = semantic_pairs[:, 1].long()
            semantic_degree = (
                torch.bincount(heads, minlength=num_entities)
                + torch.bincount(tails, minlength=num_entities)
            ).float()
            semantic_degree = torch.log1p(semantic_degree) / max(
                math.log1p(max(2 * (num_entities - 1), 1)),
                1e-6,
            )
        else:
            semantic_degree = areas.new_zeros((num_entities,))

        preliminary_context = 0.5 * (context_prior + area_rank)
        carrier_count = min(
            num_entities,
            self.context_carrier_max,
            max(
                self.context_carrier_min,
                int(math.ceil(self.context_carrier_scale * math.sqrt(num_entities))),
            ),
        )
        carrier_indices = preliminary_context.topk(carrier_count).indices
        carrier_centers = centers[carrier_indices]
        carrier_sizes = sizes[carrier_indices].clamp(min=1e-6)
        carrier_angles = angles[carrier_indices]

        delta = centers.unsqueeze(0) - carrier_centers.unsqueeze(1)
        cos = carrier_angles.cos().unsqueeze(1)
        sin = carrier_angles.sin().unsqueeze(1)
        local_x = delta[:, :, 0] * cos + delta[:, :, 1] * sin
        local_y = -delta[:, :, 0] * sin + delta[:, :, 1] * cos
        inside = (
            (local_x.abs() <= carrier_sizes[:, None, 0] * 0.5)
            & (local_y.abs() <= carrier_sizes[:, None, 1] * 0.5)
        )
        carrier_rows = torch.arange(carrier_count, device=device)
        inside[carrier_rows, carrier_indices] = False
        containment = inside.float().sum(dim=1) / max(num_entities - 1, 1)
        carrier_instance_context = 0.5 * (
            area_rank[carrier_indices] + containment
        )
        carrier_role = 0.5 * (
            context_prior[carrier_indices] + carrier_instance_context
        )
        context_role = preliminary_context.clone()
        context_role[carrier_indices] = carrier_role

        distances = torch.cdist(centers, carrier_centers)
        carrier_diag = carrier_sizes.square().sum(dim=1).sqrt().clamp(min=1e-6)
        proximity = torch.exp(
            -(distances / carrier_diag.unsqueeze(0)).clamp(max=20)
        )
        affinity = torch.maximum(proximity, inside.transpose(0, 1).float())
        affinity = affinity * carrier_role.unsqueeze(0)
        context_confidence, context_assignment = affinity.max(dim=1)
        carrier_mask = torch.zeros((num_entities,), dtype=torch.bool, device=device)
        carrier_mask[carrier_indices] = True

        alignment_role = 0.5 * (alignment_prior + elongation)
        connectivity_role = 0.5 * (connectivity_prior + semantic_degree)
        proposal.add_field("rsgp_context_role", context_role.clamp(0, 1).detach())
        proposal.add_field("rsgp_alignment_role", alignment_role.clamp(0, 1).detach())
        proposal.add_field(
            "rsgp_connectivity_role",
            connectivity_role.clamp(0, 1).detach(),
        )
        proposal.add_field("rsgp_context_assignment", context_assignment.detach())
        proposal.add_field("rsgp_context_confidence", context_confidence.detach())
        proposal.add_field("rsgp_context_carrier_mask", carrier_mask.detach())

    def _statistical_role_scores(
        self,
        proposal: BoxList,
        labels: torch.Tensor,
        centers: torch.Tensor,
        sizes: torch.Tensor,
        angles: torch.Tensor,
        pairs: torch.Tensor,
        dist_norm: torch.Tensor,
        delta: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        h, t = pairs[:, 0].long(), pairs[:, 1].long()
        zero = centers.new_zeros((pairs.size(0),))
        if not proposal.has_field("rsgp_alignment_role"):
            self._prepare_statistical_roles(proposal, pairs)

        if self.use_context_role:
            assignment = proposal.get_field("rsgp_context_assignment").to(pairs.device)
            confidence = proposal.get_field("rsgp_context_confidence").to(pairs.device)
            carrier_mask = proposal.get_field("rsgp_context_carrier_mask").to(pairs.device)
            same_context = (assignment[h] == assignment[t]).float() * confidence[h] * confidence[t]
            different_context = (
                (assignment[h] != assignment[t]).float()
                * confidence[h]
                * confidence[t]
                * 0.6
            )
            endpoint_context = (
                (carrier_mask[h] | carrier_mask[t]).float()
                * torch.maximum(confidence[h], confidence[t])
                * 0.4
            )
            context = torch.maximum(
                torch.maximum(same_context, different_context),
                endpoint_context,
            )
        else:
            context = zero

        if self.use_alignment_role:
            alignment_role = proposal.get_field("rsgp_alignment_role").to(pairs.device)
            alignment_gate = torch.sqrt(
                (alignment_role[h] * alignment_role[t]).clamp(min=0)
            )
            angle_diff = torch.atan2(
                torch.sin(angles[h] - angles[t]),
                torch.cos(angles[h] - angles[t]),
            ).abs()
            parallel = torch.cos(angle_diff).abs()
            direction = torch.stack((torch.cos(angles[h]), torch.sin(angles[h])), dim=1)
            along = (delta * direction).sum(dim=1).abs()
            lateral = (
                delta[:, 0] * direction[:, 1] - delta[:, 1] * direction[:, 0]
            ).abs()
            lateral_score = torch.exp(
                -(lateral / sizes[h].mean(dim=1).clamp(min=1e-6)).clamp(max=20)
            )
            along_score = torch.exp(
                -(along / sizes[h].square().sum(1).sqrt().clamp(min=1e-6)).clamp(max=20)
                * 0.25
            )
            alignment = alignment_gate * (
                0.55 * parallel + 0.30 * lateral_score + 0.15 * along_score
            )
        else:
            alignment = zero

        if self.use_connectivity_role:
            connectivity_role = proposal.get_field("rsgp_connectivity_role").to(pairs.device)
            connectivity_gate = torch.sqrt(
                (connectivity_role[h] * connectivity_role[t]).clamp(min=0)
            )
            connectivity = connectivity_gate * torch.exp(
                -0.5 * dist_norm.clamp(max=20)
            )
        else:
            connectivity = zero

        if self.use_rarity_prior and self.rarity_pair_support.numel() > 0:
            support = self.rarity_pair_support.to(pairs.device)
            if labels.numel() and (
                int(labels.min()) < 0
                or int(labels.max()) >= min(support.size(0), support.size(1))
            ):
                raise RuntimeError(
                    "RSGP proposal label is outside the rarity-support class range."
                )
            subj_labels = labels[h].long()
            obj_labels = labels[t].long()
            rarity = support[subj_labels, obj_labels]
        else:
            rarity = zero
        return {
            "context": context,
            "alignment": alignment,
            "connectivity": connectivity,
            "rarity": rarity,
        }

    def _rs_scores_for_pairs(self, proposal: BoxList, pairs: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = pairs.device
        if pairs.numel() == 0:
            empty = proposal.bbox.new_zeros((0,))
            if self.role_mode == "statistical":
                return {
                    "geom": empty,
                    "context": empty,
                    "alignment": empty,
                    "connectivity": empty,
                    "rarity": empty,
                    "rs_total": empty,
                }
            return {
                "geom": empty,
                "anchor": empty,
                "topo": empty,
                "tail": empty,
                "rs_total": empty,
            }
        labels = proposal.get_field("labels").long().to(device)
        centers, sizes, angles = self._box_state(proposal)
        centers, sizes, angles = centers.to(device), sizes.to(device), angles.to(device)
        h, t = pairs[:, 0].long(), pairs[:, 1].long()
        eps = 1e-6
        delta = centers[t] - centers[h]
        dist = delta.square().sum(dim=1).sqrt()
        hdiag = sizes[h].square().sum(dim=1).sqrt().clamp(min=eps)
        tdiag = sizes[t].square().sum(dim=1).sqrt().clamp(min=eps)
        dist_norm = dist / (0.5 * (hdiag + tdiag)).clamp(min=eps)

        harea = sizes[h].prod(dim=1).clamp(min=eps)
        tarea = sizes[t].prod(dim=1).clamp(min=eps)
        h_min, h_max = centers[h] - sizes[h] / 2, centers[h] + sizes[h] / 2
        t_min, t_max = centers[t] - sizes[t] / 2, centers[t] + sizes[t] / 2
        inter = (torch.minimum(h_max, t_max) - torch.maximum(h_min, t_min)).clamp(min=0).prod(dim=1)
        union = (harea + tarea - inter).clamp(min=eps)
        iou = inter / union
        union_box_area = (
            (torch.maximum(h_max, t_max) - torch.minimum(h_min, t_min)).clamp(min=eps).prod(dim=1)
        )
        compact = ((harea + tarea) / union_box_area).clamp(max=2.0)
        angle_diff = torch.atan2(torch.sin(angles[h] - angles[t]), torch.cos(angles[h] - angles[t])).abs()
        parallel = torch.cos(angle_diff).abs()
        close = torch.exp(-dist_norm.clamp(min=0, max=20))
        geom = 0.35 * iou + 0.30 * close + 0.20 * compact + 0.15 * parallel
        if not self.use_geometry:
            geom = geom.zero_()

        if self.role_mode == "statistical":
            roles = self._statistical_role_scores(
                proposal,
                labels,
                centers,
                sizes,
                angles,
                pairs,
                dist_norm,
                delta,
            )
            rs_total = (
                geom
                + 0.7 * roles["context"]
                + 0.3 * roles["alignment"]
                + 0.3 * roles["connectivity"]
                + 0.35 * roles["rarity"]
            )
            return {"geom": geom, **roles, "rs_total": rs_total}

        anchor = (
            self._anchor_scores(labels, centers, sizes, pairs)
            if self.use_anchor
            else geom.new_zeros(geom.shape)
        )
        topo = (
            self._topology_scores(labels, centers, sizes, angles, pairs, dist_norm, delta)
            if self.use_topology
            else geom.new_zeros(geom.shape)
        )
        tail = (
            self._tail_scores(labels, pairs)
            if self.use_tail_prior
            else geom.new_zeros(geom.shape)
        )
        rs_total = geom + 0.7 * anchor + 0.6 * topo + 0.35 * tail
        return {"geom": geom, "anchor": anchor, "topo": topo, "tail": tail, "rs_total": rs_total}

    def _box_state(self, proposal: BoxList) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes = proposal.bbox.float()
        if proposal.mode == "xywha":
            centers = boxes[:, :2]
            sizes = boxes[:, 2:4].clamp(min=1e-4)
            angles = angle_to_radians(boxes[:, 4], get_boxlist_angle_unit(proposal))
        else:
            xyxy = proposal.convert("xyxy").bbox.float()
            centers = 0.5 * (xyxy[:, :2] + xyxy[:, 2:])
            sizes = (xyxy[:, 2:] - xyxy[:, :2]).clamp(min=1e-4)
            angles = boxes.new_zeros((len(proposal),))
        scale = boxes.new_tensor([max(float(proposal.size[0]), 1.0), max(float(proposal.size[1]), 1.0)])
        return centers / scale, sizes / scale, angles

    def _anchor_scores(
        self,
        labels: torch.Tensor,
        centers: torch.Tensor,
        sizes: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        if not self.anchor_class_ids:
            return centers.new_zeros((pairs.size(0),))
        anchor_mask = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in self.anchor_class_ids:
            anchor_mask |= labels == int(class_id)
        if not anchor_mask.any():
            return centers.new_zeros((pairs.size(0),))
        anchor_centers = centers[anchor_mask]
        anchor_sizes = sizes[anchor_mask].clamp(min=1e-6)
        distances = torch.cdist(centers, anchor_centers)
        nearest_dist, nearest_idx = distances.min(dim=1)
        nearest_center = anchor_centers[nearest_idx]
        nearest_size = anchor_sizes[nearest_idx]
        inside = ((centers - nearest_center).abs() <= nearest_size / 2).all(dim=1).float()
        confidence = torch.exp(-(nearest_dist / nearest_size.square().sum(1).sqrt().clamp(min=1e-6)).clamp(max=20))
        confidence = torch.maximum(confidence, inside)
        h, t = pairs[:, 0].long(), pairs[:, 1].long()
        same_anchor = (nearest_idx[h] == nearest_idx[t]).float() * confidence[h] * confidence[t]
        different_anchor = (nearest_idx[h] != nearest_idx[t]).float() * confidence[h] * confidence[t] * 0.6
        endpoint_is_anchor = (anchor_mask[h] | anchor_mask[t]).float() * 0.4
        return torch.maximum(torch.maximum(same_anchor, different_anchor), endpoint_is_anchor)

    def _topology_scores(
        self,
        labels: torch.Tensor,
        centers: torch.Tensor,
        sizes: torch.Tensor,
        angles: torch.Tensor,
        pairs: torch.Tensor,
        dist_norm: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        h, t = pairs[:, 0].long(), pairs[:, 1].long()
        score = centers.new_zeros((pairs.size(0),))
        if self.vehicle_class_ids:
            vehicle_mask = torch.zeros_like(labels, dtype=torch.bool)
            for class_id in self.vehicle_class_ids:
                vehicle_mask |= labels == int(class_id)
            both_vehicle = vehicle_mask[h] & vehicle_mask[t]
            if both_vehicle.any():
                angle_diff = torch.atan2(torch.sin(angles[h] - angles[t]), torch.cos(angles[h] - angles[t])).abs()
                parallel = torch.cos(angle_diff).abs()
                direction = torch.stack((torch.cos(angles[h]), torch.sin(angles[h])), dim=1)
                along = (delta * direction).sum(dim=1).abs()
                lateral = (delta[:, 0] * direction[:, 1] - delta[:, 1] * direction[:, 0]).abs()
                lateral_score = torch.exp(-(lateral / sizes[h].mean(dim=1).clamp(min=1e-6)).clamp(max=20))
                along_score = torch.exp(-(along / sizes[h].square().sum(1).sqrt().clamp(min=1e-6)).clamp(max=20) * 0.25)
                score = torch.where(both_vehicle, 0.55 * parallel + 0.30 * lateral_score + 0.15 * along_score, score)
        if self.network_class_ids:
            network_mask = torch.zeros_like(labels, dtype=torch.bool)
            for class_id in self.network_class_ids:
                network_mask |= labels == int(class_id)
            both_network = network_mask[h] & network_mask[t]
            if both_network.any():
                network_score = torch.exp(-dist_norm.clamp(max=20) * 0.5)
                score = torch.where(both_network, torch.maximum(score, network_score), score)
        return score

    def _tail_scores(self, labels: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        if self.tail_pair_support.numel() == 0:
            return labels.new_zeros((pairs.size(0),), dtype=torch.float32)
        support = self.tail_pair_support.to(device=pairs.device)
        h = labels[pairs[:, 0]].long().clamp(min=0, max=support.size(0) - 1)
        t = labels[pairs[:, 1]].long().clamp(min=0, max=support.size(1) - 1)
        return support[h, t].float()

    def _rank_score(
        self,
        target_pairs: torch.Tensor,
        source_pairs: torch.Tensor,
        *,
        high_is_good: bool,
    ) -> torch.Tensor:
        if target_pairs.numel() == 0 or source_pairs.numel() == 0:
            return target_pairs.new_zeros((target_pairs.size(0),), dtype=torch.float32)
        scores = {}
        denom = max(int(source_pairs.size(0) - 1), 1)
        for rank, (head, tail) in enumerate(source_pairs.detach().cpu().tolist()):
            value = 1.0 - rank / denom if high_is_good else rank / denom
            scores[(int(head), int(tail))] = float(value)
        return target_pairs.new_tensor(
            [scores.get((int(h), int(t)), 0.0) for h, t in target_pairs.detach().cpu().tolist()],
            dtype=torch.float32,
        )

    def _source_score(
        self,
        target_pairs: torch.Tensor,
        source_pairs: torch.Tensor,
        source_scores: torch.Tensor,
    ) -> torch.Tensor:
        if target_pairs.numel() == 0 or source_pairs.numel() == 0 or source_scores.numel() == 0:
            return target_pairs.new_zeros((target_pairs.size(0),), dtype=torch.float32)
        score_map = {
            (int(pair[0]), int(pair[1])): float(score)
            for pair, score in zip(source_pairs.detach().cpu().tolist(), source_scores.detach().cpu().tolist())
        }
        return target_pairs.new_tensor(
            [score_map.get((int(h), int(t)), 0.0) for h, t in target_pairs.detach().cpu().tolist()],
            dtype=torch.float32,
        )

    def _unique_pairs(self, pair_lists: Iterable[torch.Tensor], device: torch.device) -> torch.Tensor:
        pairs = [p for p in pair_lists if p.numel() > 0]
        if not pairs:
            return torch.zeros((0, 2), dtype=torch.long, device=device)
        return torch.unique(torch.cat(pairs, dim=0).to(device=device, dtype=torch.long), dim=0)

    def _graph_greedy_select(
        self,
        proposal: BoxList,
        ranked_pairs: torch.Tensor,
        *,
        protected_pairs: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del scores
        labels = proposal.get_field("labels").long().detach().cpu().tolist()
        selected: list[tuple[int, int]] = []
        selected_set: set[tuple[int, int]] = set()
        out_degree = [0] * len(proposal)
        in_degree = [0] * len(proposal)
        label_counts: Dict[tuple[int, int], int] = {}

        def try_add(
            pair: tuple[int, int],
            max_out_degree: int,
            max_in_degree: int,
            label_quota: int,
            enforce: bool = True,
        ) -> bool:
            if len(selected) >= self.topk or pair in selected_set or pair[0] == pair[1]:
                return False
            if enforce:
                if self.enforce_degree_cap and (
                    out_degree[pair[0]] >= max_out_degree or in_degree[pair[1]] >= max_in_degree
                ):
                    return False
                label_pair = (int(labels[pair[0]]), int(labels[pair[1]]))
                if self.enforce_label_quota and label_counts.get(label_pair, 0) >= label_quota:
                    return False
            selected.append(pair)
            selected_set.add(pair)
            out_degree[pair[0]] += 1
            in_degree[pair[1]] += 1
            label_pair = (int(labels[pair[0]]), int(labels[pair[1]]))
            label_counts[label_pair] = label_counts.get(label_pair, 0) + 1
            return True

        for pair in protected_pairs.detach().cpu().tolist():
            try_add(
                (int(pair[0]), int(pair[1])),
                self.max_out_degree,
                self.max_in_degree,
                self.label_pair_quota,
            )
            if len(selected) >= self.topk:
                break
        for pair in ranked_pairs.detach().cpu().tolist():
            try_add(
                (int(pair[0]), int(pair[1])),
                self.max_out_degree,
                self.max_in_degree,
                self.label_pair_quota,
            )
            if len(selected) >= self.topk:
                break
        degree_capped = selected[:]
        if len(selected) < self.topk:
            for pair in ranked_pairs.detach().cpu().tolist():
                try_add(
                    (int(pair[0]), int(pair[1])),
                    self.relaxed_max_degree,
                    self.relaxed_max_degree,
                    self.relaxed_label_pair_quota,
                )
                if len(selected) >= self.topk:
                    break
        if len(selected) < self.topk:
            for pair in ranked_pairs.detach().cpu().tolist():
                try_add(
                    (int(pair[0]), int(pair[1])),
                    self.relaxed_max_degree,
                    self.relaxed_max_degree,
                    self.relaxed_label_pair_quota,
                    enforce=False,
                )
                if len(selected) >= self.topk:
                    break

        selected_tensor = proposal.bbox.new_tensor(selected, dtype=torch.long)
        if selected_tensor.numel() == 0:
            selected_tensor = torch.zeros((0, 2), dtype=torch.long, device=proposal.bbox.device)
        degree_tensor = proposal.bbox.new_tensor(degree_capped, dtype=torch.long)
        if degree_tensor.numel() == 0:
            degree_tensor = torch.zeros((0, 2), dtype=torch.long, device=proposal.bbox.device)
        return selected_tensor, degree_tensor
