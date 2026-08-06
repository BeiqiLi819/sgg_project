from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
import torch

from sgg.config.defaults import get_default_cfg
from sgg.data.rsgp_statistics import (
    build_rsgp_structural_prior,
    write_rsgp_structural_prior,
)
from sgg.data.sample import DatasetMetadata
from sgg.modeling.core.obb_ops import set_boxlist_angle_unit
from sgg.modeling.roi_heads.rsgp import RemoteSensingGraphProposalFilter
from sgg.structures.boxes import BoxList


class _RecordsDataset:
    def __init__(self, *, names, labels, split="train"):
        self.split = split
        self.box_mode = "obb"
        self.metadata = DatasetMetadata(
            dataset_name="synthetic",
            box_mode="obb",
            num_classes=4,
            num_predicates=4,
            categories={index: name for index, name in enumerate(names)},
            predicates={0: "__background__", 1: "r1", 2: "r2", 3: "r3"},
        )
        self.records = [
            {
                "image_index": 7,
                "width": 100,
                "height": 100,
                "boxes": np.asarray(
                    [
                        [25.0, 25.0, 40.0, 30.0, 0.0],
                        [20.0, 20.0, 8.0, 2.0, 0.0],
                        [32.0, 22.0, 7.0, 2.0, 5.0],
                        [75.0, 75.0, 5.0, 5.0, 45.0],
                    ],
                    dtype=np.float32,
                ),
                "labels": np.asarray(labels, dtype=np.int64),
                "relations": np.asarray(
                    [[0, 1, 1], [0, 2, 1], [1, 2, 2], [2, 1, 3]],
                    dtype=np.int64,
                ),
            }
        ]


def _write_prior(tmp_path, dataset, name):
    payload = build_rsgp_structural_prior(dataset, split="train")
    path = tmp_path / name
    write_rsgp_structural_prior(payload, path)
    return path, payload


def _cfg_for_prior(path, names):
    cfg = get_default_cfg()
    cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"] = list(names)
    rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
    rel_cfg["RELATION_NAMES"] = ["__background__", "r1", "r2", "r3"]
    rel_cfg["TEST_FILTER_METHOD"] = "RSGP"
    rel_cfg["RSGP_MODE"] = "RS_ONLY"
    rel_cfg["RSGP_ROLE_MODE"] = "statistical"
    rel_cfg["RSGP_STRUCTURAL_PRIOR_PATH"] = str(path)
    rel_cfg["RSGP_THRESHOLD"] = 0
    rel_cfg["RSGP_TOPK"] = 6
    rel_cfg["RSGP_RS_POOL_TOPK"] = 12
    rel_cfg["RSGP_MAX_OUT_DEGREE"] = 96
    rel_cfg["RSGP_MAX_IN_DEGREE"] = 96
    rel_cfg["RSGP_LABEL_PAIR_QUOTA"] = 800
    return cfg


def _proposal(labels):
    boxes = torch.tensor(
        [
            [25.0, 25.0, 40.0, 30.0, 0.0],
            [20.0, 20.0, 8.0, 2.0, 0.0],
            [32.0, 22.0, 7.0, 2.0, math.radians(5.0)],
            [75.0, 75.0, 5.0, 5.0, math.radians(45.0)],
        ],
        dtype=torch.float32,
    )
    proposal = BoxList(boxes, (100, 100), mode="xywha")
    set_boxlist_angle_unit(proposal, "radian")
    proposal.add_field("labels", torch.tensor(labels, dtype=torch.long))
    return proposal


def _all_pairs(num_entities):
    return torch.tensor(
        [[i, j] for i in range(num_entities) for j in range(num_entities) if i != j],
        dtype=torch.long,
    )


def test_prior_builder_rejects_non_train_split():
    dataset = _RecordsDataset(
        names=["__background__", "a", "b", "c"],
        labels=[1, 2, 2, 3],
        split="val",
    )
    with pytest.raises(ValueError, match="train"):
        build_rsgp_structural_prior(dataset, split="val")


def test_statistical_rsgp_is_invariant_to_category_names(tmp_path):
    labels = [1, 2, 2, 3]
    names_a = ["__background__", "carrier", "vehicle", "network"]
    names_b = ["zero", "random_x", "random_y", "random_z"]
    path_a, prior_a = _write_prior(
        tmp_path,
        _RecordsDataset(names=names_a, labels=labels),
        "prior_a.json",
    )
    path_b, prior_b = _write_prior(
        tmp_path,
        _RecordsDataset(names=names_b, labels=labels),
        "prior_b.json",
    )
    assert prior_a["class_profiles"] == prior_b["class_profiles"]
    assert prior_a["rarity_pair_support"] == prior_b["rarity_pair_support"]

    filter_a = RemoteSensingGraphProposalFilter(_cfg_for_prior(path_a, names_a))
    filter_b = RemoteSensingGraphProposalFilter(_cfg_for_prior(path_b, names_b))
    pairs = _all_pairs(4)
    out_a = filter_a.filter_pairs(_proposal(labels), pairs)
    out_b = filter_b.filter_pairs(_proposal(labels), pairs)
    assert torch.equal(out_a, out_b)


def test_statistical_rsgp_is_equivariant_to_class_id_permutation(tmp_path):
    names = ["__background__", "a", "b", "c"]
    labels = [1, 2, 2, 3]
    permutation = {0: 0, 1: 2, 2: 1, 3: 3}
    permuted_labels = [permutation[label] for label in labels]
    permuted_names = ["__background__", "b", "a", "c"]

    path_a, _ = _write_prior(
        tmp_path,
        _RecordsDataset(names=names, labels=labels),
        "prior_original.json",
    )
    path_b, _ = _write_prior(
        tmp_path,
        _RecordsDataset(names=permuted_names, labels=permuted_labels),
        "prior_permuted.json",
    )
    filter_a = RemoteSensingGraphProposalFilter(_cfg_for_prior(path_a, names))
    filter_b = RemoteSensingGraphProposalFilter(_cfg_for_prior(path_b, permuted_names))
    pairs = _all_pairs(4)
    out_a = filter_a.filter_pairs(_proposal(labels), pairs)
    out_b = filter_b.filter_pairs(_proposal(permuted_labels), pairs)
    assert torch.equal(out_a, out_b)


def test_statistical_prior_hash_tampering_fails(tmp_path):
    names = ["__background__", "a", "b", "c"]
    path, _ = _write_prior(
        tmp_path,
        _RecordsDataset(names=names, labels=[1, 2, 2, 3]),
        "prior.json",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["class_profiles"]["contextual_region"][1] += 0.1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="payload hash mismatch"):
        RemoteSensingGraphProposalFilter(_cfg_for_prior(path, names))


def test_context_carrier_workspace_is_bounded_at_6888_entities(tmp_path):
    names = ["__background__", "a", "b", "c"]
    path, _ = _write_prior(
        tmp_path,
        _RecordsDataset(names=names, labels=[1, 2, 2, 3]),
        "prior.json",
    )
    cfg = _cfg_for_prior(path, names)
    filter_model = RemoteSensingGraphProposalFilter(cfg)
    num_entities = 6888
    centers = torch.arange(num_entities, dtype=torch.float32)
    boxes = torch.stack(
        (
            centers.remainder(256),
            torch.div(centers, 256, rounding_mode="floor"),
            torch.full_like(centers, 4.0),
            torch.full_like(centers, 2.0),
            torch.zeros_like(centers),
        ),
        dim=1,
    )
    proposal = BoxList(boxes, (256, 256), mode="xywha")
    set_boxlist_angle_unit(proposal, "radian")
    proposal.add_field("labels", torch.ones((num_entities,), dtype=torch.long))
    filter_model._prepare_statistical_roles(
        proposal,
        torch.zeros((0, 2), dtype=torch.long),
    )
    carrier_mask = proposal.get_field("rsgp_context_carrier_mask")
    assert int(carrier_mask.sum()) == 128
    assert proposal.get_field("rsgp_context_assignment").shape == (num_entities,)


def test_statistical_role_buffers_are_not_checkpoint_state(tmp_path):
    names = ["__background__", "a", "b", "c"]
    path, _ = _write_prior(
        tmp_path,
        _RecordsDataset(names=names, labels=[1, 2, 2, 3]),
        "prior.json",
    )
    filter_model = RemoteSensingGraphProposalFilter(_cfg_for_prior(path, names))
    state = filter_model.state_dict()
    assert not any("profile" in key or "rarity_pair_support" in key for key in state)


def test_statistical_mode_ignores_legacy_manual_groups(tmp_path):
    names = ["__background__", "a", "b", "c"]
    labels = [1, 2, 2, 3]
    path, _ = _write_prior(
        tmp_path,
        _RecordsDataset(names=names, labels=labels),
        "prior.json",
    )
    cfg_a = _cfg_for_prior(path, names)
    cfg_b = copy.deepcopy(cfg_a)
    rel_cfg = cfg_b["MODEL"]["ROI_RELATION_HEAD"]
    rel_cfg["RSGP_ANCHOR_CLASSES"] = "does_not_exist"
    rel_cfg["RSGP_VEHICLE_CLASSES"] = "another_random_name"
    rel_cfg["RSGP_NETWORK_CLASSES"] = "irrelevant"
    rel_cfg["RSGP_TAIL_PREDICATES"] = [1, 2, 3]

    pairs = _all_pairs(4)
    output_a = RemoteSensingGraphProposalFilter(cfg_a).filter_pairs(
        _proposal(labels),
        pairs,
    )
    output_b = RemoteSensingGraphProposalFilter(cfg_b).filter_pairs(
        _proposal(labels),
        pairs,
    )
    assert torch.equal(output_a, output_b)


def test_graph_selection_keeps_budget_and_strict_capacity_stage(tmp_path):
    names = ["__background__", "a", "b", "c"]
    path, _ = _write_prior(
        tmp_path,
        _RecordsDataset(names=names, labels=[1, 2, 2, 3]),
        "prior.json",
    )
    cfg = _cfg_for_prior(path, names)
    rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
    rel_cfg["RSGP_TOPK"] = 4
    rel_cfg["RSGP_MAX_OUT_DEGREE"] = 1
    rel_cfg["RSGP_MAX_IN_DEGREE"] = 1
    rel_cfg["RSGP_RELAXED_MAX_DEGREE"] = 1
    rel_cfg["RSGP_LABEL_PAIR_QUOTA"] = 1
    rel_cfg["RSGP_RELAXED_LABEL_PAIR_QUOTA"] = 1
    filter_model = RemoteSensingGraphProposalFilter(cfg)
    proposal = _proposal([1, 2, 2, 3])
    ranked = torch.tensor(
        [[0, 1], [0, 2], [1, 2], [2, 1], [3, 0], [1, 3]],
        dtype=torch.long,
    )
    protected = torch.tensor([[3, 0]], dtype=torch.long)
    selected, strict = filter_model._graph_greedy_select(
        proposal,
        ranked,
        protected_pairs=protected,
        scores=torch.arange(ranked.size(0), dtype=torch.float32),
    )

    assert selected.shape == (4, 2)
    assert torch.equal(selected[0], protected[0])
    assert strict.size(0) <= selected.size(0)
    strict_out = torch.bincount(strict[:, 0], minlength=4)
    strict_in = torch.bincount(strict[:, 1], minlength=4)
    assert int(strict_out.max()) <= 1
    assert int(strict_in.max()) <= 1
