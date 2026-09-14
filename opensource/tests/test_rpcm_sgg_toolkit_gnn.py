import torch
from torch import nn

import sgg.modeling.roi_heads.rpcm_sgg_toolkit_original as original_module
from sgg.config.defaults import get_default_cfg
from sgg.modeling.roi_heads.roi_relation_predictors import (
    RPCMLegacy,
    _LegacyGCNLayer,
    _LegacyGraphConvolutionLayerCollect,
    _SGGToolkitGraphConvolutionLayerCollect,
    _SGGToolkitGraphConvolutionLayerUpdate,
)
from sgg.modeling.roi_heads.rpcm_sgg_toolkit_original import (
    RPCMSGGToolkitOriginal,
)


class _Proposal:
    def __init__(self, count: int):
        self.bbox = torch.zeros((count, 5), dtype=torch.float32)

    def __len__(self):
        return self.bbox.size(0)


def _mapping_only_model():
    model = RPCMLegacy.__new__(RPCMLegacy)
    nn.Module.__init__(model)
    model.relation_graph_mode = "sgg_toolkit"
    return model


def test_sgg_toolkit_collect_matches_original_degree_average():
    collect = _SGGToolkitGraphConvolutionLayerCollect(2, 2)
    for unit in collect.collect_units:
        with torch.no_grad():
            unit.fc.weight.copy_(torch.eye(2))
            unit.fc.bias.zero_()

    source = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    attention = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    expected = torch.tensor([[2.0, 3.0], [4.0, 5.0]])

    for unit_id in range(6):
        actual = collect(torch.zeros_like(expected), source, attention, unit_id)
        torch.testing.assert_close(actual, expected)


def test_semantic_multiplicity_alpha_one_is_exact_original_identity():
    collect = _SGGToolkitGraphConvolutionLayerCollect(2, 2)
    with torch.no_grad():
        collect.collect_units[0].fc.weight.copy_(torch.eye(2))
        collect.collect_units[0].fc.bias.zero_()
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    attention = torch.ones((1, 3), dtype=torch.float32)
    group_ids = torch.tensor([4, 4, 7])

    original = collect(torch.zeros((1, 2)), source, attention, 0)
    identity = collect(
        torch.zeros((1, 2)), source, attention, 0, group_ids, 1.0
    )
    assert torch.equal(identity, original)


def test_semantic_multiplicity_tempering_matches_brute_group_means():
    collect = _SGGToolkitGraphConvolutionLayerCollect(2, 2)
    with torch.no_grad():
        collect.collect_units[1].fc.weight.copy_(torch.eye(2))
        collect.collect_units[1].fc.bias.zero_()
    source = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    attention = torch.ones((1, 3), dtype=torch.float32)
    group_ids = torch.tensor([5, 5, 6])
    target = torch.zeros((1, 2))

    soft = collect(target, source, attention, 1, group_ids, 0.5)
    root_two = 2.0**0.5
    expected_soft = torch.tensor(
        [[2.0 * root_two / (root_two + 1.0), 4.0 / (root_two + 1.0)]]
    )
    torch.testing.assert_close(soft, expected_soft)

    equal = collect(target, source, attention, 1, group_ids, 0.0)
    torch.testing.assert_close(equal, torch.tensor([[1.0, 2.0]]))


def test_sgg_toolkit_update_is_target_plus_source():
    update = _SGGToolkitGraphConvolutionLayerUpdate()
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    source = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
    torch.testing.assert_close(update(target, source, 0), target + source)
    torch.testing.assert_close(update(target, source, 1), target + source)


def test_original_alternative_graph_backend_dependencies_are_imported():
    """Unified/dual-view construction must not fail with a late NameError."""

    assert original_module._LegacyGCNLayer is _LegacyGCNLayer
    assert (
        original_module._LegacyGraphConvolutionLayerCollect
        is _LegacyGraphConvolutionLayerCollect
    )


def test_original_alternative_graph_backends_construct():
    cfg = get_default_cfg()
    box_cfg = cfg["MODEL"]["ROI_BOX_HEAD"]
    rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
    box_cfg["NUM_CLASSES"] = 3
    box_cfg["CLASS_NAMES"] = ["__background__", "a", "b"]
    rel_cfg["NUM_CLASSES"] = 4
    rel_cfg["RELATION_NAMES"] = ["__background__", "r1", "r2", "r3"]
    rel_cfg["USE_GT_BOX"] = True
    rel_cfg["USE_GT_OBJECT_LABEL"] = True
    rel_cfg["CONTEXT_HIDDEN_DIM"] = 2
    rel_cfg["CONTEXT_POOLING_DIM"] = 8
    rel_cfg["EMBED_DIM"] = 3
    rel_cfg["RPCM_MLP_DIM"] = 4
    rel_cfg["RPCM_PROTO_EMBED_DIM"] = 3
    rel_cfg["RPCM_PROTO_GLOVE_PATH"] = ""
    rel_cfg["SEMANTIC_GLOVE_PATH"] = ""
    rel_cfg["RPCM_FEAT_UPDATE_STEP"] = 1
    cfg["EXP_nums"] = 3

    for mode in ("unified", "dual_view"):
        rel_cfg["RPCM_RELATION_GRAPH_MODE"] = mode
        model = RPCMSGGToolkitOriginal(cfg, in_channels=8)
        assert len(model.gcn_ent2ent) == 1
        assert len(model.gcn_ent2rel) == 1
        assert len(model.gcn_rel2rel) == 1


def test_sgg_toolkit_maps_match_original_block_and_endpoint_logic():
    model = _mapping_only_model()
    proposals = [_Proposal(3), _Proposal(2)]
    pairs = [
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        torch.tensor([[0, 1]], dtype=torch.long),
    ]

    subj, obj, rel_rel, unused, obj_obj = model._get_map_idxs(proposals, pairs)

    expected_obj_obj = torch.tensor(
        [
            [0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ],
        dtype=torch.float32,
    )
    expected_rel_rel = torch.tensor(
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        dtype=torch.float32,
    )

    assert subj.shape == (5, 3)
    assert obj.shape == (5, 3)
    assert unused.shape == (0, 0)
    torch.testing.assert_close(obj_obj, expected_obj_obj)
    torch.testing.assert_close(rel_rel, expected_rel_rel)


def test_ablation_a_selects_full_sgg_toolkit_gnn():
    from configs.star_predcls_obb_ablation_unified_rca_train import cfg

    assert cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_RELATION_GRAPH_MODE"] == "sgg_toolkit"


def test_original_four_role_union_matches_endpoint_unified_graph():
    subj = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )
    obj = torch.tensor(
        [
            [0, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 0],
        ],
        dtype=torch.float32,
    )
    role = RPCMSGGToolkitOriginal._role_relation_adjacencies(subj, obj)
    endpoint = (subj + obj).clamp(max=1.0)
    expected = (endpoint.t() @ endpoint) > 0
    expected.fill_diagonal_(False)

    torch.testing.assert_close(role["unified"], expected.float())
    torch.testing.assert_close(role["so"], role["os"].t())
    for adjacency in role.values():
        assert not torch.diagonal(adjacency).bool().any()


def test_original_entity_topology_intervention_is_parameter_free_and_exact():
    model = RPCMSGGToolkitOriginal.__new__(RPCMSGGToolkitOriginal)
    nn.Module.__init__(model)
    original = torch.tensor(
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.float32
    )
    subj = torch.tensor(
        [[1, 0], [0, 1], [0, 0]], dtype=torch.float32
    )
    obj = torch.tensor(
        [[0, 0], [1, 0], [0, 1]], dtype=torch.float32
    )

    model.entity_adjacency = "complete"
    assert model._diagnostic_entity_adjacency(original, subj, obj) is original

    model.entity_adjacency = "relation_induced"
    expected = torch.tensor(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32
    )
    torch.testing.assert_close(
        model._diagnostic_entity_adjacency(original, subj, obj), expected
    )

    model.entity_adjacency = "none"
    torch.testing.assert_close(
        model._diagnostic_entity_adjacency(original, subj, obj),
        torch.zeros_like(original),
    )


def test_original_context_path_masks_preserve_declared_sources():
    model = RPCMSGGToolkitOriginal.__new__(RPCMSGGToolkitOriginal)
    nn.Module.__init__(model)
    expected = {
        "full": {0, 1, 2, 3, 4, 5},
        "no_rel_rel": {0, 1, 2, 3, 4},
        "no_rel_ent": {2, 3, 4, 5},
        "no_ent_rel": {0, 1, 4, 5},
        "no_ent_ent": {0, 1, 2, 3, 5},
        "rel_rel_only": {5},
    }
    for variant, enabled in expected.items():
        model.context_ablation = variant
        assert {unit for unit in range(6) if model._path_enabled(unit)} == enabled
