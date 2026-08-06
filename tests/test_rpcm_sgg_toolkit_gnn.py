import torch
from torch import nn

from sgg.modeling.roi_heads.roi_relation_predictors import (
    RPCMLegacy,
    RPCMSGGToolkitOriginal,
    _RoleAwareLowRankAdapter,
    _SGGToolkitGraphConvolutionLayerCollect,
    _SGGToolkitGraphConvolutionLayerUpdate,
    _build_sgg_toolkit_object_glove_init,
    _build_sgg_toolkit_relation_glove_init,
    _endpoint_group_message,
    _matrix_squared_euclidean_distance,
    _role_aware_relation_message,
)
from sgg.structures.boxes import BoxList


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


def test_sgg_toolkit_update_is_target_plus_source():
    update = _SGGToolkitGraphConvolutionLayerUpdate()
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    source = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
    torch.testing.assert_close(update(target, source, 0), target + source)
    torch.testing.assert_close(update(target, source, 1), target + source)


def test_matrix_distance_matches_broadcast_for_regular_inputs_and_gradients():
    torch.manual_seed(13)
    left = torch.randn(17, 11, requires_grad=True)
    right = torch.randn(7, 11, requires_grad=True)
    matrix_left = left.detach().clone().requires_grad_()
    matrix_right = right.detach().clone().requires_grad_()

    reference = (
        left.unsqueeze(1) - right.unsqueeze(0)
    ).norm(dim=2).pow(2)
    bounded = _matrix_squared_euclidean_distance(
        matrix_left,
        matrix_right,
    )

    torch.testing.assert_close(bounded, reference, rtol=2e-6, atol=2e-6)
    assert torch.all(bounded >= 0)
    upstream = torch.randn_like(reference)
    (reference * upstream).sum().backward()
    (bounded * upstream).sum().backward()

    torch.testing.assert_close(
        matrix_left.grad,
        left.grad,
        rtol=2e-6,
        atol=2e-5,
    )
    torch.testing.assert_close(
        matrix_right.grad,
        right.grad,
        rtol=2e-6,
        atol=2e-5,
    )


def test_matrix_distance_clamps_roundoff_below_zero():
    torch.manual_seed(29)
    left = torch.randn(32, 4096)
    right = left + 1e-4 * torch.randn_like(left)
    distances = _matrix_squared_euclidean_distance(left, right)
    assert distances.shape == (32, 32)
    assert torch.isfinite(distances).all()
    assert torch.all(distances >= 0)


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


def test_role_aware_views_restore_both_cross_role_chains_without_dense_maps():
    # r0=(0,1), r1=(0,2) share subject; r2=(3,1) shares object
    # with r0; r3=(1,4) continues r0; r4=(5,0) precedes r0.
    subject = torch.tensor([0, 0, 3, 1, 5])
    obj = torch.tensor([1, 2, 1, 4, 0])
    features = torch.arange(1, 6, dtype=torch.float32).unsqueeze(1)

    ss, ss_valid = _endpoint_group_message(features, subject, subject, 6)
    oo, oo_valid = _endpoint_group_message(features, obj, obj, 6)
    forward_chain, forward_valid = _endpoint_group_message(
        features, subject, obj, 6
    )
    reverse_chain, reverse_valid = _endpoint_group_message(
        features, obj, subject, 6
    )

    # Relation r0 receives r1 through SS, r2 through OO, r3 through the
    # forward chain and r4 through the reverse chain.
    torch.testing.assert_close(ss[0], features[1])
    torch.testing.assert_close(oo[0], features[2])
    torch.testing.assert_close(forward_chain[0], features[3])
    torch.testing.assert_close(reverse_chain[0], features[4])
    assert ss_valid[0] and oo_valid[0] and forward_valid[0] and reverse_valid[0]


def test_role_aware_message_has_finite_gradients_and_bounded_positive_weights():
    subject = torch.tensor([0, 0, 3, 1, 5])
    obj = torch.tensor([1, 2, 1, 4, 0])
    features = torch.randn(5, 3, requires_grad=True)
    logits = torch.zeros(4, requires_grad=True)

    message, weights = _role_aware_relation_message(
        features, subject, obj, 6, logits, max_log_weight=1.0
    )
    assert message.shape == features.shape
    torch.testing.assert_close(weights, torch.ones_like(weights))
    message.square().sum().backward()
    assert torch.isfinite(features.grad).all()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_role_aware_low_rank_adapters_start_as_exact_zero_deltas():
    torch.manual_seed(123)
    rng_before = torch.random.get_rng_state().clone()
    adapters = nn.ModuleList(
        [_RoleAwareLowRankAdapter(8, 3, seed=200 + idx) for idx in range(4)]
    )
    # Adapter initialization uses a private generator, so adding the MV branch
    # does not shift initialization of any modules constructed afterwards.
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before)

    features = torch.randn(5, 8, requires_grad=True)
    for adapter in adapters:
        torch.testing.assert_close(adapter(features), torch.zeros_like(features))

    subject = torch.tensor([0, 0, 3, 1, 5])
    obj = torch.tensor([1, 2, 1, 4, 0])
    message, _ = _role_aware_relation_message(
        features,
        subject,
        obj,
        6,
        torch.zeros(4),
        max_log_weight=1.0,
        source_transforms=adapters,
    )
    torch.testing.assert_close(message, torch.zeros_like(features))
    message.sum().backward()
    # Standard LoRA behavior: zero up-projections learn immediately; down
    # projections receive gradients after the up path becomes non-zero.
    assert all(adapter.up_weight.grad is not None for adapter in adapters)
    assert sum(adapter.up_weight.grad.abs().sum() for adapter in adapters) > 0


def test_role_aware_mapping_uses_the_exact_unified_candidate_support():
    proposals = [_Proposal(6)]
    pairs = [
        torch.tensor(
            [[0, 1], [0, 2], [3, 1], [1, 4], [5, 0]],
            dtype=torch.long,
        )
    ]
    unified = _mapping_only_model()
    unified.relation_graph_mode = "unified"
    role_aware = _mapping_only_model()
    role_aware.relation_graph_mode = "role_aware"

    unified_maps = unified._get_map_idxs(proposals, pairs)
    role_maps = role_aware._get_map_idxs(proposals, pairs)
    for unified_map, role_map in zip(unified_maps, role_maps):
        torch.testing.assert_close(unified_map, role_map)


def test_controlled_unified_row_only_changes_relation_adjacency():
    from configs.star_predcls_obb_ablation_unified_rca_train import cfg

    assert cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_RELATION_GRAPH_MODE"] == "unified"


def test_controlled_role_aware_row_is_unified_equivalent_at_initialization():
    from configs.star_predcls_obb_ablation_role_aware_train import cfg

    rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
    assert rel_cfg["RPCM_RELATION_GRAPH_MODE"] == "role_aware"
    assert rel_cfg["RPCM_ROLE_AWARE_MAX_LOG_WEIGHT"] == 1.0
    assert rel_cfg["RPCM_ROLE_AWARE_RESIDUAL_MAX_WEIGHT"] == 0.25
    assert rel_cfg["RPCM_ROLE_AWARE_ADAPTER_RANK"] == 32
    assert rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] == 0.0


def test_controlled_base_selects_complete_sgg_toolkit_rpcm():
    from configs.star_predcls_obb_ablation_sgg_toolkit_base_train import cfg

    assert (
        cfg["MODEL"]["ROI_RELATION_HEAD"]["PREDICTOR"]
        == "RPCM_SGG_TOOLKIT_ORIGINAL"
    )
    assert cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_RELATION_GRAPH_MODE"] == "sgg_toolkit"
    assert cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_GLOVE_INIT_MODE"] == "sgg_toolkit"
    assert cfg["EXP_nums"] == 30
    assert cfg["SOLVER"]["STEPS"] == [5500, 8000]


def test_rpcm_audit_selects_complete_sgg_toolkit_rpcm():
    from configs.star_predcls_obb_rpcm_audit_train import cfg

    assert (
        cfg["MODEL"]["ROI_RELATION_HEAD"]["PREDICTOR"]
        == "RPCM_SGG_TOOLKIT_ORIGINAL"
    )
    assert cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_RELATION_GRAPH_MODE"] == "sgg_toolkit"


def test_sgg_toolkit_glove_initializers_preserve_source_token_rules(monkeypatch):
    word_to_idx = {
        "car": 0,
        "bridge": 1,
        "same": 2,
        "lane": 3,
        "with": 4,
    }
    vectors = torch.arange(15, dtype=torch.float32).view(5, 3)
    monkeypatch.setattr(
        "sgg.modeling.roi_heads.roi_relation_predictors._load_rpcm_glove_table",
        lambda *_: (word_to_idx, vectors),
    )
    torch.manual_seed(7)
    obj_init, obj_diag = _build_sgg_toolkit_object_glove_init(
        ["__background__", "car", "boarding_bridge"], "", 3
    )
    torch.testing.assert_close(obj_init[1], vectors[0])
    # SGG-ToolKit splits only on spaces, so the underscore name is not
    # silently converted to the available "bridge" token.
    assert "boarding_bridge" in obj_diag["missing_classes"]
    assert not torch.equal(obj_init[2], vectors[1])

    rel_init, rel_diag = _build_sgg_toolkit_relation_glove_init(
        ["__background__", "same lane with"], "", 3
    )
    torch.testing.assert_close(
        rel_init[1], torch.stack([vectors[2], vectors[3], vectors[4]]).mean(0)
    )
    assert rel_diag["missing_predicates"] == []


def _small_original_cfg(graph_mode="sgg_toolkit"):
    cfg = {
        "EXP_nums": 3,
        "MODEL": {
            "TASK": "predcls",
            "ROI_BOX_HEAD": {
                "NUM_CLASSES": 4,
                "CLASS_NAMES": [
                    "__background__",
                    "car",
                    "truck",
                    "road",
                ],
            },
            "ROI_RELATION_HEAD": {
                "PREDICTOR": "RPCM_SGG_TOOLKIT_ORIGINAL",
                "NUM_CLASSES": 5,
                "RELATION_NAMES": [
                    "__background__",
                    "near",
                    "same lane with",
                    "opposite",
                    "through",
                ],
                "USE_GT_BOX": True,
                "USE_GT_OBJECT_LABEL": True,
                "EMBED_DIM": 3,
                "CONTEXT_HIDDEN_DIM": 4,
                "CONTEXT_POOLING_DIM": 8,
                "EDGE_FEATURES_REPRESENTATION": "fusion",
                "WORD_EMBEDDING_FEATURES": True,
                "SEMANTIC_GLOVE_PATH": "",
                "RPCM_PROTO_GLOVE_PATH": "",
                "RPCM_PROTO_EMBED_DIM": 3,
                "RPCM_GLOVE_INIT_MODE": "sgg_toolkit",
                "RPCM_MLP_DIM": 4,
                "RPCM_FEAT_UPDATE_STEP": 1,
                "RPCM_DROPOUT": 0.0,
                "RPCM_RELATION_GRAPH_MODE": graph_mode,
                "RPCM_REL_SUBJ_VIEW_ENABLED": True,
                "RPCM_REL_OBJ_VIEW_ENABLED": True,
                "RPCM_LEGACY_6850_EXACT": True,
                "CAUSAL": {"SPATIAL_FOR_VISION": True},
            },
        },
    }
    return cfg


def test_complete_original_rpcm_forward_losses_and_state_layout(monkeypatch):
    def deterministic_obj(names, _path, dim):
        values = torch.arange(len(names) * dim, dtype=torch.float32)
        return values.view(len(names), dim) / 10.0, {"missing_classes": []}

    def deterministic_rel(names, _path, dim):
        values = torch.arange(len(names) * dim, dtype=torch.float32)
        return values.view(len(names), dim) / 20.0, {"missing_predicates": []}

    monkeypatch.setattr(
        "sgg.modeling.roi_heads.roi_relation_predictors._build_sgg_toolkit_object_glove_init",
        deterministic_obj,
    )
    monkeypatch.setattr(
        "sgg.modeling.roi_heads.roi_relation_predictors._build_sgg_toolkit_relation_glove_init",
        deterministic_rel,
    )

    model = RPCMSGGToolkitOriginal(_small_original_cfg(), in_channels=8)
    model.train()
    proposal = BoxList(
        torch.tensor(
            [
                [20.0, 20.0, 10.0, 6.0, 0.0],
                [40.0, 22.0, 9.0, 5.0, 0.1],
                [60.0, 25.0, 12.0, 4.0, -0.2],
            ]
        ),
        (100, 100),
        mode="xywha",
    )
    proposal.add_field("labels", torch.tensor([1, 2, 3]))
    pairs = [torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2]])]
    labels = [torch.tensor([1, 2, 3, 4])]
    roi_features = torch.randn(3, 8, requires_grad=True)
    union_features = torch.randn(4, 8, requires_grad=True)

    rel_logits, refine_logits, losses = model(
        [proposal],
        pairs,
        labels,
        None,
        roi_features,
        union_features,
    )

    assert rel_logits[0].shape == (4, 5)
    assert refine_logits[0].shape == (3, 4)
    assert set(losses) == {
        "l21_1_loss",
        "l21_2_loss",
        "dist_loss2_1",
        "dist_loss2_2",
        "loss_dis",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    (rel_logits[0].sum() + sum(losses.values())).backward()
    assert model.W_pred.layers[0].weight.grad is not None
    assert model.gate_pred.weight.grad is not None

    state = model.state_dict()
    assert "W_pred.layers.0.weight" in state
    assert "project_head.layers.0.weight" in state
    assert "obj_embed.weight" in state
    assert "rel_embed.weight" in state
    assert not any(key.startswith("rel_proto.") for key in state)


def test_original_head_accepts_current_dual_view_graph(monkeypatch):
    def deterministic(names, _path, dim):
        values = torch.arange(len(names) * dim, dtype=torch.float32)
        return values.view(len(names), dim) / 10.0, {
            "missing_classes": [],
            "missing_predicates": [],
        }

    monkeypatch.setattr(
        "sgg.modeling.roi_heads.roi_relation_predictors._build_sgg_toolkit_object_glove_init",
        deterministic,
    )
    monkeypatch.setattr(
        "sgg.modeling.roi_heads.roi_relation_predictors._build_sgg_toolkit_relation_glove_init",
        deterministic,
    )
    model = RPCMSGGToolkitOriginal(
        _small_original_cfg(graph_mode="dual_view"), in_channels=8
    )
    model.eval()
    assert model.relation_graph_mode == "dual_view"
    assert hasattr(model, "gcn_ent2ent")
    assert not hasattr(model, "gcn_collect_feat")

    proposal = BoxList(
        torch.tensor(
            [
                [20.0, 20.0, 10.0, 6.0, 0.0],
                [40.0, 22.0, 9.0, 5.0, 0.1],
                [60.0, 25.0, 12.0, 4.0, -0.2],
            ]
        ),
        (100, 100),
        mode="xywha",
    )
    proposal.add_field("labels", torch.tensor([1, 2, 3]))
    pairs = [torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2]])]
    with torch.no_grad():
        rel_logits, refine_logits, losses = model(
            [proposal],
            pairs,
            None,
            None,
            torch.randn(3, 8),
            torch.randn(4, 8),
        )
    assert rel_logits[0].shape == (4, 5)
    assert refine_logits[0].shape == (3, 4)
    assert losses == {}
