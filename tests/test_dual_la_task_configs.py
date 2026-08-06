from configs.star_sgcls_obb_dual_la_train import cfg as sgcls_cfg
from configs.star_sgdet_obb_dual_la_train import cfg as sgdet_cfg
from configs.star_sgdet_obb_dual_la_rpcm_budget_train import (
    cfg as sgdet_rpcm_budget_cfg,
)
from sgg.modeling.detectors.sgdet_detection_cache import (
    compute_sgdet_detection_cache_hash,
)


def _assert_dual_la(cfg, task):
    rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
    assert cfg["MODEL"]["TASK"] == task
    assert rel_cfg["RPCM_RELATION_GRAPH_MODE"] == "dual_view"
    assert rel_cfg["RPCM_REL_SUBJ_VIEW_ENABLED"] is True
    assert rel_cfg["RPCM_REL_OBJ_VIEW_ENABLED"] is True
    assert rel_cfg["PREDICATE_LOSS_TYPE"] == "ce"
    assert rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] == 0.1
    assert rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_TAU"] == 0.5
    assert rel_cfg["RPCM_TAIL_AUX_ENABLED"] is False
    assert rel_cfg["RPCM_TAIL_AUX_PREDICATES"] == []
    assert rel_cfg["OBJECT_REFINE_LOSS_WEIGHT"] == 1.0
    assert rel_cfg["TEST_FILTER_METHOD"] == "PPG"
    assert rel_cfg["RSGP_ROLE_MODE"] == "statistical"
    assert (
        rel_cfg["RSGP_STRUCTURAL_PRIOR_PATH"]
        == "pretrained/rsgp_structural_prior.json"
    )


def test_sgcls_paper_config_uses_dual_la_without_hprc():
    _assert_dual_la(sgcls_cfg, "sgcls")


def test_sgdet_paper_config_uses_dual_la_without_hprc():
    _assert_dual_la(sgdet_cfg, "sgdet")


def test_sgdet_relation_branch_does_not_change_detection_cache_hash():
    from configs.star_sgdet_obb_train import cfg as historical_cfg

    assert compute_sgdet_detection_cache_hash(
        sgdet_cfg
    ) == compute_sgdet_detection_cache_hash(historical_cfg)


def test_sgdet_rpcm_budget_matches_public_optimizer_exposure():
    _assert_dual_la(sgdet_rpcm_budget_cfg, "sgdet")
    assert sgdet_rpcm_budget_cfg["DATALOADER"]["TRAIN_BATCH_SIZE"] == 2
    assert sgdet_rpcm_budget_cfg["SOLVER"]["IMS_PER_BATCH"] == 2
    assert sgdet_rpcm_budget_cfg["SOLVER"]["BASE_LR"] == 1e-3
    assert sgdet_rpcm_budget_cfg["SOLVER"]["LR_SCALE_BY_BATCH"] is True
    assert sgdet_rpcm_budget_cfg["SOLVER"]["WARMUP_ITERS"] == 500
    assert sgdet_rpcm_budget_cfg["SOLVER"]["MAX_ITER"] == 5000
    assert sgdet_rpcm_budget_cfg["SOLVER"]["STEPS"] == [3000, 4000]
    assert sgdet_rpcm_budget_cfg["SOLVER"]["ITERATION_COMPAT"] is True
    assert sgdet_rpcm_budget_cfg["SOLVER"]["VAL_START_PERIOD"] == 1000
    assert sgdet_rpcm_budget_cfg["SOLVER"]["VAL_PERIOD"] == 1000


def test_sgdet_rpcm_budget_reuses_main_detection_cache():
    assert compute_sgdet_detection_cache_hash(
        sgdet_rpcm_budget_cfg
    ) == compute_sgdet_detection_cache_hash(sgdet_cfg)


def test_public_full_configs_expose_the_same_dual_la_models():
    from configs.star_predcls_obb_full import cfg as predcls_full
    from configs.star_sgcls_obb_full import cfg as sgcls_full
    from configs.star_sgdet_obb_full import cfg as sgdet_full

    for cfg, task in (
        (predcls_full, "predcls"),
        (sgcls_full, "sgcls"),
        (sgdet_full, "sgdet"),
    ):
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        assert cfg["MODEL"]["TASK"] == task
        assert rel_cfg["RPCM_RELATION_GRAPH_MODE"] == "dual_view"
        assert rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] == 0.1
        assert rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_TAU"] == 0.5
        # Training defaults to the STAR-compatible PPG protocol. The public
        # test launchers freeze statistical RSGP through explicit overrides.
        assert rel_cfg["TEST_FILTER_METHOD"] == "PPG"
        assert rel_cfg["RSGP_ROLE_MODE"] == "statistical"
        assert rel_cfg["RSGP_PPG_PROTECTED_TOPK"] == 9000

    assert sgdet_full["SOLVER"]["MAX_ITER"] == 5000
    assert compute_sgdet_detection_cache_hash(
        sgdet_full
    ) == compute_sgdet_detection_cache_hash(sgdet_rpcm_budget_cfg)


def test_predcls_rows_share_budget_with_controlled_u_d_dl_chain():
    from configs.star_predcls_obb_ablation_sgg_toolkit_base_train import (
        cfg as base,
    )
    from configs.star_predcls_obb_ablation_dual_la_train import cfg as dual_la
    from configs.star_predcls_obb_ablation_dual_rca_train import cfg as dual
    from configs.star_predcls_obb_ablation_unified_rca_train import cfg as unified
    from configs.star_predcls_obb_ablation_role_aware_train import cfg as role_aware

    solver_keys = (
        "BASE_LR",
        "WARMUP_ITERS",
        "MAX_EPOCHS",
        "STEPS",
        "VAL_PERIOD",
        "CHECKPOINT_PERIOD",
        "VAL_SPLIT",
        "EARLY_STOP_ENABLED",
        "EARLY_STOP_METRIC",
        "EARLY_STOP_PATIENCE",
        "EARLY_STOP_MIN_DELTA",
        "EARLY_STOP_START_PERIOD",
    )
    for key in solver_keys:
        assert (
            base["SOLVER"][key]
            == unified["SOLVER"][key]
            == dual["SOLVER"][key]
            == role_aware["SOLVER"][key]
            == dual_la["SOLVER"][key]
        )
    assert unified["SOLVER"]["VAL_SPLIT"] == "val"
    assert unified["SOLVER"]["EARLY_STOP_ENABLED"] is True
    assert unified["SOLVER"]["EARLY_STOP_METRIC"] == "HR"
    assert unified["SOLVER"]["EARLY_STOP_PATIENCE"] == 10
    assert unified["SOLVER"]["EARLY_STOP_MIN_DELTA"] == 0.0
    # The complete source predictor has a different convergence curve.  Its
    # first run is inspected from epoch 70, while U/D/DL keep the previously
    # established epoch-120 validation schedule.
    assert base["SOLVER"]["VAL_START_PERIOD"] == 70
    assert (
        unified["SOLVER"]["VAL_START_PERIOD"]
        == dual["SOLVER"]["VAL_START_PERIOD"]
        == role_aware["SOLVER"]["VAL_START_PERIOD"]
        == dual_la["SOLVER"]["VAL_START_PERIOD"]
        == 120
    )
    # Early validation records and saves best checkpoints. Patience starts
    # after the second LR milestone (step 8000, about epoch 164), so early
    # stopping cannot prevent either scheduled refinement phase.
    assert (
        base["SOLVER"]["EARLY_STOP_START_PERIOD"]
        == unified["SOLVER"]["EARLY_STOP_START_PERIOD"]
        == dual["SOLVER"]["EARLY_STOP_START_PERIOD"]
        == role_aware["SOLVER"]["EARLY_STOP_START_PERIOD"]
        == dual_la["SOLVER"]["EARLY_STOP_START_PERIOD"]
        == 170
    )
    assert (
        base["MODEL"]["PRETRAINED_DETECTOR"]
        == unified["MODEL"]["PRETRAINED_DETECTOR"]
        == dual["MODEL"]["PRETRAINED_DETECTOR"]
        == role_aware["MODEL"]["PRETRAINED_DETECTOR"]
        == dual_la["MODEL"]["PRETRAINED_DETECTOR"]
        == "pretrained/OBB_swin_L_OBD.pth"
    )

    base_rel = base["MODEL"]["ROI_RELATION_HEAD"]
    unified_rel = unified["MODEL"]["ROI_RELATION_HEAD"]
    dual_rel = dual["MODEL"]["ROI_RELATION_HEAD"]
    role_rel = role_aware["MODEL"]["ROI_RELATION_HEAD"]
    dual_la_rel = dual_la["MODEL"]["ROI_RELATION_HEAD"]
    assert base_rel["PREDICTOR"] == "RPCM_SGG_TOOLKIT_ORIGINAL"
    assert unified_rel["PREDICTOR"] == dual_rel["PREDICTOR"] == dual_la_rel["PREDICTOR"]
    assert base_rel["RPCM_RELATION_GRAPH_MODE"] == "sgg_toolkit"
    assert unified_rel["RPCM_RELATION_GRAPH_MODE"] == "unified"
    assert dual_rel["RPCM_RELATION_GRAPH_MODE"] == "dual_view"
    assert role_rel["RPCM_RELATION_GRAPH_MODE"] == "role_aware"
    assert role_rel["RPCM_ROLE_AWARE_ADAPTER_RANK"] == 32
    assert dual_la_rel["RPCM_RELATION_GRAPH_MODE"] == "dual_view"
    assert base_rel["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] == 0.0
    assert unified_rel["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] == 0.0
    assert dual_rel["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] == 0.0
    assert role_rel["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] == 0.0
    assert dual_la_rel["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] == 0.1


def test_dual_la_original_head_is_isolated_decision_experiment():
    from configs.star_predcls_obb_ablation_dual_la_original_head_train import (
        cfg,
    )

    rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
    assert rel_cfg["PREDICTOR"] == "RPCM_SGG_TOOLKIT_ORIGINAL"
    assert rel_cfg["RPCM_RELATION_GRAPH_MODE"] == "dual_view"
    assert rel_cfg["RPCM_REL_SUBJ_VIEW_ENABLED"] is True
    assert rel_cfg["RPCM_REL_OBJ_VIEW_ENABLED"] is True
    assert rel_cfg["RPCM_GLOVE_INIT_MODE"] == "rpcm"
    assert rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] == 0.1
    assert cfg["EXP_nums"] == 30
    assert cfg["SOLVER"]["STEPS"] == [5500, 8000]
    assert cfg["SOLVER"]["VAL_START_PERIOD"] == 70
    assert cfg["SOLVER"]["EARLY_STOP_START_PERIOD"] == 170
    assert cfg["SOLVER"]["OUTPUT_DIR"].endswith("dual_la_original_head")


def test_6850_reproduction_restores_historical_protocol():
    from configs.star_predcls_obb_ablation_dual_6850_repro_train import cfg

    rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
    solver = cfg["SOLVER"]
    runtime = cfg["RUNTIME"]

    assert rel_cfg["RPCM_RELATION_GRAPH_MODE"] == "dual_view"
    assert rel_cfg["RPCM_REL_SUBJ_VIEW_ENABLED"] is True
    assert rel_cfg["RPCM_REL_OBJ_VIEW_ENABLED"] is True
    assert rel_cfg["RPCM_LEGACY_6850_EXACT"] is True
    assert rel_cfg["RPCM_LEGACY_NUM_PROTO"] == 1
    assert rel_cfg["RPCM_LEGACY_USE_VIS_PROTO"] is False
    assert rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] == 0.0
    assert rel_cfg["RPCM_TAIL_AUX_ENABLED"] is False

    assert cfg["DATALOADER"]["TRAIN_BATCH_SIZE"] == 16
    assert solver["IMS_PER_BATCH"] == 16
    assert solver["BASE_LR"] == 0.001
    assert solver["LR_SCALE_BY_BATCH"] is True
    assert solver["WARMUP_ITERS"] == 500
    assert solver["STEPS"] == [13000, 18000]
    assert solver["MAX_ITER"] == 20000
    assert solver["ITERATION_COMPAT"] is True
    assert solver["VALIDATE_ON_ITER"] is True
    assert solver["VAL_START_PERIOD"] == 14000
    assert solver["VAL_PERIOD"] == 200
    assert solver["EARLY_STOP_ENABLED"] is False
    assert runtime["SEED"] == 1029
    assert runtime["CUDNN_BENCHMARK"] is False
    assert runtime["CUDNN_DETERMINISTIC"] is True
