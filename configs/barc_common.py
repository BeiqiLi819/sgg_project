"""Shared public configuration for the BARC-SGG method."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path


def apply_barc(cfg: dict, *, task: str) -> dict:
    """Enable BAP + Dual Role-aware RPCM + BAC for one STAR task."""

    task = str(task).lower()
    if task not in {"predcls", "sgcls", "sgdet"}:
        raise ValueError(f"Unsupported BARC task: {task}")

    cfg = copy.deepcopy(cfg)
    cfg["MODEL"]["TASK"] = task
    model = cfg["MODEL"]
    rel = model["ROI_RELATION_HEAD"]
    manifest_path = Path(
        os.environ.get(
            "BARC_HARD_PREDICATE_MANIFEST",
            "pretrained/BARC_hard_predicates_top20.json",
        )
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "barc-rule-generated-hard-predicates-v1"
        or manifest.get("dataset") != "STAR"
        or manifest.get("selection_rule") != "B_recall"
        or int(manifest.get("hard_count", -1)) != 20
        or manifest.get("test_statistics_used") is not False
    ):
        raise ValueError("BARC hard-predicate manifest must be STAR, TEST-free Top20")

    rel["PREDICTOR"] = "RPCM_ORIGINAL_LEGACY"
    rel["RPCM_RELATION_GRAPH_MODE"] = "dual_view"
    rel["RPCM_REL_SUBJ_VIEW_ENABLED"] = True
    rel["RPCM_REL_OBJ_VIEW_ENABLED"] = True

    # BAP: ordinary global Top-10k selection from the released PPG score plus
    # a whole-image, target-free residual. No hard group quota is applied.
    rel["TEST_FILTER_METHOD"] = "Q5"
    rel["Q5_TOPK"] = int(os.environ.get("BARC_PAIR_BUDGET", "10000"))
    rel["Q5_PAIR_THRESHOLD"] = rel["Q5_TOPK"]
    rel["Q5_CHECKPOINT"] = os.environ.get(
        "BARC_BAP_CHECKPOINT", "pretrained/BARC_BAP.pth"
    )

    # BAC: all-class adaptation with a frozen-validation Bottom-20 hard
    # predicate auxiliary branch. The released combined checkpoint supplies
    # the trained adapter weights; the config also supports training from a
    # compatible Dual RPCM base checkpoint.
    hard_ids = [int(row["id"]) for row in manifest["predicates"]]
    rel["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] = 0.1
    rel["PREDICATE_AUX_LOGIT_ADJUST_TAU"] = 0.5
    rel["HPRC"] = {
        "ENABLED": True,
        "ADAPTER_ONLY": False,
        "VARIANT": "HU",
        "SELECTION_SOURCE": "STAR validation-only smoothed Recall@2000",
        "SELECTION_MANIFEST": str(manifest_path),
        "SELECTION_RULE": str(manifest["selection_rule"]),
        "HARD_PREDICATES": hard_ids,
        "HIDDEN_DIM": 512,
        "DROPOUT": 0.1,
        "LOSS_WEIGHT": 0.2,
        "MAX_RESIDUAL_SCALE": 0.3,
        "GATE_PARAMETERIZATION": "positive_sigmoid",
        "GATE_INIT": 0.01,
        "ZERO_INIT_OUTPUT": True,
        "POS_WEIGHT_CAP": 20.0,
        "UNIVERSAL": {
            "ENABLED": True,
            "INCLUDE_BACKGROUND": True,
            "HIDDEN_DIM": 512,
            "DROPOUT": 0.1,
            "MAX_RESIDUAL_SCALE": 0.3,
            "GATE_PARAMETERIZATION": "positive_sigmoid",
            "GATE_INIT": 0.01,
            "ZERO_INIT_OUTPUT": True,
            "DETACH_ANCHOR_LOGITS": True,
            "SUPERVISION": "balanced_all_class_fused_ce",
        },
    }

    cfg["BARC_SGG"] = {
        "schema": "barc-sgg-public-v1",
        "components": [
            "whole-image budget-aware proposal (BAP)",
            "dual role-aware RPCM reasoning",
            "base-anchored predicate calibration (BAC)",
        ],
        "pair_budget": int(rel["Q5_TOPK"]),
        "hard_predicate_count": len(hard_ids),
        "test_free_hard_selection": True,
        "task": task,
    }
    return cfg
