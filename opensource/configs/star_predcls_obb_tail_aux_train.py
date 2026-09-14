"""6850 -> HPRC two-stage PredCls calibration recipe.

Paper-facing name: Bias-Aware Hard-Predicate Residual Calibration (HPRC).
Historical code/checkpoint tensors keep the ``tail_aux`` name for compatibility.

This file defines the successful model lineage, not a detector-only scratch
experiment. Initialize it with ``pretrained/RPCM_6850_Dual_bgfirst.pth`` via
``--init-model-checkpoint``. The historical result used joint relation-head continuation;
the formal matched entrypoint switches ``HPRC.ADAPTER_ONLY`` on to provide a
clean frozen-base post-training control.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from configs.star_predcls_obb_ablation_dual_rca_train import cfg as _dual_6850_cfg


cfg = copy.deepcopy(_dual_6850_cfg)
rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]

_hard_manifest_path = Path(
    os.environ.get(
        "HPRC_HARD_PREDICATE_MANIFEST",
        "pretrained/HPRC_STAR_hard_predicates.json",
    )
)
_hard_manifest = json.loads(_hard_manifest_path.read_text(encoding="utf-8"))
if _hard_manifest.get("dataset") != "STAR" or _hard_manifest.get("test_statistics_used") is not False:
    raise ValueError("HPRC hard-predicate manifest must be STAR and test-free")
_hard_predicates = [int(row["id"]) for row in _hard_manifest["predicates"]]
if len(_hard_predicates) != len(set(_hard_predicates)) or any(value <= 0 for value in _hard_predicates):
    raise ValueError("HPRC hard-predicate manifest contains invalid IDs")

# Dual-level bias correction:
#   global: weak train-prior logit-adjust auxiliary objective;
#   local:  sample-conditioned bounded residual on hard predicates only.
# Ordinary CE remains the primary prediction objective.
rel_cfg["PREDICATE_LOSS_TYPE"] = "ce"
rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] = float(
    os.environ.get("HPRC_GLOBAL_BIAS_WEIGHT", "0.1")
)
rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_TAU"] = float(
    os.environ.get("HPRC_GLOBAL_BIAS_TAU", "0.5")
)

# The set is not synonymous with frequency tail: it also contains persistent
# same/different-lane, polarity and docking confusions. IDs are STAR predicate
# IDs (background=0). The historical list is frozen for checkpoint/result
# compatibility; future re-selection must use a frozen-6850 TRAIN audit.
rel_cfg["HPRC"] = {
    "ENABLED": True,
    "ADAPTER_ONLY": False,  # historical joint-continuation behavior
    "SELECTION_SOURCE": str(_hard_manifest["selection_source"]),
    "SELECTION_MANIFEST": str(_hard_manifest_path),
    "HARD_PREDICATES": _hard_predicates,
    "HIDDEN_DIM": int(os.environ.get("HPRC_HIDDEN_DIM", "512")),
    "DROPOUT": float(os.environ.get("HPRC_DROPOUT", "0.1")),
    "LOSS_WEIGHT": float(os.environ.get("HPRC_LOCAL_BIAS_WEIGHT", "0.2")),
    "MAX_RESIDUAL_SCALE": float(os.environ.get("HPRC_MAX_RESIDUAL_SCALE", "0.3")),
    "GATE_INIT": 0.0,
    "POS_WEIGHT_CAP": 20.0,
}

# PredCls object logits are fixed GT one-hot values; their CE is constant.
rel_cfg["OBJECT_REFINE_LOSS_WEIGHT"] = 0.0

# Historical 6850 warm-start continuation schedule. Formal new experiments
# must use a star_predcls_hprc_*_historical_6850_train.py entrypoint per AGENTS.md.
cfg["SOLVER"]["BASE_LR"] = float(os.environ.get("BASE_LR", "0.016"))
cfg["SOLVER"]["LR_SCALE_BY_BATCH"] = False
cfg["SOLVER"]["WARMUP_ITERS"] = int(os.environ.get("WARMUP_ITERS", "500"))
cfg["SOLVER"]["MAX_EPOCHS"] = int(os.environ.get("MAX_EPOCHS", "200"))
cfg["SOLVER"]["STEPS"] = [
    int(value) for value in os.environ.get("STEPS", "6000,8500,10000").split(",")
    if value.strip()
]
cfg["SOLVER"]["VAL_START_PERIOD"] = int(os.environ.get("VAL_START_PERIOD", "80"))
cfg["SOLVER"]["VAL_PERIOD"] = int(os.environ.get("VAL_PERIOD", "2"))
cfg["SOLVER"]["CHECKPOINT_PERIOD"] = int(os.environ.get("CHECKPOINT_PERIOD", "0"))
cfg["SOLVER"]["VAL_SPLIT"] = os.environ.get("VAL_SPLIT", "val")
cfg["SOLVER"]["BEST_METRICS"] = ["HR"]
cfg["SOLVER"]["SAVE_LAST"] = False
cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get(
    "OUTPUT_DIR", "outputs/star_predcls_obb_tail_aux_posttrain"
)

cfg["RUNTIME"]["SEED"] = int(os.environ.get("SEED", "1029"))
cfg["RUNTIME"]["DISABLE_CUDNN"] = True
cfg["RUNTIME"]["CUDNN_BENCHMARK"] = False
cfg["RUNTIME"]["CUDNN_DETERMINISTIC"] = True
