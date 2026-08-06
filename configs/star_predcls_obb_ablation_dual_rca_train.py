"""Controlled PredCls row D: dual-view relation-context adjacency from scratch.

Only the frozen OBB detector is initialized from
``pretrained/OBB_swin_L_OBD.pth``. The relation predictor is not initialized
from an RPCM checkpoint.
"""

from __future__ import annotations

import copy
import os

from configs.star_predcls_obb_train import cfg as _base_cfg


cfg = copy.deepcopy(_base_cfg)
rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]

rel_cfg["RPCM_RELATION_GRAPH_MODE"] = "dual_view"
rel_cfg["RPCM_REL_SUBJ_VIEW_ENABLED"] = True
rel_cfg["RPCM_REL_OBJ_VIEW_ENABLED"] = True
rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] = 0.0
rel_cfg["RPCM_TAIL_AUX_ENABLED"] = False
rel_cfg["RPCM_TAIL_AUX_PREDICATES"] = []
rel_cfg["OBJECT_REFINE_LOSS_WEIGHT"] = 0.0

# Match the controlled Base/DL schedule. At about 49 optimizer steps per
# epoch, validation plateaus around steps 5000--5600; the two decays occur
# near epochs 113 and 164.
cfg["SOLVER"]["BASE_LR"] = float(os.environ.get("BASE_LR", "0.016"))
cfg["SOLVER"]["WARMUP_ITERS"] = int(os.environ.get("WARMUP_ITERS", "500"))
cfg["SOLVER"]["MAX_EPOCHS"] = int(os.environ.get("MAX_EPOCHS", "220"))
cfg["SOLVER"]["STEPS"] = [
    int(step.strip())
    for step in os.environ.get("STEPS", "5500,8000").split(",")
    if step.strip()
]
cfg["SOLVER"]["VAL_START_PERIOD"] = int(os.environ.get("VAL_START_PERIOD", "120"))
cfg["SOLVER"]["VAL_PERIOD"] = int(os.environ.get("VAL_PERIOD", "2"))
cfg["SOLVER"]["CHECKPOINT_PERIOD"] = int(os.environ.get("CHECKPOINT_PERIOD", "2"))
cfg["SOLVER"]["VAL_SPLIT"] = os.environ.get("VAL_SPLIT", "val")
# Patience is counted in validation events, not epochs. Ten events correspond
# to about 20 epochs with the default validation cadence.
cfg["SOLVER"]["EARLY_STOP_ENABLED"] = os.environ.get(
    "EARLY_STOP_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
cfg["SOLVER"]["EARLY_STOP_METRIC"] = os.environ.get("EARLY_STOP_METRIC", "HR")
cfg["SOLVER"]["EARLY_STOP_PATIENCE"] = int(
    os.environ.get("EARLY_STOP_PATIENCE", "10")
)
cfg["SOLVER"]["EARLY_STOP_MIN_DELTA"] = float(
    os.environ.get("EARLY_STOP_MIN_DELTA", "0.0")
)
cfg["SOLVER"]["EARLY_STOP_START_PERIOD"] = int(
    os.environ.get(
        "EARLY_STOP_START_PERIOD",
        "170",
    )
)
cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get(
    "OUTPUT_DIR", "outputs/paper_submission/predcls/dual"
)
