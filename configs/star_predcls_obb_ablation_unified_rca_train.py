"""Controlled PredCls Base: unified adjacency with the current RCA GNN.

This branch keeps the same detector, RPCM parameters, prototype head, CE loss,
training schedule, all-layer aggregation, and default PPG validation filter as
the dual-view row. Only relation adjacency changes: all relations sharing any
endpoint are merged into one graph. This isolates unified versus dual-view
adjacency without switching to the original SGG-ToolKit collect/update block.

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

# Use the same current GNN/update modules as dual_view; only the E x E
# relation adjacency changes.
rel_cfg["RPCM_RELATION_GRAPH_MODE"] = "unified"
rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] = 0.0
rel_cfg["RPCM_TAIL_AUX_ENABLED"] = False
rel_cfg["RPCM_TAIL_AUX_PREDICATES"] = []
rel_cfg["OBJECT_REFINE_LOSS_WEIGHT"] = 0.0

# The current loader performs about 49 optimizer steps per epoch. Validation
# curves from both the complete source head and the Dual+LA source-head run
# plateau around steps 5000--5600 while training loss keeps decreasing. Decay
# at steps 5500/8000 (roughly epochs 113/164) therefore gives both branches a
# useful low-LR refinement phase inside the 220-epoch budget. These defaults
# are identical across controlled Base/D/DL rows. This Unified row is the
# paper-facing Base.
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
# Stop after ten consecutive validation events without a new HMR@2000 best.
# With VAL_PERIOD=2 this is normally a 20-epoch no-improvement window.
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
    "OUTPUT_DIR", "outputs/paper_submission/predcls/unified"
)
