"""Faithful training-protocol reconstruction of RPCM ``6850_4135.pth``.

This is not an additional graph architecture.  It uses the retained
SS/OO dual-view GCN and the exact single-prototype 6850-compatible head, but
restores the historical optimization budget and deterministic seed:

* short-edge 600 / long-edge 1000 preprocessing with aspect grouping;
* batch size 16, SGD, effective LR 0.016 and 500-step warmup;
* LR decays at optimizer steps 13,000 and 18,000;
* a hard stop at 20,000 optimizer steps;
* validation with PPG every 200 steps from step 14,000; and
* seed 1029 with cuDNN disabled by default to bound peak memory.

Only the frozen OBB detector is initialized from
``pretrained/OBB_swin_L_OBD.pth``.  The relation predictor starts from its
configured random/GloVe initialization.
"""

from __future__ import annotations

import copy
import os

from configs.star_predcls_obb_train import cfg as _base_cfg


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


cfg = copy.deepcopy(_base_cfg)
rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]

# Exact graph/head path identified from the historical checkpoint.
rel_cfg["RPCM_RELATION_GRAPH_MODE"] = "dual_view"
rel_cfg["RPCM_REL_SUBJ_VIEW_ENABLED"] = True
rel_cfg["RPCM_REL_OBJ_VIEW_ENABLED"] = True
rel_cfg["RPCM_LEGACY_6850_EXACT"] = True
rel_cfg["RPCM_LEGACY_NUM_PROTO"] = 1
rel_cfg["RPCM_LEGACY_USE_VIS_PROTO"] = False
rel_cfg["RPCM_LEGACY_PROTO_2D_COMPAT"] = True
rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] = 0.0
rel_cfg["RPCM_TAIL_AUX_ENABLED"] = False
rel_cfg["RPCM_TAIL_AUX_PREDICATES"] = []
rel_cfg["OBJECT_REFINE_LOSS_WEIGHT"] = 0.0

cfg["DATALOADER"]["TRAIN_BATCH_SIZE"] = 16
# Restore the source maskrcnn-benchmark input protocol for this historical
# reproduction only.  The paper's controlled Base/D/DL rows retain their
# established fit-inside-1024 preprocessing and ungrouped loader.
cfg["DATALOADER"]["ASPECT_RATIO_GROUPING"] = True
for split in ("TRAIN", "VAL", "TEST"):
    cfg["DATASETS"][split]["RESIZE_MODE"] = "short_edge"
    cfg["DATASETS"][split]["MIN_SIZE"] = 600
    cfg["DATASETS"][split]["MAX_SIZE"] = 1000

solver = cfg["SOLVER"]
# The source command requested 1e-3 and its optimizer builder multiplied by
# IMS_PER_BATCH=16, yielding the observed effective LR 0.016.
solver["IMS_PER_BATCH"] = int(os.environ.get("TRAIN_BATCH_SIZE", "16"))
solver["BASE_LR"] = float(os.environ.get("BASE_LR", "0.001"))
solver["LR_SCALE_BY_BATCH"] = True
solver["WARMUP_ITERS"] = int(os.environ.get("WARMUP_ITERS", "500"))
solver["MAX_ITER"] = int(os.environ.get("MAX_ITER", "20000"))
# MAX_ITER is authoritative.  This is only a safety ceiling for a loader with
# a different number of batches per epoch.
solver["MAX_EPOCHS"] = int(os.environ.get("MAX_EPOCHS", "500"))
solver["STEPS"] = [
    int(step.strip())
    for step in os.environ.get("STEPS", "13000,18000").split(",")
    if step.strip()
]
solver["ITERATION_COMPAT"] = True
solver["VALIDATE_ON_ITER"] = True
solver["VAL_START_PERIOD"] = int(os.environ.get("VAL_START_PERIOD", "14000"))
solver["VAL_PERIOD"] = int(os.environ.get("VAL_PERIOD", "200"))
solver["VAL_SPLIT"] = os.environ.get("VAL_SPLIT", "val")
solver["CHECKPOINT_PERIOD"] = int(os.environ.get("CHECKPOINT_PERIOD", "0"))
# The 17,600-step mR peak is narrow.  Historical reproduction must not be
# terminated by the newer validation-patience mechanism.
solver["EARLY_STOP_ENABLED"] = False
solver["OUTPUT_DIR"] = os.environ.get(
    "OUTPUT_DIR", "outputs/paper_submission/predcls/dual_6850_repro"
)

runtime = cfg["RUNTIME"]
runtime["SEED"] = int(os.environ.get("SEED", "1029"))
# The source runtime enabled deterministic cuDNN, but its workspace is
# prohibitively large in this project's retained detector/feature path.
# Disabling it changes only the convolution kernel backend, not model math,
# parameters, losses, or the optimization schedule.  Set DISABLE_CUDNN=0 for
# an exact source-runtime audit.
runtime["DISABLE_CUDNN"] = _env_bool("DISABLE_CUDNN", True)
runtime["CUDNN_BENCHMARK"] = False
runtime["CUDNN_DETERMINISTIC"] = True
runtime["STRICT_REPRODUCIBILITY"] = _env_bool("STRICT_REPRODUCIBILITY", False)
