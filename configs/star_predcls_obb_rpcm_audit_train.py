"""One-off local audit of the complete source SGG-ToolKit/RPCM predictor.

This configuration is deliberately outside the controlled Base/D/DL ablation.
It uses the source heterogeneous collect/update block and source
GloVe/gated-fusion/KMeans prototype classifier, with plain CE,
detector-only initialization, and validation-only checkpoint selection. The
published STAR/RPCM number remains an external reference.
"""

from __future__ import annotations

import copy
import os

from configs.star_predcls_obb_train import cfg as _base_cfg


cfg = copy.deepcopy(_base_cfg)
rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
rel_cfg["PREDICTOR"] = "RPCM_SGG_TOOLKIT_ORIGINAL"
rel_cfg["RPCM_RELATION_GRAPH_MODE"] = "sgg_toolkit"
rel_cfg["RPCM_GLOVE_INIT_MODE"] = "sgg_toolkit"
rel_cfg["RPCM_PROTO_EMBED_DIM"] = 300
rel_cfg["RPCM_PROTO_GLOVE_PATH"] = "glove/glove.6B.300d.txt"
rel_cfg["RPCM_FEAT_UPDATE_STEP"] = 4
rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] = 0.0
rel_cfg["RPCM_TAIL_AUX_ENABLED"] = False
rel_cfg["RPCM_TAIL_AUX_PREDICATES"] = []
rel_cfg["OBJECT_REFINE_LOSS_WEIGHT"] = 0.0
rel_cfg["PREDICT_USE_BIAS"] = False
cfg["EXP_nums"] = 30

cfg["SOLVER"]["BASE_LR"] = float(os.environ.get("BASE_LR", "0.016"))
cfg["SOLVER"]["WARMUP_ITERS"] = int(os.environ.get("WARMUP_ITERS", "500"))
cfg["SOLVER"]["MAX_EPOCHS"] = int(os.environ.get("MAX_EPOCHS", "300"))
cfg["SOLVER"]["STEPS"] = [
    int(step.strip())
    for step in os.environ.get("STEPS", "10000,14000,16000").split(",")
    if step.strip()
]
cfg["SOLVER"]["VAL_START_PERIOD"] = int(os.environ.get("VAL_START_PERIOD", "70"))
cfg["SOLVER"]["VAL_PERIOD"] = int(os.environ.get("VAL_PERIOD", "2"))
cfg["SOLVER"]["CHECKPOINT_PERIOD"] = int(os.environ.get("CHECKPOINT_PERIOD", "4"))
cfg["SOLVER"]["VAL_SPLIT"] = "val"
cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get(
    "OUTPUT_DIR", "outputs/paper_submission/rpcm_audit"
)
