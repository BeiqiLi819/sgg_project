"""PredCls source audit: complete original SGG-ToolKit RPCM predictor.

This audit row restores both halves of the source implementation:

* the six-mode heterogeneous ``gcn_collect_feat`` and residual update; and
* the original gated subject/object/union fusion, 300-D GloVe prototype
  projection, KMeans coarse prototypes, cosine classifier, and five
  prototype regularizers.

The paper Base/D/DL rows intentionally retain the later 6850-compatible
classifier. This source branch is an audit rather than a main-table row; the
controlled adjacency ablation is ``Base (Unified) -> D``.

Only the frozen OBB detector is initialized from
``pretrained/OBB_swin_L_OBD.pth``; relation parameters start from their
configured random/GloVe initialization.
"""

from __future__ import annotations

import copy
import os

from configs.star_predcls_obb_ablation_unified_rca_train import cfg as _unified_cfg


cfg = copy.deepcopy(_unified_cfg)
rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
rel_cfg["PREDICTOR"] = "RPCM_SGG_TOOLKIT_ORIGINAL"
rel_cfg["RPCM_RELATION_GRAPH_MODE"] = "sgg_toolkit"
rel_cfg["RPCM_GLOVE_INIT_MODE"] = "sgg_toolkit"
rel_cfg["RPCM_PROTO_EMBED_DIM"] = 300
rel_cfg["RPCM_PROTO_GLOVE_PATH"] = "glove/glove.6B.300d.txt"
rel_cfg["RPCM_FEAT_UPDATE_STEP"] = 4
rel_cfg["PREDICT_USE_BIAS"] = False
rel_cfg["RPCM_TAIL_AUX_ENABLED"] = False
rel_cfg["RPCM_TAIL_AUX_PREDICATES"] = []
rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] = 0.0
cfg["EXP_nums"] = 30

# Use the common data-driven schedule for the controlled rows. With about 49
# optimizer steps per epoch, these milestones occur near epochs 113 and 164.
# The first decay is placed at the observed Base HMR plateau (best at step
# 5586), rather than after it.
cfg["SOLVER"]["STEPS"] = [
    int(step.strip())
    for step in os.environ.get("STEPS", "5500,8000").split(",")
    if step.strip()
]

# The source classifier follows a different convergence curve from the paper
# Base/D/DL rows. Start recording validation checkpoints at
# epoch 70, but delay patience accounting until both LR milestones have run.
cfg["SOLVER"]["VAL_START_PERIOD"] = int(
    os.environ.get("VAL_START_PERIOD", "70")
)
# Do not let validation early stopping terminate the run before the second LR
# decay around epoch 164.
cfg["SOLVER"]["EARLY_STOP_START_PERIOD"] = int(
    os.environ.get("EARLY_STOP_START_PERIOD", "170")
)

cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get(
    "OUTPUT_DIR",
    "outputs/paper_submission/predcls/base_original",
)
