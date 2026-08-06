"""Decision experiment: Dual-view RCA + LA + original SGG-ToolKit head.

This branch keeps the current 6850-compatible pairwise extractor and
dual-view/all-layer RCA front end used by the main DL experiment.  Only the
post-GNN relation classifier is replaced by the source SGG-ToolKit pipeline:
gated subject/object/union fusion, 300-D GloVe predicate prototypes,
per-forward KMeans coarse prototypes, cosine logits, and the five source
prototype losses.

It starts from the frozen detector checkpoint only and writes to an isolated
directory.  It cannot resume a DL or complete-Original checkpoint.
"""

from __future__ import annotations

import copy
import os

from configs.star_predcls_obb_ablation_dual_la_train import cfg as _dual_la_cfg


cfg = copy.deepcopy(_dual_la_cfg)
rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]

rel_cfg["PREDICTOR"] = "RPCM_SGG_TOOLKIT_ORIGINAL"
rel_cfg["RPCM_RELATION_GRAPH_MODE"] = "dual_view"
rel_cfg["RPCM_REL_SUBJ_VIEW_ENABLED"] = True
rel_cfg["RPCM_REL_OBJ_VIEW_ENABLED"] = True

# Keep the current 200-D pairwise object initialization so this run tests the
# post-GNN replacement. The predictor independently uses the literal source
# 300-D object/predicate GloVe rules in its gated prototype classifier.
rel_cfg["RPCM_GLOVE_INIT_MODE"] = "rpcm"
rel_cfg["RPCM_PROTO_EMBED_DIM"] = 300
rel_cfg["RPCM_PROTO_GLOVE_PATH"] = "glove/glove.6B.300d.txt"
rel_cfg["RPCM_FEAT_UPDATE_STEP"] = 4
rel_cfg["PREDICT_USE_BIAS"] = False
rel_cfg["RPCM_TAIL_AUX_ENABLED"] = False
rel_cfg["RPCM_TAIL_AUX_PREDICATES"] = []
rel_cfg["OBJECT_REFINE_LOSS_WEIGHT"] = 0.0
cfg["EXP_nums"] = 30

# Use the same schedule as the controlled Base/D/DL rows. The current run
# reaches its best validation range around steps 5000--5600, so the first
# decay at 5500 begins refinement immediately after that plateau.
cfg["SOLVER"]["STEPS"] = [
    int(step.strip())
    for step in os.environ.get("STEPS", "5500,8000").split(",")
    if step.strip()
]

# Observe the source head from epoch 70, but delay patience accounting until
# both LR milestones have run.
cfg["SOLVER"]["VAL_START_PERIOD"] = int(
    os.environ.get("VAL_START_PERIOD", "70")
)
cfg["SOLVER"]["EARLY_STOP_START_PERIOD"] = int(
    os.environ.get("EARLY_STOP_START_PERIOD", "170")
)

cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get(
    "OUTPUT_DIR",
    "outputs/paper_submission/predcls/dual_la_original_head",
)
