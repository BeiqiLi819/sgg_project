"""Paper-main STAR OBB SGCls configuration: dual-view RCA + auxiliary LA.

This file reuses the source-compatible SGCls detector/object-refinement
protocol from ``star_sgcls_obb_train.py`` while removing the historical HPRC
(``tail_aux``) residual head. RSGP remains an inference/validation-time pair
filter and does not alter the supervised relation training pairs.
"""

from __future__ import annotations

import copy
import os

from configs.star_sgcls_obb_train import cfg as _task_cfg


cfg = copy.deepcopy(_task_cfg)
rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]

rel_cfg["RPCM_RELATION_GRAPH_MODE"] = "dual_view"
rel_cfg["RPCM_REL_SUBJ_VIEW_ENABLED"] = True
rel_cfg["RPCM_REL_OBJ_VIEW_ENABLED"] = True
rel_cfg["PREDICATE_LOSS_TYPE"] = "ce"
rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] = float(
    os.environ.get("PREDICATE_AUX_LOGIT_ADJUST_WEIGHT", "0.1")
)
rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_TAU"] = float(
    os.environ.get("PREDICATE_AUX_LOGIT_ADJUST_TAU", "0.5")
)
rel_cfg["RPCM_TAIL_AUX_ENABLED"] = False
rel_cfg["RPCM_TAIL_AUX_PREDICATES"] = []

# SGCls predicts object labels, so object-refinement CE remains trainable.
rel_cfg["OBJECT_REFINE_LOSS_WEIGHT"] = 1.0

# Select the relation checkpoint with PPG, then evaluate the exact checkpoint
# with RSGP. This keeps the filter ablation independent of checkpoint choice.
_filter_method = os.environ.get("FILTER_METHOD", "PPG").upper()
if _filter_method not in {"PPG", "PPN", "RSGP"}:
    raise ValueError("FILTER_METHOD must be PPG, PPN, or RSGP")
rel_cfg["TEST_FILTER_METHOD"] = _filter_method
rel_cfg["PPG_ENABLED"] = _filter_method == "PPG"
rel_cfg["PPN_ENABLED"] = _filter_method == "PPN"
rel_cfg["RSGP_ENABLED"] = _filter_method == "RSGP"

cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get(
    "OUTPUT_DIR", "outputs/paper_submission/sgcls"
)
cfg["SOLVER"]["VAL_SPLIT"] = os.environ.get("VAL_SPLIT", "val")
