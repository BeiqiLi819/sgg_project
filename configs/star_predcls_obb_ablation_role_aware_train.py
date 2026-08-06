"""PredCls trial: chain-complete role-aware RCA with the current head.

The dense candidate support and main GNN are identical to the controlled
Unified row.  Four zero-output low-rank adapters distinguish shared-subject
(SS), shared-object (OO), and the two directed cross-role chain views (OS/SO).
The initial forward result is exactly the Unified model; training determines
whether endpoint roles add useful information.
"""

from __future__ import annotations

import copy
import os

from configs.star_predcls_obb_ablation_unified_rca_train import cfg as _unified_cfg


cfg = copy.deepcopy(_unified_cfg)
rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]

rel_cfg["RPCM_RELATION_GRAPH_MODE"] = "role_aware"
rel_cfg["RPCM_ROLE_AWARE_MAX_LOG_WEIGHT"] = float(
    os.environ.get("RPCM_ROLE_AWARE_MAX_LOG_WEIGHT", "1.0")
)
rel_cfg["RPCM_ROLE_AWARE_RESIDUAL_MAX_WEIGHT"] = float(
    os.environ.get("RPCM_ROLE_AWARE_RESIDUAL_MAX_WEIGHT", "0.25")
)
rel_cfg["RPCM_ROLE_AWARE_ADAPTER_RANK"] = int(
    os.environ.get("RPCM_ROLE_AWARE_ADAPTER_RANK", "32")
)
rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] = 0.0

cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get(
    "OUTPUT_DIR", "outputs/paper_submission/predcls/role_aware"
)
