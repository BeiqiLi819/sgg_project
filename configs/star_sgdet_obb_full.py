"""Public STAR OBB SGDet Full configuration.

This exposes the paper's fixed 5,000-step SGDet budget, frozen OBB detector,
v5 detection-cache protocol, Dual-view RCA, auxiliary logit adjustment, and
statistical RSGP Hybrid 9000/1000 final filter.
"""

from __future__ import annotations

import copy
import os

from configs.star_sgdet_obb_dual_la_rpcm_budget_train import cfg as _cfg


cfg = copy.deepcopy(_cfg)
cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get(
    "OUTPUT_DIR", "outputs/star_sgdet_obb_full"
)
