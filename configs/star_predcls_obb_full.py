"""Public STAR OBB PredCls Full configuration.

Full consists of the Dual-view RCA relation graph, auxiliary logit adjustment,
and the validation-selected statistical RSGP Hybrid 9000/1000 filter. RSGP is
inference-only; training still uses PPG for checkpoint selection.
"""

from __future__ import annotations

import copy
import os

from configs.star_predcls_obb_ablation_dual_la_train import cfg as _cfg


cfg = copy.deepcopy(_cfg)
cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get(
    "OUTPUT_DIR", "outputs/star_predcls_obb_full"
)
