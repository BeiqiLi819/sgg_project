"""Public STAR OBB SGCls Full configuration.

The relation branch uses Dual-view RCA and auxiliary logit adjustment. Final
evaluation uses statistical RSGP Hybrid 9000/1000. The benchmark-compatible
default filter-label source remains the STAR/SGG-ToolKit GT-label protocol.
"""

from __future__ import annotations

import copy
import os

from configs.star_sgcls_obb_dual_la_train import cfg as _cfg


cfg = copy.deepcopy(_cfg)
cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get(
    "OUTPUT_DIR", "outputs/star_sgcls_obb_full"
)
