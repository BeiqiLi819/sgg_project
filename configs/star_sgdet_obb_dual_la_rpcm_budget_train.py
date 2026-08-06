"""SGDet Dual+LA control using the original SGG-ToolKit RPCM budget.

This is a training-budget control, not a second method. It keeps the current
paper-main detector/cache protocol, dual-view RCA, auxiliary LA, object
refinement, and mandatory pair filter, while restoring the optimization
settings from ``Scripts/LOBB_RPCM_sgdet_train.sh``:

* batch size 2;
* configured base learning rate 1e-3 and effective learning rate 2e-3,
  matching the source optimizer's ``rl_factor=IMS_PER_BATCH`` behavior;
* 5,000 optimizer iterations;
* LR milestones at 3,000 and 4,000 iterations;
* 500 warmup iterations;
* validation at the first epoch boundary at or after each nominal 1,000-step
  interval (about every three epochs for the current loader), because the
  trainer executes validation only between epochs.
"""

from __future__ import annotations

import copy
import os

from configs.star_sgdet_obb_dual_la_train import cfg as _main_cfg


cfg = copy.deepcopy(_main_cfg)

# Freeze the comparison budget rather than inheriting generic shell exports.
cfg["DATALOADER"]["TRAIN_BATCH_SIZE"] = 2
cfg["DATALOADER"]["VAL_BATCH_SIZE"] = 1
cfg["DATALOADER"]["TEST_BATCH_SIZE"] = 1
cfg["SOLVER"]["IMS_PER_BATCH"] = 2
cfg["TEST"]["IMS_PER_BATCH"] = 1

cfg["SOLVER"]["BASE_LR"] = 1e-3
cfg["SOLVER"]["LR_SCALE_BY_BATCH"] = True
cfg["SOLVER"]["WARMUP_ITERS"] = 500
cfg["SOLVER"]["MAX_ITER"] = 5000
cfg["SOLVER"]["MAX_EPOCHS"] = 100000
cfg["SOLVER"]["STEPS"] = [3000, 4000]
cfg["SOLVER"]["ITERATION_COMPAT"] = True
cfg["SOLVER"]["VAL_START_PERIOD"] = 1000
cfg["SOLVER"]["VAL_PERIOD"] = 1000
cfg["SOLVER"]["CHECKPOINT_PERIOD"] = 0

cfg["SOLVER"]["OUTPUT_DIR"] = os.environ.get(
    "OUTPUT_DIR", "outputs/paper_submission/sgdet_rpcm_budget"
)
cfg["SOLVER"]["VAL_SPLIT"] = "val"
