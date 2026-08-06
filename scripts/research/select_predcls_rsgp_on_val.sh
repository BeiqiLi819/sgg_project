#!/usr/bin/env bash
set -euo pipefail

# Select the RSGP mode/protected-pool ratio on validation only. Never point
# this script at test when choosing paper hyperparameters.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SPLIT=val \
CONFIG="${CONFIG:-configs/star_predcls_obb_ablation_dual_la_train.py}" \
CHECKPOINT="${CHECKPOINT:-outputs/paper_submission/predcls/dual_la/model_best_HR.pth}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/paper_submission/predcls/dual_la/rsgp_val_selection}" \
bash "${ROOT_DIR}/scripts/_internal/eval_rsgp_grid.sh"
