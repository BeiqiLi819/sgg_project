#!/usr/bin/env bash
set -euo pipefail

# Train the current SGDet Dual+LA model with the exact optimization budget of
# SGG-ToolKit's LOBB_RPCM_sgdet_train.sh. The detector cache is unchanged.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${CONFIG:-configs/star_sgdet_obb_dual_la_rpcm_budget_train.py}" \
OUTPUT_DIR="${OUTPUT_DIR:-outputs/paper_submission/sgdet_rpcm_budget}" \
VAL_SPLIT=val \
FILTER_METHOD=PPG \
RUN_BACKGROUND="${RUN_BACKGROUND:-1}" \
bash "${ROOT_DIR}/scripts/_internal/run_star_sgdet_experiment.sh"
