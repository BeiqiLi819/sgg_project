#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FILTER_METHOD=PPG
OUTPUT_DIR="${OUTPUT_DIR:-outputs/paper_submission/sgcls}"
CONFIG="${CONFIG:-configs/star_sgcls_obb_dual_la_train.py}"
AUTO_TEST_AFTER_TRAIN="${AUTO_TEST_AFTER_TRAIN:-1}"
RUN_BACKGROUND="${RUN_BACKGROUND:-1}"
WORKFLOW_WORKER="${SGCLS_WORKFLOW_WORKER:-0}"
RESUME="${RESUME:-}"

if [[ "${AUTO_TEST_AFTER_TRAIN}" != "0" && "${RUN_BACKGROUND}" != "0" && "${WORKFLOW_WORKER}" != "1" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  nohup env \
    SGCLS_WORKFLOW_WORKER=1 \
    AUTO_TEST_AFTER_TRAIN=1 \
    RUN_BACKGROUND=0 \
    CONFIG="${CONFIG}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    RESUME="${RESUME}" \
    bash "${SCRIPT_DIR}/run_star_sgcls_experiment.sh" > "${OUTPUT_DIR}/workflow.log" 2>&1 &
  echo "$!" > "${OUTPUT_DIR}/workflow.pid"
  echo "Started SGCls train-and-test workflow with PID $!"
  echo "Workflow log: ${OUTPUT_DIR}/workflow.log"
  exit 0
fi

# Paper-main SGCls route: source-compatible frozen OBB detector/object
# refinement + dual-view RCA + auxiliary LA. OBB_swin_L_OBD.pth is loaded
# inside the config; the relation stack starts from scratch by default.
# SGCLS_FILTER_LABEL_SOURCE=gt matches the SGG-Toolkit STAR pair-filter
# protocol. Prefix the command with SGCLS_FILTER_LABEL_SOURCE=pred for a strict
# predicted-label ablation.
CONFIG="${CONFIG}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
FILTER_METHOD="${FILTER_METHOD}" \
VAL_SPLIT=val \
RSGP_MODE="${RSGP_MODE:-HYBRID}" \
RSGP_TOPK="${RSGP_TOPK:-10000}" \
RSGP_PPG_PROTECTED_TOPK="${RSGP_PPG_PROTECTED_TOPK:-9000}" \
INIT_RPCM="" \
RUN_BACKGROUND="${RUN_BACKGROUND}" \
bash "${ROOT_DIR}/scripts/_internal/run_star_experiment.sh"

if [[ "${AUTO_TEST_AFTER_TRAIN}" != "0" ]]; then
  BEST_CHECKPOINT="${OUTPUT_DIR}/model_best_HR.pth"
  if [[ ! -f "${BEST_CHECKPOINT}" ]]; then
    echo "SGCls training completed but best checkpoint is missing: ${BEST_CHECKPOINT}" >&2
    exit 1
  fi
  echo "SGCls training complete; evaluating best validation-HMR checkpoint on test."
  CONFIG="${CONFIG}" \
  CHECKPOINT="${BEST_CHECKPOINT}" \
  SPLIT=test \
  FILTER_METHOD=PPG \
  SGCLS_FILTER_LABEL_SOURCE="${SGCLS_FILTER_LABEL_SOURCE:-gt}" \
  PROFILE_INFERENCE=1 \
  OUTPUT_DIR="${OUTPUT_DIR}/test/legacy/ppg" \
  RUN_BACKGROUND=0 \
  bash "${ROOT_DIR}/scripts/_internal/eval_star_sgcls.sh"
fi
