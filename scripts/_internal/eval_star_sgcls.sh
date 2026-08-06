#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG="${CONFIG:-configs/star_sgcls_obb_dual_la_train.py}"
CHECKPOINT_LOAD_MODE="${CHECKPOINT_LOAD_MODE:-full}"
SPLIT="${SPLIT:-test}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
FILTER_METHOD="${FILTER_METHOD:-RSGP}"
FILTER_METHOD="${FILTER_METHOD^^}"
RSGP_MODE="${RSGP_MODE:-HYBRID}"
RSGP_TOPK="${RSGP_TOPK:-10000}"
RSGP_PPG_PROTECTED_TOPK="${RSGP_PPG_PROTECTED_TOPK:-9000}"
RSGP_ROLE_MODE="${RSGP_ROLE_MODE:-statistical}"
RSGP_STRUCTURAL_PRIOR_PATH="${RSGP_STRUCTURAL_PRIOR_PATH:-pretrained/rsgp_structural_prior.json}"
FILTER_SLUG="${FILTER_METHOD,,}"
if [[ "${FILTER_METHOD}" == "RSGP" ]]; then
  FILTER_SLUG="${FILTER_SLUG}_${RSGP_ROLE_MODE}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-outputs/paper_submission/sgcls/test/${FILTER_SLUG}}"
SGCLS_FILTER_LABEL_SOURCE="${SGCLS_FILTER_LABEL_SOURCE:-gt}"

if [[ -z "${CHECKPOINT:-}" ]]; then
  for candidate in \
    "outputs/paper_submission/sgcls/model_best_HR.pth" \
    "outputs/paper_submission/sgcls/model_best_mR.pth" \
    "outputs/paper_submission/sgcls/model_best_R.pth" \
    "outputs/paper_submission/sgcls/model_last.pth"; do
    if [[ -f "${ROOT_DIR}/${candidate}" ]]; then
      CHECKPOINT="${candidate}"
      break
    fi
  done
fi

if [[ -z "${CHECKPOINT:-}" ]]; then
  echo "No default sgcls checkpoint found. Set CHECKPOINT=/path/to/model.pth." >&2
  exit 1
fi

CONFIG="${CONFIG}" \
CHECKPOINT="${CHECKPOINT}" \
CHECKPOINT_LOAD_MODE="${CHECKPOINT_LOAD_MODE}" \
SPLIT="${SPLIT}" \
TEST_BATCH_SIZE="${TEST_BATCH_SIZE}" \
VAL_BATCH_SIZE="${VAL_BATCH_SIZE}" \
FILTER_METHOD="${FILTER_METHOD}" \
RSGP_MODE="${RSGP_MODE}" \
RSGP_TOPK="${RSGP_TOPK}" \
RSGP_PPG_PROTECTED_TOPK="${RSGP_PPG_PROTECTED_TOPK}" \
RSGP_ROLE_MODE="${RSGP_ROLE_MODE}" \
RSGP_STRUCTURAL_PRIOR_PATH="${RSGP_STRUCTURAL_PRIOR_PATH}" \
SGCLS_FILTER_LABEL_SOURCE="${SGCLS_FILTER_LABEL_SOURCE}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash "${SCRIPT_DIR}/eval_once.sh"
