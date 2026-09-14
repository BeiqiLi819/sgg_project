#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT="${CHECKPOINT:-pretrained/RPCM_6850_Dual_bgfirst.pth}"
CONFIG="${CONFIG:-configs/star_predcls_obb_ablation_dual_rca_train.py}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rpcm_6850_dual_eval}"
CHECKPOINT_LOAD_MODE="${CHECKPOINT_LOAD_MODE:-legacy-rpcm}"
SPLIT="${SPLIT:-test}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
FILTER_METHOD="${FILTER_METHOD:-PPG}"
RSGP_MODE="${RSGP_MODE:-HYBRID}"
RSGP_TOPK="${RSGP_TOPK:-10000}"
RSGP_PPG_PROTECTED_TOPK="${RSGP_PPG_PROTECTED_TOPK:-8000}"

if [[ -z "${CHECKPOINT:-}" ]]; then
  for candidate in \
    "pretrained/RPCM_6850_Dual_bgfirst.pth" \
    "outputs/rpcm_toolkit_2x/hprc_6850_posttrain/model_best_HR.pth" \
    "outputs/star_predcls_obb_train/model_best_HR.pth" \
    "outputs/star_predcls_obb_train/model_last.pth"; do
    if [[ -f "${ROOT_DIR}/${candidate}" ]]; then
      CHECKPOINT="${candidate}"
      break
    fi
  done
fi

if [[ -z "${CHECKPOINT:-}" ]]; then
  echo "No default predcls checkpoint found. Set CHECKPOINT=/path/to/model.pth." >&2
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
OUTPUT_DIR="${OUTPUT_DIR}" \
bash "${SCRIPT_DIR}/eval_once.sh"
