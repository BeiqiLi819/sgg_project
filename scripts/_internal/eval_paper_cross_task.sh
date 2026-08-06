#!/usr/bin/env bash
set -euo pipefail

# Evaluate PPG or RSGP on one fixed SGCls/SGDet checkpoint.
#
# Protocol:
#   legacy : STAR/SGG-ToolKit-compatible filter labels (GT for SGCls,
#            matched GT for SGDet)
#   pred   : strict predicted-label filtering
#
# Examples:
#   bash scripts/_internal/eval_paper_cross_task.sh sgcls PPG legacy
#   bash scripts/_internal/eval_paper_cross_task.sh sgcls RSGP pred
#   bash scripts/_internal/eval_paper_cross_task.sh sgdet RSGP pred

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="${1:-}"
FILTER_METHOD="${2:-}"
PROTOCOL="${3:-legacy}"
SPLIT="${SPLIT:-test}"
RSGP_ROLE_MODE="${RSGP_ROLE_MODE:-statistical}"
RSGP_STRUCTURAL_PRIOR_PATH="${RSGP_STRUCTURAL_PRIOR_PATH:-pretrained/rsgp_structural_prior.json}"

TASK="${TASK,,}"
FILTER_METHOD="${FILTER_METHOD^^}"
PROTOCOL="${PROTOCOL,,}"
FILTER_SLUG="${FILTER_METHOD,,}"
if [[ "${FILTER_METHOD}" == "RSGP" ]]; then
  FILTER_SLUG="${FILTER_SLUG}_${RSGP_ROLE_MODE}"
fi

if [[ "${FILTER_METHOD}" != "PPG" && "${FILTER_METHOD}" != "RSGP" ]]; then
  echo "Filter must be PPG or RSGP." >&2
  exit 2
fi
if [[ "${PROTOCOL}" != "legacy" && "${PROTOCOL}" != "pred" ]]; then
  echo "Protocol must be legacy or pred." >&2
  exit 2
fi

case "${TASK}" in
  sgcls)
    CONFIG="${CONFIG:-configs/star_sgcls_obb_dual_la_train.py}"
    RUN_DIR="${RUN_DIR:-outputs/paper_submission/sgcls}"
    CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/model_best_HR.pth}"
    if [[ "${PROTOCOL}" == "legacy" ]]; then
      LABEL_SOURCE="gt"
    else
      LABEL_SOURCE="pred"
    fi
    OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/${SPLIT}/${PROTOCOL}/${FILTER_SLUG}}"
    CONFIG="${CONFIG}" CHECKPOINT="${CHECKPOINT}" SPLIT="${SPLIT}" \
    FILTER_METHOD="${FILTER_METHOD}" SGCLS_FILTER_LABEL_SOURCE="${LABEL_SOURCE}" \
    PPG_TOPK=10000 RSGP_TOPK=10000 \
    RSGP_MODE="${RSGP_MODE:-HYBRID}" \
    RSGP_PPG_PROTECTED_TOPK="${RSGP_PPG_PROTECTED_TOPK:-9000}" \
    RSGP_ROLE_MODE="${RSGP_ROLE_MODE}" \
    RSGP_STRUCTURAL_PRIOR_PATH="${RSGP_STRUCTURAL_PRIOR_PATH}" \
    PROFILE_INFERENCE="${PROFILE_INFERENCE:-1}" \
    OUTPUT_DIR="${OUTPUT_DIR}" RUN_BACKGROUND="${RUN_BACKGROUND:-1}" \
    bash "${SCRIPT_DIR}/eval_star_sgcls.sh"
    ;;
  sgdet)
    CONFIG="${CONFIG:-configs/star_sgdet_obb_dual_la_rpcm_budget_train.py}"
    RUN_DIR="${RUN_DIR:-outputs/paper_submission/sgdet_rpcm_budget}"
    CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/model_last.pth}"
    if [[ "${PROTOCOL}" == "legacy" ]]; then
      LABEL_SOURCE="matched_gt"
    else
      LABEL_SOURCE="pred"
    fi
    OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/${SPLIT}/${PROTOCOL}/${FILTER_SLUG}}"
    CONFIG="${CONFIG}" CHECKPOINT="${CHECKPOINT}" SPLIT="${SPLIT}" \
    FILTER_METHOD="${FILTER_METHOD}" SGDET_FILTER_LABEL_SOURCE="${LABEL_SOURCE}" \
    PPG_TOPK=10000 RSGP_TOPK=10000 \
    RSGP_MODE="${RSGP_MODE:-HYBRID}" \
    RSGP_PPG_PROTECTED_TOPK="${RSGP_PPG_PROTECTED_TOPK:-9000}" \
    RSGP_ROLE_MODE="${RSGP_ROLE_MODE}" \
    RSGP_STRUCTURAL_PRIOR_PATH="${RSGP_STRUCTURAL_PRIOR_PATH}" \
    PROFILE_INFERENCE="${PROFILE_INFERENCE:-1}" \
    SGDET_DETECTION_CACHE_ENABLED=1 \
    SGDET_DETECTION_CACHE_DIR="${SGDET_DETECTION_CACHE_DIR:-star_sgdet_detection_cache}" \
    SGDET_DETECTION_CACHE_REQUIRE_HIT=1 \
    OUTPUT_DIR="${OUTPUT_DIR}" RUN_BACKGROUND="${RUN_BACKGROUND:-1}" \
    bash "${SCRIPT_DIR}/eval_star_sgdet.sh"
    ;;
  *)
    echo "Usage: bash scripts/_internal/eval_paper_cross_task.sh sgcls|sgdet PPG|RSGP legacy|pred" >&2
    exit 2
    ;;
esac
