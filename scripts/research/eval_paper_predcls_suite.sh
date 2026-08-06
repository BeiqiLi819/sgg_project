#!/usr/bin/env bash
set -euo pipefail

# Final PredCls test suite. It evaluates controlled rows under PPG, compares
# Base(Unified)/Dual under the same statistical RSGP graph, and finally compares
# PPG/PPN/RSGP on the exact same Dual+LA checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SUITE_DIR="${SUITE_DIR:-outputs/paper_submission/predcls}"
WORKER="${PAPER_EVAL_WORKER:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RSGP_ROLE_MODE="${RSGP_ROLE_MODE:-statistical}"
RSGP_STRUCTURAL_PRIOR_PATH="${RSGP_STRUCTURAL_PRIOR_PATH:-pretrained/rsgp_structural_prior.json}"

cd "${ROOT_DIR}"
mkdir -p "${SUITE_DIR}"

if [[ "${WORKER}" != "1" ]]; then
  LOG_FILE="${SUITE_DIR}/final_test_suite.log"
  PID_FILE="${SUITE_DIR}/final_test_suite.pid"
  nohup env \
    PAPER_EVAL_WORKER=1 \
    SUITE_DIR="${SUITE_DIR}" \
    RSGP_MODE="${RSGP_MODE:-HYBRID}" \
    RSGP_PPG_PROTECTED_TOPK="${RSGP_PPG_PROTECTED_TOPK:-9000}" \
    RSGP_ROLE_MODE="${RSGP_ROLE_MODE}" \
    RSGP_STRUCTURAL_PRIOR_PATH="${RSGP_STRUCTURAL_PRIOR_PATH}" \
    bash "$0" > "${LOG_FILE}" 2>&1 &
  echo "$!" > "${PID_FILE}"
  echo "Started final PredCls test suite with PID $!"
  echo "Log: ${LOG_FILE}"
  exit 0
fi

for row in BASE D DL; do
  case "${row}" in
    BASE) slug="unified" ;;
    D) slug="dual" ;;
    DL) slug="dual_la" ;;
  esac
  existing="${SUITE_DIR}/${slug}/test/ppg/test_metrics.json"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${existing}" ]]; then
    echo "Skip existing ${existing}"
  else
    SPLIT=test FILTER_METHOD=PPG RUN_BACKGROUND=0 \
      bash "${ROOT_DIR}/scripts/_internal/eval_predcls_minimal_ablation.sh" "${row}"
  fi
done

# A strict 2x2 graph/filter comparison. In PredCls, U and D see the same GT
# boxes and labels, so statistical RSGP constructs the same candidate graph;
# only the learned relation graph aggregation differs between these rows.
for row in BASE D; do
  case "${row}" in
    BASE) slug="unified" ;;
    D) slug="dual" ;;
  esac
  existing="${SUITE_DIR}/${slug}/test/rsgp_${RSGP_ROLE_MODE}/test_metrics.json"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${existing}" ]]; then
    echo "Skip existing ${existing}"
  else
    SPLIT=test FILTER_METHOD=RSGP RUN_BACKGROUND=0 \
      RSGP_MODE="${RSGP_MODE:-HYBRID}" \
      RSGP_PPG_PROTECTED_TOPK="${RSGP_PPG_PROTECTED_TOPK:-9000}" \
      RSGP_ROLE_MODE="${RSGP_ROLE_MODE}" \
      RSGP_STRUCTURAL_PRIOR_PATH="${RSGP_STRUCTURAL_PRIOR_PATH}" \
      bash "${ROOT_DIR}/scripts/_internal/eval_predcls_minimal_ablation.sh" "${row}"
  fi
done

for method in PPN RSGP; do
  method_slug="${method,,}"
  if [[ "${method}" == "RSGP" ]]; then
    method_slug="${method_slug}_${RSGP_ROLE_MODE}"
  fi
  existing="${SUITE_DIR}/dual_la/test/${method_slug}/test_metrics.json"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${existing}" ]]; then
    echo "Skip existing ${existing}"
  else
    SPLIT=test FILTER_METHOD="${method}" RUN_BACKGROUND=0 \
      RSGP_MODE="${RSGP_MODE:-HYBRID}" \
      RSGP_PPG_PROTECTED_TOPK="${RSGP_PPG_PROTECTED_TOPK:-9000}" \
      RSGP_ROLE_MODE="${RSGP_ROLE_MODE}" \
      RSGP_STRUCTURAL_PRIOR_PATH="${RSGP_STRUCTURAL_PRIOR_PATH}" \
      bash "${ROOT_DIR}/scripts/_internal/eval_predcls_minimal_ablation.sh" DL
  fi
done

echo "[$(date '+%F %T')] final PredCls test suite complete"
