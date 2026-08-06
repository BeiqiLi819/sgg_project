#!/usr/bin/env bash
set -euo pipefail

# Queue SGCls/SGDet PPG-vs-RSGP comparisons for both legacy-compatible and
# strict predicted-label protocols. Each pair uses one identical checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SUITE_DIR="${SUITE_DIR:-outputs/paper_submission/cross_task}"
WORKER="${CROSS_TASK_EVAL_WORKER:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RSGP_ROLE_MODE="${RSGP_ROLE_MODE:-statistical}"
RSGP_STRUCTURAL_PRIOR_PATH="${RSGP_STRUCTURAL_PRIOR_PATH:-pretrained/rsgp_structural_prior.json}"
# Recovered task checkpoints are fixed here for the cross-task completeness
# experiment. PredCls remains the validation-controlled main experiment.
SGCLS_CHECKPOINT="${SGCLS_CHECKPOINT:-outputs/star_sgcls_obb_dual_la_train/model_best_HR.pth}"
SGDET_CHECKPOINT="${SGDET_CHECKPOINT:-outputs/star_sgdet_obb_dual_la_rpcm_budget/model_last.pth}"

cd "${ROOT_DIR}"
mkdir -p "${SUITE_DIR}"

if [[ "${WORKER}" != "1" ]]; then
  LOG_FILE="${SUITE_DIR}/suite.log"
  nohup env \
    CROSS_TASK_EVAL_WORKER=1 \
    SUITE_DIR="${SUITE_DIR}" \
    RSGP_MODE="${RSGP_MODE:-HYBRID}" \
    RSGP_PPG_PROTECTED_TOPK="${RSGP_PPG_PROTECTED_TOPK:-9000}" \
    RSGP_ROLE_MODE="${RSGP_ROLE_MODE}" \
    RSGP_STRUCTURAL_PRIOR_PATH="${RSGP_STRUCTURAL_PRIOR_PATH}" \
    SGCLS_CHECKPOINT="${SGCLS_CHECKPOINT}" \
    SGDET_CHECKPOINT="${SGDET_CHECKPOINT}" \
    bash "$0" > "${LOG_FILE}" 2>&1 &
  echo "$!" > "${SUITE_DIR}/suite.pid"
  echo "Started cross-task final test suite with PID $!"
  echo "Log: ${LOG_FILE}"
  exit 0
fi

if [[ ! -f "${SGCLS_CHECKPOINT}" ]]; then
  echo "SGCls checkpoint not found: ${SGCLS_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${SGDET_CHECKPOINT}" ]]; then
  echo "SGDet checkpoint not found: ${SGDET_CHECKPOINT}" >&2
  exit 1
fi

for task in sgcls sgdet; do
  for protocol in legacy pred; do
    for method in PPG RSGP; do
      if [[ "${task}" == "sgcls" ]]; then
        task_dir="sgcls"
      else
        task_dir="sgdet_rpcm_budget"
      fi
      method_slug="${method,,}"
      if [[ "${method}" == "RSGP" ]]; then
        method_slug="${method_slug}_${RSGP_ROLE_MODE}"
      fi
      existing="outputs/paper_submission/${task_dir}/test/${protocol}/${method_slug}/test_metrics.json"
      if [[ "${SKIP_EXISTING}" == "1" && -f "${existing}" ]]; then
        echo "Skip existing ${existing}"
      else
        if [[ "${task}" == "sgcls" ]]; then
          checkpoint="${SGCLS_CHECKPOINT}"
        else
          checkpoint="${SGDET_CHECKPOINT}"
        fi
        CHECKPOINT="${checkpoint}" \
        RUN_BACKGROUND=0 \
          bash "${ROOT_DIR}/scripts/_internal/eval_paper_cross_task.sh" \
          "${task}" "${method}" "${protocol}"
      fi
    done
  done
done

echo "[$(date '+%F %T')] cross-task final test suite complete"
