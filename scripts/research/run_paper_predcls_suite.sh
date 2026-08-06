#!/usr/bin/env bash
set -euo pipefail

# Queue the controlled Base(Unified)/D/DL train-and-test runs in one background
# process. Every successful training is immediately evaluated on test with
# its validation-selected best checkpoint and PPG. Completed rows are
# identified by their final test_metrics.json and skipped on later launches.
#
# Examples:
#   bash scripts/research/run_paper_predcls_suite.sh
#   CASES=BASE,D bash scripts/research/run_paper_predcls_suite.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CASES="${CASES:-BASE,D,DL}"
SUITE_DIR="${SUITE_DIR:-outputs/paper_submission/predcls}"
AUTO_RESUME="${AUTO_RESUME:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
WORKER="${PAPER_SUITE_WORKER:-0}"

cd "${ROOT_DIR}"
mkdir -p "${SUITE_DIR}"

if [[ "${WORKER}" != "1" ]]; then
  LOG_FILE="${SUITE_DIR}/suite.log"
  PID_FILE="${SUITE_DIR}/suite.pid"
  nohup env \
    PAPER_SUITE_WORKER=1 \
    CASES="${CASES}" \
    SUITE_DIR="${SUITE_DIR}" \
    AUTO_RESUME="${AUTO_RESUME}" \
    SKIP_COMPLETED="${SKIP_COMPLETED}" \
    bash "$0" >> "${LOG_FILE}" 2>&1 &
  echo "$!" > "${PID_FILE}"
  echo "Started controlled PredCls suite with PID $!"
  echo "Rows: ${CASES}"
  echo "Log: ${LOG_FILE}"
  exit 0
fi

IFS=',' read -r -a case_values <<< "${CASES}"

echo
echo "================================================================"
echo "[$(date '+%F %T')] controlled PredCls suite session"
echo "Rows: ${CASES}; auto_resume=${AUTO_RESUME}; skip_completed=${SKIP_COMPLETED}"
echo "================================================================"

for row in "${case_values[@]}"; do
  case "${row^^}" in
    BASE|U|UNIFIED) slug="unified" ;;
    SOURCE|SOURCE_BASE|SGG_TOOLKIT) slug="base_original" ;;
    D|DUAL) slug="dual" ;;
    D6850|DUAL_6850|6850|REPRO|MV) slug="dual_6850_repro" ;;
    DL|DUAL_LA|LA|MAIN|C) slug="dual_la" ;;
    *)
      echo "Unknown controlled PredCls row: ${row}" >&2
      exit 2
      ;;
  esac
  row_output="${SUITE_DIR}/${slug}"
  completed_metrics="${row_output}/test/ppg/test_metrics.json"
  if [[ "${SKIP_COMPLETED}" != "0" && -s "${completed_metrics}" ]]; then
    echo "[$(date '+%F %T')] skip completed row=${row}: ${completed_metrics}"
    continue
  fi
  if [[ "${AUTO_RESUME}" != "0" && -f "${row_output}/model_last.pth" ]]; then
    echo "[$(date '+%F %T')] continue row=${row} from ${row_output}/model_last.pth"
  else
    echo "[$(date '+%F %T')] start row=${row} from detector-only initialization"
  fi
  OUTPUT_DIR="${row_output}" \
  AUTO_RESUME="${AUTO_RESUME}" \
  RESUME="" \
  RUN_BACKGROUND=0 \
  bash "${SCRIPT_DIR}/run_predcls_minimal_ablation.sh" "${row}"
  echo "[$(date '+%F %T')] done row=${row}"
done

echo "[$(date '+%F %T')] controlled PredCls suite complete"
