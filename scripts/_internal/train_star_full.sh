#!/usr/bin/env bash
set -euo pipefail

# Shared backend for the three public Full training launchers. Training uses
# PPG for validation/checkpoint selection; the frozen best/final checkpoint is
# then evaluated with the public statistical RSGP Full test launcher.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TASK="${1:-}"
RUN_BACKGROUND="${RUN_BACKGROUND:-1}"
WORKER="${FULL_TRAIN_WORKER:-0}"
AUTO_TEST_AFTER_TRAIN="${AUTO_TEST_AFTER_TRAIN:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"

case "${TASK}" in
  predcls)
    CONFIG="${CONFIG:-configs/star_predcls_obb_full.py}"
    OUTPUT_DIR="${OUTPUT_DIR:-outputs/star_predcls_obb_full}"
    FINAL_CHECKPOINT_NAME=model_best_HR.pth
    TEST_LAUNCHER="${ROOT_DIR}/scripts/test_star_predcls_full.sh"
    TRAIN_BACKEND="${SCRIPT_DIR}/run_star_experiment.sh"
    ;;
  sgcls)
    CONFIG="${CONFIG:-configs/star_sgcls_obb_full.py}"
    OUTPUT_DIR="${OUTPUT_DIR:-outputs/star_sgcls_obb_full}"
    FINAL_CHECKPOINT_NAME=model_best_HR.pth
    TEST_LAUNCHER="${ROOT_DIR}/scripts/test_star_sgcls_full.sh"
    TRAIN_BACKEND="${SCRIPT_DIR}/run_star_experiment.sh"
    ;;
  sgdet)
    CONFIG="${CONFIG:-configs/star_sgdet_obb_full.py}"
    OUTPUT_DIR="${OUTPUT_DIR:-outputs/star_sgdet_obb_full}"
    FINAL_CHECKPOINT_NAME=model_last.pth
    TEST_LAUNCHER="${ROOT_DIR}/scripts/test_star_sgdet_full.sh"
    TRAIN_BACKEND="${SCRIPT_DIR}/run_star_sgdet_experiment.sh"
    ;;
  *)
    echo "Usage: bash scripts/_internal/train_star_full.sh predcls|sgcls|sgdet" >&2
    exit 2
    ;;
esac

cd "${ROOT_DIR}"
RESUME="${RESUME:-}"
if [[ "${AUTO_RESUME}" != "0" && -z "${RESUME}" && -f "${OUTPUT_DIR}/model_last.pth" ]]; then
  RESUME="${OUTPUT_DIR}/model_last.pth"
fi

if [[ "${RUN_BACKGROUND}" != "0" && "${WORKER}" != "1" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  nohup env \
    FULL_TRAIN_WORKER=1 \
    RUN_BACKGROUND=0 \
    AUTO_TEST_AFTER_TRAIN="${AUTO_TEST_AFTER_TRAIN}" \
    AUTO_RESUME="${AUTO_RESUME}" \
    RESUME="${RESUME}" \
    CONFIG="${CONFIG}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    CONDA_ENV="${CONDA_ENV:-sgg}" \
    SGDET_DETECTION_CACHE_ENABLED="${SGDET_DETECTION_CACHE_ENABLED:-1}" \
    SGDET_DETECTION_CACHE_DIR="${SGDET_DETECTION_CACHE_DIR:-star_sgdet_detection_cache}" \
    bash "$0" "${TASK}" > "${OUTPUT_DIR}/workflow.log" 2>&1 &
  echo "$!" > "${OUTPUT_DIR}/workflow.pid"
  echo "Started STAR OBB Full ${TASK} training with PID $!"
  echo "Workflow log: ${OUTPUT_DIR}/workflow.log"
  exit 0
fi

echo "Training STAR OBB Full ${TASK}"
echo "Config: ${CONFIG}"
echo "Initialization: detector-only pretrained/OBB_swin_L_OBD.pth"
echo "Checkpoint-selection filter: PPG"
echo "Resume: ${RESUME:-<none>}"

train_env=(
  CONFIG="${CONFIG}"
  OUTPUT_DIR="${OUTPUT_DIR}"
  FILTER_METHOD=PPG
  VAL_SPLIT=val
  INIT_RPCM=
  RESUME="${RESUME}"
  RUN_BACKGROUND=0
  CONDA_ENV="${CONDA_ENV:-sgg}"
)
if [[ "${TASK}" == "sgdet" ]]; then
  train_env+=(
    AUTO_TEST_AFTER_TRAIN=0
    SGDET_DETECTION_CACHE_ENABLED="${SGDET_DETECTION_CACHE_ENABLED:-1}"
    SGDET_DETECTION_CACHE_DIR="${SGDET_DETECTION_CACHE_DIR:-star_sgdet_detection_cache}"
  )
fi
env "${train_env[@]}" bash "${TRAIN_BACKEND}"

FINAL_CHECKPOINT="${OUTPUT_DIR}/${FINAL_CHECKPOINT_NAME}"
if [[ ! -f "${FINAL_CHECKPOINT}" ]]; then
  echo "Training finished but expected checkpoint is missing: ${FINAL_CHECKPOINT}" >&2
  exit 1
fi

if [[ "${AUTO_TEST_AFTER_TRAIN}" != "0" ]]; then
  echo "Training complete; running frozen Full RSGP test."
  CHECKPOINT="${FINAL_CHECKPOINT}" \
  OUTPUT_DIR="${OUTPUT_DIR}/test/rsgp" \
  RUN_BACKGROUND=0 \
  CONDA_ENV="${CONDA_ENV:-sgg}" \
  SGDET_DETECTION_CACHE_ENABLED="${SGDET_DETECTION_CACHE_ENABLED:-1}" \
  SGDET_DETECTION_CACHE_DIR="${SGDET_DETECTION_CACHE_DIR:-star_sgdet_detection_cache}" \
  bash "${TEST_LAUNCHER}"
fi
