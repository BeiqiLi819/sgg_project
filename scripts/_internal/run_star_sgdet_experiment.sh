#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FILTER_METHOD=PPG
OUTPUT_DIR="${OUTPUT_DIR:-outputs/paper_submission/sgdet}"
CONFIG="${CONFIG:-configs/star_sgdet_obb_dual_la_train.py}"
SGDET_DETECTION_CACHE_ENABLED="${SGDET_DETECTION_CACHE_ENABLED:-1}"
SGDET_DETECTION_CACHE_DIR="${SGDET_DETECTION_CACHE_DIR:-star_sgdet_detection_cache}"
AUTO_TEST_AFTER_TRAIN="${AUTO_TEST_AFTER_TRAIN:-1}"
RUN_BACKGROUND="${RUN_BACKGROUND:-1}"
WORKFLOW_WORKER="${SGDET_WORKFLOW_WORKER:-0}"

if [[ "${SGDET_DETECTION_CACHE_ENABLED}" != "0" ]]; then
  cache_root="${SGDET_DETECTION_CACHE_DIR}"
  if [[ "${cache_root}" != /* ]]; then
    cache_root="${ROOT_DIR}/${cache_root}"
  fi
  for required_split in train val; do
    if [[ ! -f "${cache_root}/${required_split}/manifest.json" ]]; then
      echo "Missing SGDet ${required_split} cache manifest: ${cache_root}/${required_split}/manifest.json" >&2
      echo "Build it first with:" >&2
      echo "  SPLITS=${required_split} OVERWRITE=0 bash scripts/build_sgdet_detection_cache.sh" >&2
      exit 1
    fi
  done
fi

if [[ "${AUTO_TEST_AFTER_TRAIN}" != "0" && "${RUN_BACKGROUND}" != "0" && "${WORKFLOW_WORKER}" != "1" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  nohup env \
    SGDET_WORKFLOW_WORKER=1 \
    AUTO_TEST_AFTER_TRAIN=1 \
    RUN_BACKGROUND=0 \
    CONFIG="${CONFIG}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    SGDET_DETECTION_CACHE_ENABLED="${SGDET_DETECTION_CACHE_ENABLED}" \
    SGDET_DETECTION_CACHE_DIR="${SGDET_DETECTION_CACHE_DIR}" \
    bash "${SCRIPT_DIR}/run_star_sgdet_experiment.sh" > "${OUTPUT_DIR}/workflow.log" 2>&1 &
  echo "$!" > "${OUTPUT_DIR}/workflow.pid"
  echo "Started SGDet train-and-test workflow with PID $!"
  echo "Final test checkpoint: model_last.pth"
  echo "Workflow log: ${OUTPUT_DIR}/workflow.log"
  exit 0
fi

# Paper-main SGDet route: source-compatible frozen d1/d2 OBB detector and
# cache protocol + dual-view RCA + auxiliary LA. PPG/PPN/RSGP is mandatory.
# The detector is frozen and its full-resolution patch pass is prohibitively
# slow inside the training loop, so the sgdet launcher uses the prebuilt,
# read-only detection cache by default.  Set
# SGDET_DETECTION_CACHE_ENABLED=0 explicitly only for detector-path debugging.
#
# Common override examples:
#   RESUME=outputs/.../model_last.pth bash scripts/_internal/run_star_sgdet_experiment.sh
#   FILTER_METHOD=PPG bash scripts/_internal/run_star_sgdet_experiment.sh
#   PRINT_TRAIN_STEP_FREQ=10 bash scripts/_internal/run_star_sgdet_experiment.sh
#
# Before formal training, both train and test manifests must exist in the
# cache directory. The launcher defaults to REQUIRE_HIT=1, so a missing file
# or hash mismatch fails immediately instead of silently running the slow
# detector.
CONFIG="${CONFIG}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
FILTER_METHOD="${FILTER_METHOD}" \
VAL_SPLIT=val \
RSGP_MODE="${RSGP_MODE:-HYBRID}" \
RSGP_TOPK="${RSGP_TOPK:-10000}" \
RSGP_PPG_PROTECTED_TOPK="${RSGP_PPG_PROTECTED_TOPK:-9000}" \
SGDET_TRAIN_LABEL_SOURCE="${SGDET_TRAIN_LABEL_SOURCE:-matched_gt}" \
SGDET_FILTER_LABEL_SOURCE="${SGDET_FILTER_LABEL_SOURCE:-matched_gt}" \
ADD_GTBOX_TO_PROPOSAL_IN_TRAIN="${ADD_GTBOX_TO_PROPOSAL_IN_TRAIN:-0}" \
SGDET_DETECTION_CACHE_ENABLED="${SGDET_DETECTION_CACHE_ENABLED}" \
SGDET_DETECTION_CACHE_DIR="${SGDET_DETECTION_CACHE_DIR}" \
SGDET_DETECTION_CACHE_REQUIRE_HIT="${SGDET_DETECTION_CACHE_REQUIRE_HIT:-1}" \
SGDET_DETECTION_CACHE_HASH="${SGDET_DETECTION_CACHE_HASH:-}" \
INIT_RPCM="" \
RUN_BACKGROUND="${RUN_BACKGROUND}" \
bash "${SCRIPT_DIR}/run_star_experiment.sh"

if [[ "${AUTO_TEST_AFTER_TRAIN}" != "0" ]]; then
  LAST_CHECKPOINT="${OUTPUT_DIR}/model_last.pth"
  if [[ ! -f "${LAST_CHECKPOINT}" ]]; then
    echo "SGDet training completed but final checkpoint is missing: ${LAST_CHECKPOINT}" >&2
    exit 1
  fi
  if [[ "${SGDET_DETECTION_CACHE_ENABLED}" != "0" && ! -f "${cache_root}/test/manifest.json" ]]; then
    echo "SGDet final test cache is missing: ${cache_root}/test/manifest.json" >&2
    exit 1
  fi
  echo "SGDet training complete; evaluating model_last.pth on test."
  CONFIG="${CONFIG}" \
  CHECKPOINT="${LAST_CHECKPOINT}" \
  SPLIT=test \
  FILTER_METHOD=PPG \
  SGDET_FILTER_LABEL_SOURCE="${SGDET_FILTER_LABEL_SOURCE:-matched_gt}" \
  SGDET_DETECTION_CACHE_ENABLED="${SGDET_DETECTION_CACHE_ENABLED}" \
  SGDET_DETECTION_CACHE_DIR="${SGDET_DETECTION_CACHE_DIR}" \
  SGDET_DETECTION_CACHE_REQUIRE_HIT=1 \
  PROFILE_INFERENCE=1 \
  OUTPUT_DIR="${OUTPUT_DIR}/test/legacy/ppg" \
  RUN_BACKGROUND=0 \
  bash "${SCRIPT_DIR}/eval_star_sgdet.sh"
fi
