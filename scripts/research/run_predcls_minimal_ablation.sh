#!/usr/bin/env bash
set -euo pipefail

# Controlled PredCls training for the paper.
#
# Rows Base (Unified), D and DL form the paper sequence. D6850 is a historical
# protocol-reproduction experiment that replaces the inconclusive MV trial.
# DL_ORIGINAL_HEAD is an
# isolated decision experiment and is not inserted into that sequence unless
# its result justifies rebuilding the controlled rows around the source head.
# RSGP is inference-only.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CASE="${1:-}"

case "${CASE^^}" in
  SOURCE|SOURCE_BASE|SGG_TOOLKIT)
    CONFIG="configs/star_predcls_obb_ablation_sgg_toolkit_base_train.py"
    SLUG="base_original"
    LABEL="Complete original SGG-ToolKit RPCM audit"
    ;;
  BASE|U|UNIFIED)
    CONFIG="configs/star_predcls_obb_ablation_unified_rca_train.py"
    SLUG="unified"
    LABEL="Paper Base: unified adjacency + current RCA GNN"
    ;;
  D|DUAL)
    CONFIG="configs/star_predcls_obb_ablation_dual_rca_train.py"
    SLUG="dual"
    LABEL="Dual-view adjacency + current RCA GNN"
    ;;
  D6850|DUAL_6850|6850|REPRO|MV)
    CONFIG="configs/star_predcls_obb_ablation_dual_6850_repro_train.py"
    SLUG="dual_6850_repro"
    LABEL="Historical 6850 Dual protocol reproduction"
    DEFAULT_AUTO_RESUME=0
    # Python must inherit this before interpreter startup for hash-order
    # determinism. The config seeds Python/NumPy/Torch/CUDA as well.
    export PYTHONHASHSEED="${PYTHONHASHSEED:-1029}"
    ;;
  DL|DUAL_LA|LA|MAIN|C)
    CONFIG="configs/star_predcls_obb_ablation_dual_la_train.py"
    SLUG="dual_la"
    LABEL="Dual-view RCA + auxiliary logit adjustment"
    ;;
  DL_ORIGINAL_HEAD|DUAL_LA_ORIGINAL_HEAD|ORIGINAL_HEAD|DLOH)
    CONFIG="configs/star_predcls_obb_ablation_dual_la_original_head_train.py"
    SLUG="dual_la_original_head"
    LABEL="Dual-view RCA + LA + original SGG-ToolKit gated prototype head"
    ;;
  FULL|DL_RSGP|RSGP)
    echo "RSGP has no additional training parameters." >&2
    echo "Train DL, select RSGP on val, then evaluate the same checkpoint on test:" >&2
    echo "  bash scripts/research/run_predcls_minimal_ablation.sh DL" >&2
    echo "  bash scripts/research/select_predcls_rsgp_on_val.sh" >&2
    echo "  FILTER_METHOD=RSGP bash scripts/_internal/eval_predcls_minimal_ablation.sh DL" >&2
    exit 2
    ;;
  *)
    echo "Usage: bash scripts/research/run_predcls_minimal_ablation.sh BASE|D|DL|SOURCE|D6850|DL_ORIGINAL_HEAD" >&2
    echo "  BASE (alias U/UNIFIED): unified adjacency + current RCA GNN" >&2
    echo "  SOURCE: complete original SGG-ToolKit GNN and prototype classifier audit" >&2
    echo "  D : dual-view adjacency + the same RCA GNN" >&2
    echo "  D6850: Dual SS/OO + exact 6850 head and 20k-iteration protocol" >&2
    echo "         (MV remains a compatibility alias for D6850)" >&2
    echo "  DL: dual-view RCA + auxiliary LA" >&2
    echo "  DL_ORIGINAL_HEAD: current Dual+LA front end + source gated prototype head" >&2
    exit 2
    ;;
esac

cd "${ROOT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/paper_submission/predcls/${SLUG}}"
AUTO_TEST_AFTER_TRAIN="${AUTO_TEST_AFTER_TRAIN:-1}"
AUTO_RESUME="${AUTO_RESUME:-${DEFAULT_AUTO_RESUME:-1}}"
RESUME="${RESUME:-}"
RESET_SOLVER_ON_RESUME="${RESET_SOLVER_ON_RESUME:-0}"
RESET_SCHEDULER_ON_RESUME="${RESET_SCHEDULER_ON_RESUME:-0}"
RUN_BACKGROUND="${RUN_BACKGROUND:-1}"
WORKFLOW_WORKER="${PREDCLS_WORKFLOW_WORKER:-0}"

if [[ "${AUTO_RESUME}" != "0" && -z "${RESUME}" && -f "${OUTPUT_DIR}/model_last.pth" ]]; then
  RESUME="${OUTPUT_DIR}/model_last.pth"
  echo "Auto-resume checkpoint detected: ${RESUME}"
fi

if [[ "${AUTO_TEST_AFTER_TRAIN}" != "0" && "${RUN_BACKGROUND}" != "0" && "${WORKFLOW_WORKER}" != "1" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  WORKFLOW_LOG="${OUTPUT_DIR}/workflow.log"
  WORKFLOW_PID="${OUTPUT_DIR}/workflow.pid"
  nohup env \
    PREDCLS_WORKFLOW_WORKER=1 \
    AUTO_TEST_AFTER_TRAIN=1 \
    AUTO_RESUME="${AUTO_RESUME}" \
    RESUME="${RESUME}" \
    RESET_SOLVER_ON_RESUME="${RESET_SOLVER_ON_RESUME}" \
    RESET_SCHEDULER_ON_RESUME="${RESET_SCHEDULER_ON_RESUME}" \
    RUN_BACKGROUND=0 \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    bash "${SCRIPT_DIR}/run_predcls_minimal_ablation.sh" "${CASE}" > "${WORKFLOW_LOG}" 2>&1 &
  echo "$!" > "${WORKFLOW_PID}"
  echo "Started PredCls train-and-test workflow with PID $!"
  echo "Row: ${SLUG}"
  echo "Workflow log: ${WORKFLOW_LOG}"
  exit 0
fi

echo "Launching controlled PredCls row: ${LABEL}"
echo "Checkpoint-selection split: val"
echo "Checkpoint-selection filter: PPG (fixed top-10000 protocol)"
if [[ "${SLUG}" == "dual_6850_repro" ]]; then
  echo "Schedule: 20,000 iter; LR steps 13,000/18,000; validation every 200 iter from 14,000"
  echo "Early stop: disabled (historical protocol)"
  echo "Seed: 1029 (Python/NumPy/Torch/CUDA)"
  echo "Input: short edge=600, long edge<=1000, aspect-ratio grouped batches"
  echo "cuDNN: disabled by default for bounded memory (set DISABLE_CUDNN=0 to restore source runtime)"
  echo "Auto-resume: ${AUTO_RESUME} (fresh start is the reproduction default)"
else
  echo "Early stop: validation HMR@2000, patience=10 validation events (configurable)"
fi
echo "Initialization: detector-only pretrained/OBB_swin_L_OBD.pth"
echo "Resume: ${RESUME:-<none>}"
echo "Current config will refresh run_manifest.json and source_config.py"
echo "Output: ${OUTPUT_DIR}"

CONFIG="${CONFIG}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
INIT_RPCM="" \
RESUME="${RESUME}" \
RESET_SOLVER_ON_RESUME="${RESET_SOLVER_ON_RESUME}" \
RESET_SCHEDULER_ON_RESUME="${RESET_SCHEDULER_ON_RESUME}" \
FILTER_METHOD=PPG \
VAL_SPLIT=val \
RSGP_TOPK=10000 \
RUN_BACKGROUND="${RUN_BACKGROUND}" \
bash "${ROOT_DIR}/scripts/_internal/run_star_experiment.sh"

if [[ "${AUTO_TEST_AFTER_TRAIN}" != "0" ]]; then
  BEST_CHECKPOINT="${OUTPUT_DIR}/model_best_HR.pth"
  if [[ ! -f "${BEST_CHECKPOINT}" ]]; then
    # A continuation in a new output directory inherits all-history
    # best_metrics from RESUME. If no later validation surpasses that score,
    # no new model_best_HR.pth is written; the resume checkpoint remains the
    # mathematically correct checkpoint to test.
    if [[ -n "${RESUME}" && -f "${RESUME}" ]]; then
      BEST_CHECKPOINT="${RESUME}"
      echo "No resumed epoch surpassed the inherited validation-HMR best."
      echo "Using the resume checkpoint for the final test: ${BEST_CHECKPOINT}"
    else
      echo "Training completed but best checkpoint is missing: ${BEST_CHECKPOINT}" >&2
      exit 1
    fi
  fi
  echo "Training complete; evaluating best validation-HMR checkpoint on test."
  CHECKPOINT="${BEST_CHECKPOINT}" \
  SPLIT=test \
  FILTER_METHOD=PPG \
  OUTPUT_DIR="${OUTPUT_DIR}/test/ppg" \
  PROFILE_INFERENCE=1 \
  RUN_BACKGROUND=0 \
  bash "${ROOT_DIR}/scripts/_internal/eval_predcls_minimal_ablation.sh" "${CASE}"
fi
