#!/usr/bin/env bash
set -euo pipefail

# One-command qualitative export for the final PredCls Dual+LA checkpoint.
# It runs selected STAR images under PPG and statistical RSGP, exports
# candidate/prediction tensors, then renders a satellite crop plus spatially
# aligned GT/PPG/RSGP scene graphs with identical node layouts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

WORKER="${QUALITATIVE_WORKER:-0}"
RUN_BACKGROUND="${RUN_BACKGROUND:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/star_predcls_obb_ablation_dual_la_train.py}"
CHECKPOINT="${CHECKPOINT:-outputs/paper_submission/predcls/dual_la/model_best_HR.pth}"
IMAGE_IDS="${IMAGE_IDS:-440,748,235}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/paper_submission/qualitative_figures}"
TOPK="${TOPK:-2000}"
PANEL_SIZE="${PANEL_SIZE:-700}"
RENDER_SCALE="${RENDER_SCALE:-2}"
PANEL_GUTTER="${PANEL_GUTTER:-24}"
MIN_NODE_SPACING="${MIN_NODE_SPACING:-58}"
EDGE_ROUTE_SPACING="${EDGE_ROUTE_SPACING:-16}"
CROP_FRACTION="${CROP_FRACTION:-0.10}"
CROP_OVERRIDES="${CROP_OVERRIDES:-440:1707,4771,2907,5971;235:6325,4650,6710,5035}"
FOCUS_LIMIT="${FOCUS_LIMIT:-6}"
UNMATCHED_LIMIT="${UNMATCHED_LIMIT:-8}"
SKIP_INFERENCE="${SKIP_INFERENCE:-0}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [[ "${RUN_BACKGROUND}" == "1" && "${WORKER}" != "1" ]]; then
  LOG_FILE="${OUTPUT_DIR}/render.log"
  PID_FILE="${OUTPUT_DIR}/render.pid"
  nohup env \
    QUALITATIVE_WORKER=1 \
    RUN_BACKGROUND=0 \
    PYTHON_BIN="${PYTHON_BIN}" \
    CONFIG="${CONFIG}" \
    CHECKPOINT="${CHECKPOINT}" \
    IMAGE_IDS="${IMAGE_IDS}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    TOPK="${TOPK}" \
    PANEL_SIZE="${PANEL_SIZE}" \
    RENDER_SCALE="${RENDER_SCALE}" \
    PANEL_GUTTER="${PANEL_GUTTER}" \
    MIN_NODE_SPACING="${MIN_NODE_SPACING}" \
    EDGE_ROUTE_SPACING="${EDGE_ROUTE_SPACING}" \
    CROP_FRACTION="${CROP_FRACTION}" \
    CROP_OVERRIDES="${CROP_OVERRIDES}" \
    FOCUS_LIMIT="${FOCUS_LIMIT}" \
    UNMATCHED_LIMIT="${UNMATCHED_LIMIT}" \
    SKIP_INFERENCE="${SKIP_INFERENCE}" \
    bash "$0" > "${LOG_FILE}" 2>&1 &
  echo "$!" > "${PID_FILE}"
  echo "Started qualitative export with PID $!"
  echo "Log: ${LOG_FILE}"
  echo "Figures: ${OUTPUT_DIR}/scene_graphs"
  exit 0
fi

run_inference() {
  local method="$1"
  local slug="${method,,}"
  local method_dir="${OUTPUT_DIR}/artifacts/${slug}"
  mkdir -p "${method_dir}"
  echo "[$(date '+%F %T')] Exporting ${method} artifacts for ${IMAGE_IDS}"
  env \
    TEST_BATCH_SIZE=1 \
    VAL_BATCH_SIZE=1 \
    NUM_WORKERS=0 \
    RSGP_MODE=HYBRID \
    RSGP_TOPK=10000 \
    RSGP_PPG_PROTECTED_TOPK=9000 \
    RSGP_ROLE_MODE=statistical \
    RSGP_STRUCTURAL_PRIOR_PATH=pretrained/rsgp_structural_prior.json \
    "${PYTHON_BIN}" tools/eval_once.py \
      --config "${CONFIG}" \
      --checkpoint "${CHECKPOINT}" \
      --checkpoint-load-mode full \
      --split test \
      --device cuda \
      --filter-method "${method}" \
      --image-ids "${IMAGE_IDS}" \
      --artifact-output-dir "${method_dir}" \
      --output "${method_dir}/metrics.json"
}

if [[ "${SKIP_INFERENCE}" != "1" ]]; then
  run_inference PPG
  run_inference RSGP
else
  echo "Skipping inference and reusing ${OUTPUT_DIR}/artifacts/{ppg,rsgp}"
fi

echo "[$(date '+%F %T')] Rendering spatially aligned scene graphs"
"${PYTHON_BIN}" tools/render_spatial_scene_graph_cases.py \
  --ppg-dir "${OUTPUT_DIR}/artifacts/ppg" \
  --rsgp-dir "${OUTPUT_DIR}/artifacts/rsgp" \
  --image-ids "${IMAGE_IDS}" \
  --output-dir "${OUTPUT_DIR}/scene_graphs" \
  --topk "${TOPK}" \
  --panel-size "${PANEL_SIZE}" \
  --render-scale "${RENDER_SCALE}" \
  --gutter "${PANEL_GUTTER}" \
  --min-node-spacing "${MIN_NODE_SPACING}" \
  --edge-route-spacing "${EDGE_ROUTE_SPACING}" \
  --crop-fraction "${CROP_FRACTION}" \
  --crop-overrides "${CROP_OVERRIDES}" \
  --focus-limit "${FOCUS_LIMIT}" \
  --unmatched-limit "${UNMATCHED_LIMIT}"

echo "[$(date '+%F %T')] Qualitative export complete"
echo "Figures: ${OUTPUT_DIR}/scene_graphs"
