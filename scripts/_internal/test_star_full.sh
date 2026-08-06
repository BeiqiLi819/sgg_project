#!/usr/bin/env bash
set -euo pipefail

# Shared backend for the three public Full test launchers.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TASK="${1:-}"

case "${TASK}" in
  predcls)
    CONFIG="${CONFIG:-configs/star_predcls_obb_full.py}"
    CHECKPOINT="${CHECKPOINT:-pretrained/full/STAR_OBB_Full_PredCls.pth}"
    OUTPUT_DIR="${OUTPUT_DIR:-outputs/star_predcls_obb_full_eval}"
    EVAL_BACKEND="${SCRIPT_DIR}/eval_star_predcls.sh"
    ;;
  sgcls)
    CONFIG="${CONFIG:-configs/star_sgcls_obb_full.py}"
    CHECKPOINT="${CHECKPOINT:-pretrained/full/STAR_OBB_Full_SGCls.pth}"
    OUTPUT_DIR="${OUTPUT_DIR:-outputs/star_sgcls_obb_full_eval}"
    EVAL_BACKEND="${SCRIPT_DIR}/eval_star_sgcls.sh"
    ;;
  sgdet)
    CONFIG="${CONFIG:-configs/star_sgdet_obb_full.py}"
    CHECKPOINT="${CHECKPOINT:-pretrained/full/STAR_OBB_Full_SGDet.pth}"
    OUTPUT_DIR="${OUTPUT_DIR:-outputs/star_sgdet_obb_full_eval}"
    EVAL_BACKEND="${SCRIPT_DIR}/eval_star_sgdet.sh"
    ;;
  *)
    echo "Usage: bash scripts/_internal/test_star_full.sh predcls|sgcls|sgdet" >&2
    exit 2
    ;;
esac

cd "${ROOT_DIR}"

required_assets=(
  "${CHECKPOINT}"
  "pretrained/OBB_swin_L_OBD.pth"
  "pretrained/STAR_OBB.pth"
  "pretrained/PPN_OBB.pth"
  "pretrained/rsgp_structural_prior.json"
)
for asset in "${required_assets[@]}"; do
  if [[ ! -f "${asset}" ]]; then
    echo "Missing required Full asset: ${asset}" >&2
    echo "See pretrained/full/README.md for the release layout." >&2
    exit 1
  fi
done

common_env=(
  CONFIG="${CONFIG}"
  CHECKPOINT="${CHECKPOINT}"
  CHECKPOINT_LOAD_MODE=full
  SPLIT="${SPLIT:-test}"
  FILTER_METHOD=RSGP
  RSGP_MODE=HYBRID
  RSGP_TOPK=10000
  RSGP_PPG_PROTECTED_TOPK=9000
  RSGP_ROLE_MODE=statistical
  RSGP_STRUCTURAL_PRIOR_PATH=pretrained/rsgp_structural_prior.json
  OUTPUT_DIR="${OUTPUT_DIR}"
  RUN_BACKGROUND="${RUN_BACKGROUND:-1}"
  CONDA_ENV="${CONDA_ENV:-sgg}"
)

if [[ "${TASK}" == "sgcls" ]]; then
  common_env+=(SGCLS_FILTER_LABEL_SOURCE="${SGCLS_FILTER_LABEL_SOURCE:-gt}")
elif [[ "${TASK}" == "sgdet" ]]; then
  common_env+=(
    SGDET_FILTER_LABEL_SOURCE="${SGDET_FILTER_LABEL_SOURCE:-matched_gt}"
    SGDET_DETECTION_CACHE_ENABLED="${SGDET_DETECTION_CACHE_ENABLED:-1}"
    SGDET_DETECTION_CACHE_DIR="${SGDET_DETECTION_CACHE_DIR:-star_sgdet_detection_cache}"
    SGDET_DETECTION_CACHE_REQUIRE_HIT="${SGDET_DETECTION_CACHE_REQUIRE_HIT:-1}"
  )
fi

echo "Testing STAR OBB Full ${TASK}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Filter: statistical RSGP Hybrid 9000/1000"
env "${common_env[@]}" bash "${EVAL_BACKEND}"
