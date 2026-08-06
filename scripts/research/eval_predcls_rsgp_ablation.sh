#!/usr/bin/env bash
set -euo pipefail

# Inference-only test-set component ablation using the validation-frozen
# relation checkpoint and RSGP hyperparameters. Run one case at a time to
# avoid concurrent GPU evaluations; never use these results to retune RSGP.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CASE="${1:-}"

FILTER_METHOD=RSGP
RSGP_MODE=HYBRID
RSGP_USE_PPN_COMPLETION=1
RSGP_USE_GEOMETRY=1
RSGP_USE_CONTEXT_ROLE=1
RSGP_USE_ALIGNMENT_ROLE=1
RSGP_USE_CONNECTIVITY_ROLE=1
RSGP_USE_RARITY_PRIOR=1
RSGP_USE_DEGREE_SCORE=1
RSGP_ENFORCE_DEGREE_CAP=1
RSGP_ENFORCE_LABEL_QUOTA=1
RSGP_RS_POOL_TOPK=12000
RSGP_ROLE_MODE="${RSGP_ROLE_MODE:-statistical}"
RSGP_STRUCTURAL_PRIOR_PATH="${RSGP_STRUCTURAL_PRIOR_PATH:-pretrained/rsgp_structural_prior.json}"

case "${CASE^^}" in
  PPG)
    FILTER_METHOD=PPG
    NAME="ppg"
    ;;
  PPN)
    FILTER_METHOD=PPN
    NAME="ppn"
    ;;
  FULL|RSGP)
    NAME="rsgp_full"
    ;;
  LEGACY_MANUAL)
    RSGP_ROLE_MODE=legacy_manual
    NAME="rsgp_legacy_manual"
    ;;
  NO_PPN)
    RSGP_USE_PPN_COMPLETION=0
    NAME="rsgp_no_ppn_completion"
    ;;
  NO_RS)
    RSGP_RS_POOL_TOPK=0
    RSGP_USE_GEOMETRY=0
    RSGP_USE_CONTEXT_ROLE=0
    RSGP_USE_ALIGNMENT_ROLE=0
    RSGP_USE_CONNECTIVITY_ROLE=0
    RSGP_USE_RARITY_PRIOR=0
    NAME="rsgp_no_rs_priors"
    ;;
  NO_STRUCTURE|NO_ROLES)
    RSGP_USE_CONTEXT_ROLE=0
    RSGP_USE_ALIGNMENT_ROLE=0
    RSGP_USE_CONNECTIVITY_ROLE=0
    NAME="rsgp_no_structural_roles"
    ;;
  NO_GEOMETRY|NO_GEOM)
    RSGP_USE_GEOMETRY=0
    NAME="rsgp_no_geometry"
    ;;
  NO_DEGREE)
    RSGP_USE_DEGREE_SCORE=0
    RSGP_ENFORCE_DEGREE_CAP=0
    NAME="rsgp_no_degree_control"
    ;;
  NO_QUOTA)
    RSGP_ENFORCE_LABEL_QUOTA=0
    NAME="rsgp_no_label_pair_quota"
    ;;
  NO_RARITY|NO_HARD_PRIOR|NO_TAIL)
    RSGP_USE_RARITY_PRIOR=0
    NAME="rsgp_no_rarity_support"
    ;;
  *)
    echo "Usage: bash scripts/research/eval_predcls_rsgp_ablation.sh \\" >&2
    echo "  PPG|PPN|FULL|LEGACY_MANUAL|NO_PPN|NO_RS|NO_STRUCTURE|NO_GEOMETRY|NO_DEGREE|NO_QUOTA|NO_RARITY" >&2
    exit 2
    ;;
esac

cd "${ROOT_DIR}"
CHECKPOINT="${CHECKPOINT:-outputs/paper_submission/predcls/dual_la/model_best_HR.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/paper_submission/predcls/dual_la/rsgp_component_test}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${NAME}}"
SPLIT="${SPLIT:-test}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

CHECKPOINT="${CHECKPOINT}" \
CONFIG=configs/star_predcls_obb_ablation_dual_la_train.py \
SPLIT="${SPLIT}" \
FILTER_METHOD="${FILTER_METHOD}" \
RSGP_MODE="${RSGP_MODE}" \
RSGP_PPG_PROTECTED_TOPK="${RSGP_PPG_PROTECTED_TOPK:-9000}" \
RSGP_ROLE_MODE="${RSGP_ROLE_MODE}" \
RSGP_STRUCTURAL_PRIOR_PATH="${RSGP_STRUCTURAL_PRIOR_PATH}" \
RSGP_USE_PPN_COMPLETION="${RSGP_USE_PPN_COMPLETION}" \
RSGP_USE_GEOMETRY="${RSGP_USE_GEOMETRY}" \
RSGP_USE_CONTEXT_ROLE="${RSGP_USE_CONTEXT_ROLE}" \
RSGP_USE_ALIGNMENT_ROLE="${RSGP_USE_ALIGNMENT_ROLE}" \
RSGP_USE_CONNECTIVITY_ROLE="${RSGP_USE_CONNECTIVITY_ROLE}" \
RSGP_USE_RARITY_PRIOR="${RSGP_USE_RARITY_PRIOR}" \
RSGP_USE_DEGREE_SCORE="${RSGP_USE_DEGREE_SCORE}" \
RSGP_ENFORCE_DEGREE_CAP="${RSGP_ENFORCE_DEGREE_CAP}" \
RSGP_ENFORCE_LABEL_QUOTA="${RSGP_ENFORCE_LABEL_QUOTA}" \
RSGP_RS_POOL_TOPK="${RSGP_RS_POOL_TOPK}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash "${ROOT_DIR}/scripts/_internal/eval_star_predcls.sh"
