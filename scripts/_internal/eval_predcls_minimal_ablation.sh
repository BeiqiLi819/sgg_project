#!/usr/bin/env bash
set -euo pipefail

# Final/validation evaluation for one controlled PredCls checkpoint.
# RSGP is inference-only: DL+RSGP must point to the exact DL checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CASE="${1:-}"
FILTER_METHOD="${FILTER_METHOD:-PPG}"
FILTER_METHOD="${FILTER_METHOD^^}"
SPLIT="${SPLIT:-test}"

case "${CASE^^}" in
  SOURCE|SOURCE_BASE|SGG_TOOLKIT)
    CONFIG="configs/star_predcls_obb_ablation_sgg_toolkit_base_train.py"
    SLUG="base_original"
    ;;
  BASE|U|UNIFIED)
    CONFIG="configs/star_predcls_obb_ablation_unified_rca_train.py"
    SLUG="unified"
    ;;
  D|DUAL)
    CONFIG="configs/star_predcls_obb_ablation_dual_rca_train.py"
    SLUG="dual"
    ;;
  D6850|DUAL_6850|6850|REPRO|MV)
    CONFIG="configs/star_predcls_obb_ablation_dual_6850_repro_train.py"
    SLUG="dual_6850_repro"
    ;;
  DL|DUAL_LA|LA|MAIN|C|FULL|DL_RSGP)
    CONFIG="configs/star_predcls_obb_ablation_dual_la_train.py"
    SLUG="dual_la"
    ;;
  DL_ORIGINAL_HEAD|DUAL_LA_ORIGINAL_HEAD|ORIGINAL_HEAD|DLOH)
    CONFIG="configs/star_predcls_obb_ablation_dual_la_original_head_train.py"
    SLUG="dual_la_original_head"
    ;;
  *)
    echo "Usage: FILTER_METHOD=PPG|PPN|RSGP \\" >&2
    echo "       bash scripts/_internal/eval_predcls_minimal_ablation.sh BASE|D|DL|SOURCE|D6850|DL_ORIGINAL_HEAD" >&2
    exit 2
    ;;
esac

case "${FILTER_METHOD}" in
  PPG|PPN|RSGP) ;;
  *)
    echo "FILTER_METHOD must be PPG, PPN, or RSGP; got ${FILTER_METHOD}" >&2
    exit 2
    ;;
esac

cd "${ROOT_DIR}"
RUN_DIR="outputs/paper_submission/predcls/${SLUG}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/model_best_HR.pth}"
RSGP_ROLE_MODE="${RSGP_ROLE_MODE:-statistical}"
FILTER_SLUG="${FILTER_METHOD,,}"
if [[ "${FILTER_METHOD}" == "RSGP" ]]; then
  FILTER_SLUG="${FILTER_SLUG}_${RSGP_ROLE_MODE}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/${SPLIT}/${FILTER_SLUG}}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

echo "Evaluating controlled PredCls row=${SLUG}"
echo "Split=${SPLIT}, filter=${FILTER_METHOD}, fixed pair budget=10000"
echo "Checkpoint=${CHECKPOINT}"

CONFIG="${CONFIG}" \
CHECKPOINT="${CHECKPOINT}" \
CHECKPOINT_LOAD_MODE=full \
SPLIT="${SPLIT}" \
FILTER_METHOD="${FILTER_METHOD}" \
PPG_TOPK=10000 \
PPN_TOPK=10000 \
RSGP_TOPK=10000 \
RSGP_MODE="${RSGP_MODE:-HYBRID}" \
RSGP_PPG_PROTECTED_TOPK="${RSGP_PPG_PROTECTED_TOPK:-9000}" \
RSGP_ROLE_MODE="${RSGP_ROLE_MODE}" \
RSGP_STRUCTURAL_PRIOR_PATH="${RSGP_STRUCTURAL_PRIOR_PATH:-pretrained/rsgp_structural_prior.json}" \
PROFILE_INFERENCE="${PROFILE_INFERENCE:-1}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
RUN_BACKGROUND="${RUN_BACKGROUND:-1}" \
bash "${SCRIPT_DIR}/eval_star_predcls.sh"
