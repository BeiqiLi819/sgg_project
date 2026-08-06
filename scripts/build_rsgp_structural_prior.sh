#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-sgg}"
CONFIG="${CONFIG:-configs/star_predcls_obb_ablation_dual_la_train.py}"
OUTPUT="${OUTPUT:-pretrained/rsgp_structural_prior.json}"
RARITY_BETA="${RARITY_BETA:-0.5}"
CONTAINMENT_BLOCK_SIZE="${CONTAINMENT_BLOCK_SIZE:-64}"

cd "${ROOT_DIR}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

python tools/build_rsgp_structural_prior.py \
  --config "${CONFIG}" \
  --output "${OUTPUT}" \
  --rarity-beta "${RARITY_BETA}" \
  --containment-block-size "${CONTAINMENT_BLOCK_SIZE}"
