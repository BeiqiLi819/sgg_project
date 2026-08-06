#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-sgg}"
CONFIG="${CONFIG:-configs/star_predcls_obb_ablation_dual_la_train.py}"
CHECKPOINT="${CHECKPOINT:-outputs/paper_submission/predcls/dual_la/model_best_HR.pth}"
CHECKPOINT_LOAD_MODE="${CHECKPOINT_LOAD_MODE:-full}"
SPLIT="${SPLIT:-val}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/paper_submission/predcls/dual_la/rsgp_val_selection}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
PPG_MODEL_PATH_OBB="${PPG_MODEL_PATH_OBB:-pretrained/STAR_OBB.pth}"
PPN_MODEL_PATH="${PPN_MODEL_PATH:-pretrained/PPN_OBB.pth}"
RSGP_ROLE_MODE="${RSGP_ROLE_MODE:-statistical}"
RSGP_STRUCTURAL_PRIOR_PATH="${RSGP_STRUCTURAL_PRIOR_PATH:-pretrained/rsgp_structural_prior.json}"

cd "${ROOT_DIR}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

run_case() {
  local name="$1"
  local filter_method="$2"
  local rsgp_mode="${3:-HYBRID}"
  local protected_topk="${4:-9000}"
  local role_mode="${5:-${RSGP_ROLE_MODE}}"

  local out_dir="${OUTPUT_ROOT}/${name}"
  mkdir -p "${out_dir}"

  echo "Running ${name}: filter=${filter_method}, rsgp_mode=${rsgp_mode}, protected=${protected_topk}, roles=${role_mode}"
  env \
    FILTER_METHOD="${filter_method}" \
    TEST_BATCH_SIZE="${TEST_BATCH_SIZE}" \
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE}" \
    PPG_MODEL_PATH_OBB="${PPG_MODEL_PATH_OBB}" \
    PPN_MODEL_PATH="${PPN_MODEL_PATH}" \
    PPG_TOPK=10000 \
    PPN_TOPK=10000 \
    RSGP_TOPK=10000 \
    RSGP_MODE="${rsgp_mode}" \
    RSGP_PPG_PROTECTED_TOPK="${protected_topk}" \
    RSGP_ROLE_MODE="${role_mode}" \
    RSGP_STRUCTURAL_PRIOR_PATH="${RSGP_STRUCTURAL_PRIOR_PATH}" \
    python tools/eval_once.py \
      --config "${CONFIG}" \
      --checkpoint "${CHECKPOINT}" \
      --checkpoint-load-mode "${CHECKPOINT_LOAD_MODE}" \
      --split "${SPLIT}" \
      --device "${DEVICE}" \
      --filter-method "${filter_method}" \
      --output "${out_dir}/${SPLIT}_metrics.json" \
      > "${out_dir}/${SPLIT}.log" 2>&1
}

run_case "ppg_10000" "PPG"
run_case "ppn_10000" "PPN"
run_case "rsgp_legacy_manual_8000_2000" "RSGP" "HYBRID" "8000" "legacy_manual"
run_case "rsgp_rs_only" "RSGP" "RS_ONLY" "0"
run_case "rsgp_ppn_graph" "RSGP" "PPN_GRAPH" "0"
run_case "rsgp_hybrid_9000_1000" "RSGP" "HYBRID" "9000"
run_case "rsgp_hybrid_8000_2000" "RSGP" "HYBRID" "8000"
run_case "rsgp_hybrid_7000_3000" "RSGP" "HYBRID" "7000"

python - "${OUTPUT_ROOT}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
rows = []
for metrics_path in sorted(root.glob("*/*_metrics.json")):
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    coverage = metrics.get("candidate-stage-coverage", {})
    rows.append(
        {
            "case": metrics_path.parent.name,
            "R@1500": metrics.get("R", {}).get("1500"),
            "R@2000": metrics.get("R", {}).get("2000"),
            "mR@1500": metrics.get("mR", {}).get("1500"),
            "mR@2000": metrics.get("mR", {}).get("2000"),
            "HMR@1500": metrics.get("HR", {}).get("1500"),
            "HMR@2000": metrics.get("HR", {}).get("2000"),
            "final_pair_coverage": coverage.get("final"),
            "ppn_pool_coverage": coverage.get("ppn"),
            "strict_degree_cap_coverage": coverage.get("degree_cap"),
            "candidate_graph": metrics.get("candidate-graph", {}),
            "candidate_pressure": metrics.get("candidate-pressure", {}),
        }
    )
out = root / "comparison.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {out}")

by_case = {row["case"]: row for row in rows}
baseline = by_case.get("ppg_10000")
statistical_names = (
    "rsgp_rs_only",
    "rsgp_ppn_graph",
    "rsgp_hybrid_9000_1000",
    "rsgp_hybrid_8000_2000",
    "rsgp_hybrid_7000_3000",
)
selection = {
    "split": "val",
    "baseline": "ppg_10000",
    "criterion": {
        "HMR@2000": "strictly greater than PPG",
        "R@2000": "at least PPG minus 0.002",
    },
    "eligible": [],
    "selected": None,
}
if baseline is not None:
    baseline_r = baseline.get("R@2000")
    baseline_hmr = baseline.get("HMR@2000")
    if baseline_r is not None and baseline_hmr is not None:
        eligible = [
            by_case[name]
            for name in statistical_names
            if name in by_case
            and by_case[name].get("R@2000") is not None
            and by_case[name].get("HMR@2000") is not None
            and by_case[name]["R@2000"] >= baseline_r - 0.002
            and by_case[name]["HMR@2000"] > baseline_hmr
        ]
        eligible.sort(key=lambda row: (row["HMR@2000"], row["R@2000"]), reverse=True)
        selection["eligible"] = [row["case"] for row in eligible]
        if eligible:
            selection["selected"] = eligible[0]["case"]
selection_path = root / "selection.json"
selection_path.write_text(
    json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(f"Wrote {selection_path}")
PY

echo "Done. Logs and JSON files are under ${OUTPUT_ROOT}"
