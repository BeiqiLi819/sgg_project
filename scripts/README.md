# Script entry points

Only the six `*_full.sh` launchers below are the stable public training and
evaluation API. They expose the same Full method for all three OBB tasks:
Dual-view RCA + auxiliary LA + statistical RSGP Hybrid 9000/1000.

## Public Full release

| Task | Train from detector initialization | Test released checkpoint |
|---|---|---|
| PredCls | `bash scripts/train_star_predcls_full.sh` | `bash scripts/test_star_predcls_full.sh` |
| SGCls | `bash scripts/train_star_sgcls_full.sh` | `bash scripts/test_star_sgcls_full.sh` |
| SGDet | `bash scripts/train_star_sgdet_full.sh` | `bash scripts/test_star_sgdet_full.sh` |

The train launchers use PPG on validation for checkpoint selection and then
automatically test the frozen checkpoint with statistical RSGP. RSGP never
changes the supervised training graph. PredCls/SGCls select
`model_best_HR.pth`; SGDet uses the final 5,000-step `model_last.pth`.

The test launchers read the task checkpoint from `pretrained/full/`, run the
test split, and write to `outputs/star_<task>_obb_full_eval/`. They accept
environment overrides without editing code, for example:

```bash
CHECKPOINT=/path/to/model.pth \
OUTPUT_DIR=outputs/custom_predcls_eval \
RUN_BACKGROUND=0 \
  bash scripts/test_star_predcls_full.sh
```

SGCls defaults to the STAR-compatible GT filter-label protocol; SGDet defaults
to `matched_gt` and requires the frozen v5 detection cache.

## Setup and cached artifacts

```bash
# Create and verify a clean environment
bash scripts/create_clean_env.sh
bash scripts/smoke_test_clean_env.sh

# Build train-derived RSGP statistics and frozen SGDet detections
bash scripts/build_rsgp_structural_prior.sh
bash scripts/build_sgdet_detection_cache.sh
```

## Research reproduction (not public API)

```bash
# Controlled PredCls Base(Unified)/D/DL training and best-checkpoint PPG tests
bash scripts/research/run_paper_predcls_suite.sh

# Select RSGP only on validation, then run final PredCls tests. The final
# suite includes the Base/PPG, Base/RSGP, Dual/PPG, and Dual/RSGP 2x2
# comparison, plus the fixed Dual+LA PPG/PPN/RSGP comparison.
bash scripts/research/select_predcls_rsgp_on_val.sh
bash scripts/research/eval_paper_predcls_suite.sh

# SGCls and matched-budget SGDet training
bash scripts/research/run_star_sgcls_experiment.sh
bash scripts/research/run_star_sgdet_rpcm_budget.sh

# Same-checkpoint PPG/RSGP cross-task tests
bash scripts/research/eval_paper_cross_task_suite.sh

# Export selected cases and render spatially aligned GT/PPG/RSGP scene graphs
bash scripts/research/export_paper_qualitative_cases.sh
```

These commands preserve the paper ablations and selection provenance. They are
not additional released models or supported public entry points. The frozen
paper default is statistical RSGP Hybrid 9000/1000 with all components
enabled. `select_predcls_rsgp_on_val.sh` is the provenance run; public and
internal task wrappers use protected top-k 9000 unless an ablation explicitly
overrides it.

`eval_paper_cross_task_suite.sh` accepts `SGCLS_CHECKPOINT` and
`SGDET_CHECKPOINT`, allowing compatible recovered checkpoints to be evaluated
without copying multi-gigabyte files into the formal output directories.

## Useful advanced entry points

- `research/run_predcls_minimal_ablation.sh`: run one controlled PredCls row. In
  addition to `BASE/D/DL`, `SOURCE` runs the original SGG-ToolKit audit and
  `D6850` runs the historical 6850 reconstruction:
  Dual SS/OO, the exact single-prototype head, seed 1029, and the original
  20,000-step schedule. `MV` is retained only as a command-line alias for
  `D6850`; the low-rank cross-role trial is no longer a public experiment.
- `_internal/run_star_experiment.sh`: generic training backend used by task
  launchers.
- `_internal/run_star_sgdet_experiment.sh`: cached SGDet training backend.
- `_internal/eval_star_{predcls,sgcls,sgdet}.sh`: evaluate one
  task/checkpoint/filter.
- `_internal/eval_once.sh`: low-level evaluator with explicit config and
  checkpoint.
- `research/eval_predcls_rsgp_ablation.sh`: one-shot test component ablation after
  validation has frozen the statistical RSGP mode and protected-pool size.
- `plot_train_log_curves.py`: render loss and metric curves from a log.

Historical HPRC, RPCM-reproduction, probe, compare, and compatibility-alias
launchers are deliberately not retained. Their result files may remain under
archived output directories, but they are not part of the release workflow.
The `research/` directory contains the paper-only entry points. The
`_internal/` directory contains helper scripts invoked by public and research
launchers and is not an additional experiment surface.
