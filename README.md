# STAR OBB Scene Graph Generation

PyTorch implementation of oriented-bounding-box (OBB) scene graph generation
on STAR. The current project covers Predicate Classification (PredCls), Scene
Graph Classification (SGCls), and Scene Graph Detection (SGDet).

The main method contains three parts:

- **Role-aware Relation Context Aggregation (RCA)** separates
  shared-subject and shared-object edge propagation.
- **Auxiliary Logit Adjustment (LA)** keeps the main CE classifier unchanged
  and adds weak class-prior-aware supervision during training.
- **Remote-sensing Graph-aware Pair Proposal (RSGP)** combines PPG, PPN, OBB
  geometry, train-derived category-name-agnostic structural roles, degree
  capacity, and semantic-type capacity to construct the inference candidate
  graph.

The paper-facing route is therefore **dual-view RCA + LA + RSGP**. The former
HPRC/`tail_aux` route is retained only for checkpoint compatibility and
historical experiments; it is not part of the current main claim.

The paper-ready main-text/appendix split, formulas, final tables, claim
boundaries, and qualitative figures are collected in
[docs/paper_ready_manuscript.md](docs/paper_ready_manuscript.md).

SGDet uses a read-only detection cache so the frozen full-resolution
multi-scale detector is not rerun during every relation-head update.

## Task protocols

| Task | Boxes | Object labels | Predicates |
|---|---|---|---|
| PredCls | ground truth | ground truth | predicted |
| SGCls | ground truth | predicted | predicted |
| SGDet | predicted | predicted | predicted |

The default SGCls/SGDet pair-filter label sources reproduce the STAR
SGG-ToolKit comparison protocol:

```text
SGCLS_FILTER_LABEL_SOURCE=gt
SGDET_FILTER_LABEL_SOURCE=matched_gt
```

These labels are used for relation candidate filtering only. Final object and
triplet predictions still come from the model. Use `pred` for stricter
fully-predicted-label ablations.

## Repository layout

```text
configs/   OBB task and paper-ablation configurations
scripts/   one-command training, evaluation, cache, and experiment launchers
sgg/       datasets, detector, relation models, structures, and evaluation
tools/     focused diagnostics, cache builders, migrations, and PPN utilities
tests/     numerical, checkpoint-compatibility, and integration tests
train.py   common training entry point
```

Generated datasets, embeddings, detection caches, logs, and checkpoints are
excluded from Git.

## Installation

The tested environment uses Python 3.11, PyTorch 2.2.2 with CUDA 12.1,
torchvision 0.17.2, and `mmcv-full==1.7.2`.

```bash
ENV_NAME=sgg \
CUDA_HOME=/usr/local/cuda-12.1 \
MAX_JOBS=8 \
  bash scripts/create_clean_env.sh

conda activate sgg
python tools/check_environment.py --strict --require-cuda
```

The installer creates a new environment from scratch and does not import the
original RPCM or SGG-ToolKit checkout. See [INSTALL.md](INSTALL.md) for the
complete procedure and [docs/environment_setup.md](docs/environment_setup.md)
for the dependency boundary.

## Data and pretrained artifacts

Set the dataset root:

```bash
export STAR_SGG_ROOT=/path/to/STAR_SGG
```

Expected STAR files:

```text
$STAR_SGG_ROOT/
├── STAR_img/
├── STAR-SGG-with-attri.h5
├── STAR-SGG-dicts-with-attri.json
└── STAR_image_data_v1.json
```

Expected local artifacts:

```text
pretrained/
├── OBB_swin_L_OBD.pth              # frozen OBB detector
├── STAR_OBB.pth                    # original PPG
├── PPN_OBB.pth                     # independent PPN
├── rsgp_structural_prior.json      # train-derived statistical roles
├── SF_list_support.json            # semantic label-pair support
└── full/
    ├── STAR_OBB_Full_PredCls.pth
    ├── STAR_OBB_Full_SGCls.pth
    ├── STAR_OBB_Full_SGDet.pth
    └── manifest.json

glove/
├── glove.6B.200d.txt
└── glove.6B.300d.txt
```

Binary weights and GloVe files are intentionally ignored by Git. The semantic
support JSON, RSGP prior, and checkpoint manifest remain trackable. See
[pretrained/full/README.md](pretrained/full/README.md) for release provenance
and SHA256 checksums.

## Public Full training and evaluation

The stable public interface is deliberately limited to one train and one test
launcher per OBB task. Full means **Dual-view RCA + auxiliary LA + statistical
RSGP Hybrid 9000/1000**.

| Task | Train from detector initialization | Test released checkpoint |
|---|---|---|
| PredCls | `bash scripts/train_star_predcls_full.sh` | `bash scripts/test_star_predcls_full.sh` |
| SGCls | `bash scripts/train_star_sgcls_full.sh` | `bash scripts/test_star_sgcls_full.sh` |
| SGDet | `bash scripts/train_star_sgdet_full.sh` | `bash scripts/test_star_sgdet_full.sh` |

Training starts from `pretrained/OBB_swin_L_OBD.pth`, uses PPG on validation
for checkpoint selection, and automatically tests the frozen checkpoint with
Full RSGP. PredCls and SGCls select `model_best_HR.pth`; SGDet uses
`model_last.pth` at the fixed 5,000-step budget. RSGP is inference-only and
never changes the supervised relation pairs.

SGDet additionally needs the matching frozen v5 detector cache. Build it once
before training or evaluation:

```bash
bash scripts/build_sgdet_detection_cache.sh
```

All commands run in the background and write `train.log`, `train.pid`, and
`exit_code.txt` below their output directory.

The final submission protocol uses validation for all checkpoint and RSGP
choices and test only for final reporting. Experiments are single runs and do
not explicitly fix a random seed. See
[docs/submission_experiment_protocol.md](docs/submission_experiment_protocol.md).

## Research training and ablations

The commands below preserve experiment provenance and are not additional
stable public entry points. See [scripts/README.md](scripts/README.md) for the
complete distinction between public launchers and research helpers.

### Controlled PredCls rows

```bash
bash scripts/research/run_predcls_minimal_ablation.sh BASE
bash scripts/research/run_predcls_minimal_ablation.sh U
bash scripts/research/run_predcls_minimal_ablation.sh D
bash scripts/research/run_predcls_minimal_ablation.sh DL
```

Each fresh run loads only `pretrained/OBB_swin_L_OBD.pth`; the relation stack
is initialized from the row's configured RPCM/GloVe initializers.

### PredCls main relation model: dual-view RCA + LA

```bash
bash scripts/research/run_predcls_minimal_ablation.sh DL
```

This detector-only scratch run writes to
`outputs/paper_submission/predcls/dual_la/`. PPG on the validation
split selects the checkpoint. RSGP remains inference-only and reuses that
exact checkpoint. The launcher automatically runs one PPG test with
`model_best_HR.pth` after successful training.

### SGCls

```bash
bash scripts/research/run_star_sgcls_experiment.sh
```

The default paper configuration is
`configs/star_sgcls_obb_dual_la_train.py` and writes to
`outputs/paper_submission/sgcls`. It retains SGCls object refinement but
uses the same Dual+LA relation branch as PredCls. The current paper artifact
reuses `outputs/star_sgcls_obb_dual_la_train/model_best_HR.pth` and evaluates
PPG and statistical RSGP on that same checkpoint through
`eval_paper_cross_task_suite.sh`. The resulting SGCls and SGDet tables provide
the full three-task comparison with RPCM.

### SGDet

Build the frozen-detector cache once:

```bash
bash scripts/build_sgdet_detection_cache.sh
```

The default root is `star_sgdet_detection_cache/` and train, validation, and
test manifests are all required for the final protocol. The checked workspace
already contains the complete v5 train/val/test cache (771/245/264 images), so
paper SGDet runs should reuse it. Rebuild only after changing a detector-side
setting covered by the cache hash.

Train the SGDet model under the optimization budget published in
SGG-ToolKit's `LOBB_RPCM_sgdet_train.sh`, run:

```bash
bash scripts/research/run_star_sgdet_rpcm_budget.sh
```

This control uses batch size 2, configured LR `1e-3` (effective LR `2e-3`
after the source-compatible batch multiplier), 5,000 optimizer steps, LR
milestones at 3,000/4,000, and 500 warmup steps. It reuses the same v5
detection cache and writes to
`outputs/paper_submission/sgdet_rpcm_budget`, leaving the longer development
run untouched. This RPCM-matched-budget run is the paper-facing SGDet result.
After training it automatically evaluates `model_last.pth`; unlike PredCls and
SGCls, SGDet does not switch to a best-validation checkpoint.
Evaluate the full 5,000-step endpoint through the maintained cross-task suite:

```bash
bash scripts/research/eval_paper_cross_task_suite.sh
```

The paper-facing RPCM-budget SGDet launcher fixes PPG for training-time
validation/checkpoint protocol; PPG/RSGP are compared afterward on its same
final checkpoint.

The filter remains inference-only, matching the STAR/PPG protocol: it controls
periodic validation, test-time candidate graphs, and therefore best-checkpoint
selection, but it does not prune supervised relation pairs used to compute the
training loss.

### Resume

Use the same task launcher with a full checkpoint:

```bash
RESUME=outputs/paper_submission/predcls/dual_la/model_last.pth \
  bash scripts/research/run_predcls_minimal_ablation.sh DL

RESUME=outputs/paper_submission/sgcls/model_last.pth \
  bash scripts/research/run_star_sgcls_experiment.sh

RESUME=outputs/paper_submission/sgdet_rpcm_budget/model_last.pth \
  bash scripts/research/run_star_sgdet_rpcm_budget.sh
```

## Research evaluation

Build the one-time structural prior, select RSGP on validation, and run the
fixed-checkpoint PredCls comparisons:

```bash
bash scripts/build_rsgp_structural_prior.sh
bash scripts/research/select_predcls_rsgp_on_val.sh
bash scripts/research/eval_paper_predcls_suite.sh
```

The paper default is `RSGP_ROLE_MODE=statistical`. It never matches object
class names or fixed predicate IDs. Use
`RSGP_ROLE_MODE=legacy_manual FILTER_METHOD=RSGP ...` only to replay the old
manual-role RSGP-v1 results.

Validation selected the final Hybrid 9000/1000 graph: 9,000 PPG edges are
protected and the remaining budget is completed with PPN and statistical
RSGP evidence. A second validation-only check retained all three statistical
structural roles. These choices are now the defaults for PredCls, SGCls, and
SGDet; test component ablations do not change them.

For custom evaluations, the task wrappers and low-level evaluator live under
`scripts/_internal/`; they require explicit configs/checkpoints and are
documented in `scripts/README.md`.

## Paper experiment provenance

```bash
# Controlled Base(Unified)/D/DL runs; common detector/budget/selection
bash scripts/research/run_paper_predcls_suite.sh

# Optional complete source-RPCM audit outside the main causal chain
bash scripts/research/run_predcls_minimal_ablation.sh SOURCE

# One-off decision run: current Dual+LA graph front end with the source head
bash scripts/research/run_predcls_minimal_ablation.sh DL_ORIGINAL_HEAD

# Historical Dual/6850 protocol reconstruction (isolated audit)
bash scripts/research/run_predcls_minimal_ablation.sh D6850

# Select RSGP on validation
bash scripts/research/select_predcls_rsgp_on_val.sh

# Final Base/D/DL PPG and fixed-checkpoint PPG/PPN/RSGP test
bash scripts/research/eval_paper_predcls_suite.sh
python tools/summarize_paper_experiments.py

# Extract the complete 58-class table and selected qualitative cases
python tools/extract_paper_supplement.py --render-thumbnails

# Run only the selected cases and render aligned GT/PPG/RSGP scene graphs
bash scripts/research/export_paper_qualitative_cases.sh

# SGCls and same-budget SGDet completeness experiments
bash scripts/research/run_star_sgcls_experiment.sh
bash scripts/research/run_star_sgdet_rpcm_budget.sh
bash scripts/research/eval_paper_cross_task_suite.sh
```

`Base` is the former Unified row: current RCA GNN, unified relation adjacency,
the common 6850-compatible classifier, and PPG. The strict causal chain is
Base→D→DL→Full. `SOURCE` retains the isolated `RPCM_SGG_TOOLKIT_ORIGINAL`
predictor as an audit and is not duplicated in the main ablation table.
The optional `DL_ORIGINAL_HEAD` run writes to
`outputs/paper_submission/predcls/dual_la_original_head`; it does not replace
the paper DL row automatically.

The PredCls suite validates every two epochs. Base/D/DL start at epoch 120;
the optional source audit and `DL_ORIGINAL_HEAD` start at epoch 70 so their
best-epoch ranges can be searched again. Early-stop
patience is activated at epoch 120 for every row and stops training after 10
validation events without a new HMR@2000 best; 220 epochs is a safety cap.
Existing
`model_last.pth` files are detected and resumed automatically, so rerunning
the suite continues an interrupted Base/D/DL row before moving to the next
one. Rows with an existing `test/ppg/test_metrics.json` are skipped. Each
launch appends a timestamped session to `suite.log`, and the active config is
recopied into the run directory on resume. Use `AUTO_RESUME=0` only for an
intentional fresh run in a different output directory; use
`SKIP_COMPLETED=0` to force traversal of completed rows.

For checkpoints saved before early stopping was introduced, resume rebuilds
the patience counter from that row's `validation_history.jsonl`. Therefore a
row that has already exceeded the no-improvement window exits training
immediately and proceeds to the automatic test of `model_best_HR.pth`.

The cross-task suite produces both STAR-compatible (`legacy`) and strict
predicted-label (`pred`) PPG/RSGP results. Every final evaluation JSON also
contains candidate-graph statistics, candidate-load pressure groups, class-wise
recall, and image-wise case records. The frozen older JSON files remain
supported through their evaluation logs and compact failure cases.

Method definitions, formulas, current results, and table provenance are
recorded in
[docs/paper_contributions_and_experiments.md](docs/paper_contributions_and_experiments.md).
The complete paper supplement table is in
[docs/per_predicate_supplement.md](docs/per_predicate_supplement.md).
The RSGP design boundary is recorded in
[docs/rsgp_technical_route.md](docs/rsgp_technical_route.md).

## Diagnostics and tests

```bash
# Environment and rotated CUDA operators
python tools/check_environment.py --strict --require-cuda

# Object/detection error decomposition
python tools/diagnose_object_classification.py --help

# Parse and plot a training log
python scripts/plot_train_log_curves.py outputs/.../train.log

# Optional test/plot dependencies, then the test suite
pip install -r requirements.analysis.txt
pytest -q
```

The complete clean-environment smoke evaluation is:

```bash
bash scripts/smoke_test_clean_env.sh
```

The maintained script surface and the distinction between public workflows
and internal helpers are listed in [scripts/README.md](scripts/README.md).

## Reproducibility notes

- `FILTER_METHOD` must be one of `PPG`, `PPN`, or `RSGP`; unfiltered STAR
  relation graphs are disabled because all-pairs inference can cause OOM.
- PredCls excludes the constant GT object-refinement loss where configured.
  SGCls and SGDet retain trainable object-refinement CE.
- `outputs/`, `outputs_old/`, and `star_sgdet_detection_cache/` are not
  versioned. Preserve checkpoint hashes, configs, logs, and
  metric JSONs used for publication separately.
- Existing HPRC results and `tail_aux` keys are historical compatibility
  artifacts and are not used as evidence for the current main method.
- Existing RSGP result tables generated before the statistical structural
  prior are `RSGP-v1/legacy_manual` evidence and must not be relabeled as the
  category-name-agnostic method.
- The legacy task configs `star_sgcls_obb_train.py` and
  `star_sgdet_obb_train.py` remain available only for old HPRC checkpoint
  replay. New paper runs use their `_dual_la_train.py` counterparts.

## License

No license has been selected. Add a `LICENSE` file before public distribution.
