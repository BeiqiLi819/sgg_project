# BARC-SGG

Budget-Aware, Role-structured and Calibrated Scene Graph Generation for
large-scale remote-sensing imagery.

This is the minimal public release tree for BARC-SGG. The default pipeline is
organized around the three method components:

1. **BAP** — whole-image budget-aware pair proposal using a target-free
   residual over the released PPG score;
2. **Dual Role-aware RPCM** — separate shared-subject and shared-object
   relation propagation;
3. **BAC** — base-anchored predicate recalibration with an all-class residual
   and a frozen-validation hard-predicate auxiliary branch.

## Tasks

| Task | Configuration |
|---|---|
| PredCls | `configs/star_barc_predcls.py` |
| SGCls | `configs/star_barc_sgcls.py` |
| SGDet | `configs/star_barc_sgdet.py` |

The `sgg/` package contains the BARC runtime and compatible STAR OBB relation
components. Generic STAR task configs are kept only as inheritance bases;
BARC configs are the public entry points.

## Installation

The tested environment uses Python 3.11, PyTorch 2.2.2 with CUDA 12.1,
torchvision 0.17.2, and `mmcv-full==1.7.2`.

```bash
ENV_NAME=sgg CUDA_HOME=/usr/local/cuda-12.1 \
  bash scripts/create_clean_env.sh
conda activate sgg
python tools/check_environment.py --strict --require-cuda
```

## Data and external artifacts

Set the STAR dataset root:

```bash
export STAR_SGG_ROOT=/path/to/STAR_SGG
```

STAR images/annotations, GloVe embeddings, the released PPG checkpoint, the
Dual RPCM base checkpoint, and trained BAP/BAC checkpoints are not bundled.
See [`docs/data_and_checkpoints.md`](docs/data_and_checkpoints.md) for the
expected paths and artifact boundaries.

The lightweight, TEST-free BARC hard-predicate manifest is included at
`pretrained/BARC_hard_predicates_top20.json`.

## BARC evaluation

Provide a combined BARC checkpoint and the trained BAP proposal checkpoint:

```bash
CONFIG=configs/star_barc_predcls.py \
CHECKPOINT=/path/to/barc_predcls.pth \
BARC_BAP_CHECKPOINT=/path/to/barc_bap.pth \
OUTPUT_DIR=outputs/barc_predcls_eval \
  bash scripts/eval_once.sh
```

For task wrappers:

```bash
bash scripts/eval_barc_predcls.sh
bash scripts/eval_barc_sgcls.sh
bash scripts/eval_barc_sgdet.sh
```

SGDet requires a compatible frozen-detector cache before evaluation.

## BARC training stages

The method-specific tools are named after their BARC stage:

```bash
# Build target-free BAP context metadata and train the BAP residual.
python tools/build_barc_bap_context.py --help
python tools/train_barc_bap.py --help
python tools/export_barc_bap_pairs.py --help

# Export cached BAC features and train proposal-consistent BAC.
python tools/export_barc_bac_cache.py --help
python tools/train_barc_bac.py --help
```

These stages require external STAR data, compatible base checkpoints, and
their corresponding intermediate caches. Unannotated pairs are not treated as
background negatives in BAP/BAC.

## Tests

```bash
python -m pytest -q tests
```

The included tests cover core OBB/RPCM compatibility and run without external
weights; tests that inspect the released detector checkpoint are skipped when
that optional artifact is absent.

## Release boundary

This tree intentionally excludes experiment outputs, paper figures, internal
research-round diagnostics, local caches, and binary model weights. Add the
project's chosen license before public distribution, and comply with the
licenses of STAR, PyTorch, MMCV, torchvision, GloVe, and all external weights.
