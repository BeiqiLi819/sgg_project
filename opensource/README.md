# STAR OBB Scene Graph Generation

This directory is the minimal release tree for the STAR oriented-bounding-box
(OBB) scene-graph-generation implementation. It contains the runnable model,
STAR data adapter, training/evaluation entry points, three task configurations,
and a small compatibility test set.

## Supported tasks

| Task | Boxes | Object labels | Predicates |
|---|---|---|---|
| PredCls | annotated OBBs | annotated labels | predicted |
| SGCls | annotated OBBs | predicted | predicted |
| SGDet | predicted OBBs | predicted | predicted |

The relation stack includes the RPCM-compatible predictor, PPG/PPN/RSGP pair
proposal filters, OBB geometry utilities, and STAR evaluation. Research-only
diagnostics, paper artifacts, experiment sweeps, generated outputs, and local
checkpoints are intentionally not part of this release tree.

## Installation

The tested environment uses Python 3.11, PyTorch 2.2.2 with CUDA 12.1,
torchvision 0.17.2, and `mmcv-full==1.7.2`.

```bash
ENV_NAME=sgg CUDA_HOME=/usr/local/cuda-12.1 \
  bash scripts/create_clean_env.sh
conda activate sgg
python tools/check_environment.py --strict --require-cuda
```

PyTorch, CUDA, and MMCV must be installed with mutually compatible versions.
The installer builds MMCV with CUDA operators when no compatible wheel is
provided.

## Data and optional artifacts

Set the STAR dataset root:

```bash
export STAR_SGG_ROOT=/path/to/STAR_SGG
```

The expected dataset files are documented in
[`docs/data_and_checkpoints.md`](docs/data_and_checkpoints.md). Dataset files,
GloVe embeddings, detector checkpoints, relation checkpoints, and SGDet
detection caches must be obtained separately under their applicable licenses;
none are redistributed here.

The small semantic-support metadata file is included at
`pretrained/SF_list_support.json`. The binary files referenced by the configs
are external artifacts.

## Training

```bash
# PredCls
bash scripts/run_star_experiment.sh

# SGCls
bash scripts/run_star_sgcls_experiment.sh

# SGDet: build the frozen detector cache first
bash scripts/build_sgdet_detection_cache.sh
bash scripts/run_star_sgdet_experiment.sh
```

Use `CONFIG`, `OUTPUT_DIR`, `DEVICE`, `RESUME`, and the task-specific
environment variables to override defaults. Training outputs are written below
`outputs/`, which is ignored by Git.

## Evaluation

```bash
bash scripts/eval_star_predcls.sh
bash scripts/eval_star_sgcls.sh
bash scripts/eval_star_sgdet.sh
```

For an explicit run, use the lower-level entry point:

```bash
CONFIG=configs/star_predcls_obb_train.py \
CHECKPOINT=/path/to/checkpoint.pth \
OUTPUT_DIR=outputs/eval \
  bash scripts/eval_once.sh
```

## Tests

```bash
pytest -q tests
```

The included tests focus on OBB detector compatibility, classifier-channel
ordering, and RPCM graph behavior. Full-data smoke tests require the STAR data
root and applicable checkpoints.

## License and third-party components

This release tree does not yet declare a project license. Add the intended
license before public distribution. STAR, PyTorch, MMCV, torchvision, and any
external pretrained weights retain their respective licenses and terms.
