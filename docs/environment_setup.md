# Clean environment for STAR SGG

## Dependency boundary

The public Full training and evaluation paths directly require eight external
runtime packages:

```text
torch, torchvision, numpy, opencv-python, Pillow, h5py, tqdm, mmcv-full
```

`mmcv-full==1.7.2` additionally imports four small runtime dependencies, all of
which are pinned in `requirements.clean.txt`:

```text
addict, packaging, PyYAML, yapf
```

The resulting Full-route dependency closure is therefore 12 distributions.
Conda supplies Python, pip, and Ninja; the installation script pins
setuptools/wheel for the legacy MMCV build and installs CUDA-matched
PyTorch/torchvision separately.

These packages are also present in the original RPCM environment, but they are
normal shared framework dependencies. `mmcv-full` is the only unavoidable
OpenMMLab binary dependency because the project uses its rotated IoU, rotated
NMS, and rotated RoIAlign CUDA operators.

The project does **not** import `maskrcnn_benchmark`, `mmdet`, `mmrotate`,
`torch_geometric`, `torch_scatter`, or `torch_sparse`. OBB/polygon conversion is
implemented locally in `sgg/modeling/core/obb_ops.py`; it is numerically aligned
with the three conversion helpers previously imported from mmrotate.

The public Full predictor is `RPCM_ORIGINAL_LEGACY`. It does not execute the
source-audit KMeans branch, so `scikit-learn` and `scipy` are not Full-route
dependencies. They are needed only by the research-only
`RPCM_SGG_TOOLKIT_ORIGINAL` audit. Plotting and pytest are likewise optional
and live in `requirements.analysis.txt`.

## Full-route dependency map

| Function | Package(s) |
|---|---|
| tensors, optimization, CUDA, data loading | torch |
| pretrained model utilities / optional axis-aligned RoIAlign | torchvision |
| rotated IoU, NMS, and RoIAlign | mmcv-full |
| STAR arrays and RSGP statistics | numpy |
| image/OBB geometry helpers | opencv-python |
| large satellite-image loading | Pillow |
| STAR annotation storage | h5py |
| train/test progress | tqdm |
| MMCV configuration/runtime helpers | addict, packaging, PyYAML, yapf |

Model weights, GloVe vectors, STAR data, the RSGP structural prior, and the
SGDet detection cache are runtime artifacts rather than Python packages.

The old `pyg` environment is not isolated. Its `easy-install.pth` contains
absolute paths to:

```text
/home/ubuntu/research/ssd/RPCM
/home/ubuntu/research/ssd/RPCM/mmrote_RS
```

Consequently, importing `maskrcnn_benchmark` or `mmrotate` executes source code
from the original RPCM checkout rather than a self-contained installed wheel.

## Setup from scratch

The installer creates a new minimal environment from pinned dependencies. It
does not read or clone the existing `pyg` environment:

```bash
ENV_NAME=sgg CUDA_HOME=/usr/local/cuda-12.1 MAX_JOBS=8 \
  bash scripts/create_clean_env.sh
conda activate sgg
```

The script installs the tested versions:

```text
Python       3.11
PyTorch      2.2.2+cu121
torchvision  0.17.2+cu121
mmcv-full    1.7.2 (compiled with CUDA ops)
numpy        1.26.4
```

`mmcv-full` 1.7.2 has no generally usable prebuilt wheel for every
Python/PyTorch/CUDA combination. The scratch route therefore compiles it. If a
matching wheel has already been built, avoid recompilation with:

```bash
MMCV_WHEEL=/path/to/mmcv_full-1.7.2-...whl \
ENV_NAME=sgg \
  bash scripts/create_clean_env.sh
```

Do not install both `mmcv` and `mmcv-full`: the packages share the same Python
module name, while only `mmcv-full` contains the required compiled operators.

## Validation

Run the strict environment audit after any dependency change:

```bash
python tools/check_environment.py --strict --require-cuda
```

It verifies all 12 pinned distributions, executes rotated IoU/NMS and
RoIAlignRotated on CPU and CUDA, imports all three public Full configs plus the
dataset/trainer/evaluator/detector modules, checks the Full predictor and graph
contract, checks local OBB helpers, rejects absolute RPCM/SGG-ToolKit paths,
and reports legacy packages still importable. Omit `--require-cuda` only when
checking a CPU-only development shell.

The audit covers the Python dependency closure. End-to-end artifact and data
loading is covered separately by the one-image public smoke test:

```bash
MAX_IMAGES=1 RUN_BACKGROUND=0 bash scripts/test_star_predcls_full.sh
```

Optional curve plotting and tests use:

```bash
pip install -r requirements.analysis.txt
```
