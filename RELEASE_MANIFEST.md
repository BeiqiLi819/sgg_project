# Minimal release manifest

Included:

- `sgg/`: runnable STAR OBB SGG runtime and model implementation;
- `train.py`: training entry point;
- `tools/`: environment check, evaluation, detector-cache build, and checkpoint
  class-order migration;
- `configs/`: base PredCls/SGCls/SGDet configs and their required inheritance
  chain;
- `scripts/`: clean environment setup and task launch/evaluation wrappers;
- `tests/`: small dependency-light compatibility and model smoke tests;
- lightweight metadata and dependency documentation.

Intentionally excluded:

- STAR images, annotations, and GloVe embeddings;
- detector/relation checkpoints and SGDet caches;
- `outputs/`, `outputs_old/`, logs, experiment results, and paper figures;
- research-round diagnostics, ablation sweeps, downstream studies, and
  internal scripts;
- local editor, agent, and environment state.

The parent research workspace is unchanged. Publish the contents of this
directory only after adding the project's chosen license and confirming the
redistribution terms of all external dependencies and artifacts.
