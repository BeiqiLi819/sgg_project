# STAR OBB Full checkpoints

This directory is the public checkpoint layout for the three STAR OBB tasks.

| Task | Public checkpoint | Selection rule |
|---|---|---|
| PredCls | `STAR_OBB_Full_PredCls.pth` | best validation HMR checkpoint |
| SGCls | `STAR_OBB_Full_SGCls.pth` | best validation HMR checkpoint |
| SGDet | `STAR_OBB_Full_SGDet.pth` | final checkpoint at the fixed 5,000-step budget |

Frozen statistical RSGP Hybrid 9000/1000 test results at the public
STAR-compatible filter-label protocol are:

| Task | R@2000 | mR@2000 | HMR@2000 |
|---|---:|---:|---:|
| PredCls | 68.58 | 45.38 | 54.62 |
| SGCls | 57.78 | 33.75 | 42.61 |
| SGDet | 33.65 | 14.55 | 20.31 |

SGCls uses GT object labels only for pair filtering; SGDet uses matched-GT
labels only for pair filtering. Object and triplet outputs remain model
predictions, matching the STAR comparison protocol.

Full means Dual-view RCA + auxiliary logit adjustment + statistical RSGP
Hybrid 9000/1000. RSGP is inference-only, so its PPG/PPN proposal weights and
train-derived structural prior are shared by all three task checkpoints.

Required files outside this directory:

```text
pretrained/OBB_swin_L_OBD.pth
pretrained/STAR_OBB.pth
pretrained/PPN_OBB.pth
pretrained/rsgp_structural_prior.json
```

The binary `.pth` files are intentionally ignored by Git. Publish them as
release assets or through the project download location, preserving the names
above. `manifest.json` records their provenance and checksums.

Public evaluation:

```bash
bash scripts/test_star_predcls_full.sh
bash scripts/test_star_sgcls_full.sh
bash scripts/test_star_sgdet_full.sh
```

Public training from the detector-only initialization:

```bash
bash scripts/train_star_predcls_full.sh
bash scripts/train_star_sgcls_full.sh
bash scripts/train_star_sgdet_full.sh
```
