# Data and checkpoint layout

The release does not redistribute STAR images, annotations, GloVe embeddings,
detector weights, relation checkpoints, or generated SGDet caches.

Set `STAR_SGG_ROOT` to a local STAR checkout containing the files required by
the selected task. The default configs expect the STAR image directory,
annotation HDF5/JSON files, and the STAR dictionary metadata. If the local
dataset uses a different layout, override the corresponding `DATASETS.*`
paths in a copied config.

The model configs refer to these optional local artifacts:

```text
pretrained/OBB_swin_L_OBD.pth   frozen OBB detector
pretrained/STAR_OBB.pth         STAR PPG checkpoint
pretrained/PPN_OBB.pth          optional PPN checkpoint
glove/glove.6B.200d.txt         relation semantic initialization
```

Place compatible files at those paths or override the paths through the
configuration/environment variables. SGDet additionally requires a generated
cache when `SGDET_DETECTION_CACHE_ENABLED=1`.

The included `pretrained/SF_list_support.json` is metadata only; it is not a
model checkpoint.
