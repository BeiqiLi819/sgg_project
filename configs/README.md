# BARC-SGG configurations

Public entry points:

- `star_barc_predcls.py`
- `star_barc_sgcls.py`
- `star_barc_sgdet.py`

Each entry point enables the same BARC components: BAP Q5 ordinary global
Top-10k proposal selection, Dual Role-aware RPCM reasoning, and BAC H+U
predicate recalibration. The `star_*_obb*.py` files are minimal STAR detector
and task bases kept only to avoid duplicating the model/data contract.

Override external artifacts with:

```bash
BARC_BAP_CHECKPOINT=/path/to/bap.pth
BARC_HARD_PREDICATE_MANIFEST=/path/to/hard_top20.json
```
