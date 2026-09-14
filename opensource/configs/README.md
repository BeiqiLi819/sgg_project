# Configurations

The release keeps only the base OBB task configs and the small inheritance
chain needed by SGCls/SGDet:

- `star_predcls_obb_train.py`
- `star_predcls_obb_ablation_unified_rca_train.py`
- `star_predcls_obb_ablation_dual_rca_train.py`
- `star_predcls_obb_tail_aux_train.py`
- `star_sgcls_obb_train.py`
- `star_sgdet_obb_train.py`

Copy a config before making local changes. Dataset paths and checkpoint paths
should be supplied through environment variables or the copied config rather
than committed into the release.
