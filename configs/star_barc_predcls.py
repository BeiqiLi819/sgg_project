"""Public BARC-SGG PredCls configuration."""

from configs.barc_common import apply_barc
from configs.star_predcls_obb_train import cfg as _base

cfg = apply_barc(_base, task="predcls")
