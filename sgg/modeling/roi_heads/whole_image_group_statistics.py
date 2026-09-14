"""Compatibility surface for STAR-Q2 whole-image semantic statistics."""

from sgg.modeling.roi_heads.set_context import (
    ImageSetStatistics,
    build_image_set_statistics,
    exact_within_group_percentile,
    set_context_features,
)

__all__ = [
    "ImageSetStatistics",
    "build_image_set_statistics",
    "exact_within_group_percentile",
    "set_context_features",
]

