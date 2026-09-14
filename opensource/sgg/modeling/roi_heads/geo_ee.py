"""Geometry-aware entity-to-entity attention for the Original RPCM graph.

The module deliberately consumes an already-built entity adjacency.  It does
not construct or expand the graph, so GeoEE can only reweight RPCM's existing
entity-to-entity edges.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from sgg.modeling.core.obb_ops import angle_to_radians, get_boxlist_angle_unit


class OBBGeometryEncoder(nn.Module):
    """Encode eight directed OBB-relative features as per-head attention bias."""

    feature_dim = 8

    def __init__(self, num_heads: int = 4, hidden_dim: int = 32):
        super().__init__()
        if int(num_heads) != 4:
            raise ValueError(f"GeoEE is fixed to 4 heads, got {num_heads}")
        if int(hidden_dim) <= 0:
            raise ValueError("GeoEE geometry hidden_dim must be positive")
        self.num_heads = int(num_heads)
        self.bias_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), self.num_heads),
        )

    @staticmethod
    def complete_adjacency(
        proposals: Sequence,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Materialize the unchanged per-image complete RPCM EE topology."""

        total = sum(len(proposal) for proposal in proposals)
        adjacency = torch.zeros((total, total), device=device, dtype=dtype)
        offset = 0
        for proposal in proposals:
            count = len(proposal)
            if count > 1:
                adjacency[offset : offset + count, offset : offset + count] = 1
                adjacency[
                    offset : offset + count, offset : offset + count
                ].fill_diagonal_(0)
            offset += count
        return adjacency

    @classmethod
    def build_node_geometry(
        cls,
        proposals: Sequence,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return compact per-node OBB state used to form geometry on demand."""

        node_rows: list[torch.Tensor] = []
        for image_index, proposal in enumerate(proposals):
            if getattr(proposal, "mode", None) != "xywha":
                raise ValueError(
                    "GeoEE requires OBB proposals in xywha mode, got "
                    f"{getattr(proposal, 'mode', None)!r}"
                )
            boxes = proposal.bbox.to(device=device, dtype=dtype)
            if boxes.ndim != 2 or boxes.size(1) != 5:
                raise ValueError(
                    f"GeoEE expects OBB [N, 5], got {tuple(boxes.shape)}"
                )
            if not bool(torch.isfinite(boxes).all()):
                raise FloatingPointError("GeoEE received NaN/Inf OBB coordinates")
            width = boxes[:, 2]
            height = boxes[:, 3]
            if bool((width <= 0).any()) or bool((height <= 0).any()):
                raise ValueError("GeoEE requires strictly positive OBB widths/heights")
            angle = angle_to_radians(
                boxes[:, 4], get_boxlist_angle_unit(proposal)
            )
            width_is_long = width >= height
            long_side = torch.maximum(width, height)
            short_side = torch.minimum(width, height)
            long_angle = (
                angle
                + torch.where(
                    width_is_long,
                    angle.new_zeros(angle.shape),
                    angle.new_full(angle.shape, math.pi / 2.0),
                )
            )
            image_width, image_height = map(float, proposal.size)
            if not math.isfinite(image_width) or not math.isfinite(image_height):
                raise ValueError("GeoEE requires finite proposal image sizes")
            if image_width <= 0 or image_height <= 0:
                raise ValueError("GeoEE requires positive proposal image sizes")
            node_rows.append(
                torch.stack(
                    (
                        boxes[:, 0],
                        boxes[:, 1],
                        long_side,
                        short_side,
                        long_angle,
                        boxes.new_full((len(proposal),), image_width),
                        boxes.new_full((len(proposal),), image_height),
                        boxes.new_full((len(proposal),), float(image_index)),
                    ),
                    dim=1,
                )
            )
        if not node_rows:
            return torch.zeros((0, 8), device=device, dtype=dtype)
        node_geometry = torch.cat(node_rows, dim=0)
        if not bool(torch.isfinite(node_geometry).all()):
            raise FloatingPointError("GeoEE constructed NaN/Inf node geometry")
        return node_geometry

    @classmethod
    def edge_geometry_from_nodes(
        cls,
        node_geometry: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Construct the requested directed 8-D features for selected edges."""

        if node_geometry.ndim != 2 or node_geometry.size(1) != 8:
            raise ValueError(
                "GeoEE node geometry must be [N, 8], got "
                f"{tuple(node_geometry.shape)}"
            )
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"GeoEE edge index must be [2, E], got {tuple(edge_index.shape)}"
            )
        if edge_index.size(1) == 0:
            return node_geometry.new_zeros((0, cls.feature_dim))
        target, source = edge_index.unbind(dim=0)
        if bool((target < 0).any()) or bool((source < 0).any()):
            raise ValueError("GeoEE edge index cannot be negative")
        if bool((target >= node_geometry.size(0)).any()) or bool(
            (source >= node_geometry.size(0)).any()
        ):
            raise ValueError("GeoEE edge index exceeds the node count")
        center = node_geometry[:, :2]
        long_side = node_geometry[:, 2]
        short_side = node_geometry[:, 3]
        long_angle = node_geometry[:, 4]
        image_size = node_geometry[:, 5:7]
        image_id = node_geometry[:, 7]
        if bool((image_id[target] != image_id[source]).any()):
            raise ValueError("GeoEE found an EE edge crossing image boundaries")

        delta = center[source] - center[target]
        image_distance = torch.linalg.vector_norm(
            delta / image_size[target], dim=1
        )
        area_target = (long_side[target] * short_side[target]).clamp_min(1.0e-6)
        area_source = (long_side[source] * short_side[source]).clamp_min(1.0e-6)
        object_distance = torch.linalg.vector_norm(delta, dim=1) / torch.sqrt(
            area_target
        )
        log_area_ratio = torch.log(area_source / area_target)
        aspect_target = long_side[target] / short_side[target].clamp_min(1.0e-6)
        aspect_source = long_side[source] / short_side[source].clamp_min(1.0e-6)
        log_aspect_ratio = torch.log(aspect_source / aspect_target)
        delta_angle = long_angle[source] - long_angle[target]

        cos_target = torch.cos(long_angle[target])
        sin_target = torch.sin(long_angle[target])
        long_projection = (
            delta[:, 0] * cos_target + delta[:, 1] * sin_target
        ).abs() / long_side[target].clamp_min(1.0e-6)
        short_projection = (
            -delta[:, 0] * sin_target + delta[:, 1] * cos_target
        ).abs() / short_side[target].clamp_min(1.0e-6)

        geometry = torch.stack(
            (
                image_distance,
                object_distance,
                log_area_ratio,
                log_aspect_ratio,
                torch.cos(2.0 * delta_angle),
                torch.sin(2.0 * delta_angle),
                long_projection,
                short_projection,
            ),
            dim=1,
        )
        if geometry.shape != (edge_index.size(1), cls.feature_dim):
            raise RuntimeError(
                "GeoEE geometry shape mismatch: "
                f"{tuple(geometry.shape)} vs {(edge_index.size(1), cls.feature_dim)}"
            )
        if not bool(torch.isfinite(geometry).all()):
            raise FloatingPointError("GeoEE constructed NaN/Inf geometry features")
        return geometry

    @classmethod
    def build_edge_geometry(
        cls,
        proposals: Sequence,
        adjacency: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize edge geometry for tests/audits; training uses node state."""

        if adjacency.ndim != 2 or adjacency.size(0) != adjacency.size(1):
            raise ValueError(
                "GeoEE adjacency must be square, got "
                f"{tuple(adjacency.shape)}"
            )
        total = sum(len(proposal) for proposal in proposals)
        if adjacency.size(0) != total:
            raise ValueError(
                "GeoEE adjacency/object mismatch: "
                f"{adjacency.size(0)} rows for {total} objects"
            )
        node_geometry = cls.build_node_geometry(
            proposals,
            device=adjacency.device,
            dtype=dtype or adjacency.dtype,
        )
        edge_index = (adjacency > 0).nonzero(as_tuple=False).t().contiguous()
        return edge_index, cls.edge_geometry_from_nodes(node_geometry, edge_index)

    def forward(self, edge_geometry: torch.Tensor) -> torch.Tensor:
        if edge_geometry.ndim != 2 or edge_geometry.size(1) != self.feature_dim:
            raise ValueError(
                "GeoEE geometry encoder expects [E, 8], got "
                f"{tuple(edge_geometry.shape)}"
            )
        return self.bias_mlp(edge_geometry)


class GeometryAwareEEAttention(nn.Module):
    """Four-head attention masked to an externally supplied EE adjacency."""

    def __init__(
        self,
        feature_dim: int,
        *,
        num_heads: int = 4,
        head_dim: int = 64,
        geometry_hidden_dim: int = 32,
        query_chunk_size: int = 64,
    ):
        super().__init__()
        if int(feature_dim) <= 0 or int(head_dim) <= 0:
            raise ValueError("GeoEE feature_dim and head_dim must be positive")
        if int(num_heads) != 4:
            raise ValueError(f"GeoEE is fixed to 4 heads, got {num_heads}")
        self.feature_dim = int(feature_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.query_chunk_size = int(query_chunk_size)
        if self.query_chunk_size <= 0:
            raise ValueError("GeoEE query_chunk_size must be positive")
        projected_dim = self.num_heads * self.head_dim
        self.query = nn.Linear(self.feature_dim, projected_dim)
        self.key = nn.Linear(self.feature_dim, projected_dim)
        self.value = nn.Linear(self.feature_dim, projected_dim)
        self.output = nn.Linear(projected_dim, self.feature_dim)
        self.geometry_encoder = OBBGeometryEncoder(
            num_heads=self.num_heads,
            hidden_dim=int(geometry_hidden_dim),
        )

    def _attend_query_chunk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        adjacency: torch.Tensor,
        node_geometry: torch.Tensor,
        target_offset: torch.Tensor,
    ) -> torch.Tensor:
        """Attend one bounded query block while preserving the exact edge mask."""

        chunk_size = query.size(0)
        count = key.size(0)
        scores = torch.einsum("ihd,jhd->hij", query, key) / math.sqrt(
            float(self.head_dim)
        )
        mask = adjacency > 0
        scores = scores.masked_fill(
            ~mask.unsqueeze(0), torch.finfo(scores.dtype).min
        )
        local_edges = mask.nonzero(as_tuple=False)
        if local_edges.numel():
            local_target, source = local_edges.unbind(dim=1)
            edge_index = torch.stack(
                (local_target + target_offset, source), dim=0
            )
            edge_geometry = OBBGeometryEncoder.edge_geometry_from_nodes(
                node_geometry, edge_index
            )
            edge_bias = self.geometry_encoder(
                edge_geometry.to(device=query.device, dtype=query.dtype)
            )
            dense_bias = edge_bias.new_zeros(
                (self.num_heads, chunk_size, count)
            )
            dense_bias[:, local_target, source] = edge_bias.t()
            scores = scores + dense_bias

        weights = torch.softmax(scores, dim=-1) * mask.unsqueeze(0).to(
            dtype=scores.dtype
        )
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        return torch.einsum("hij,jhd->ihd", weights, value)

    def forward(
        self,
        entity_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_geometry: torch.Tensor,
    ) -> torch.Tensor:
        if entity_features.ndim != 2 or entity_features.size(1) != self.feature_dim:
            raise ValueError(
                "GeoEE entity features must be [N, feature_dim], got "
                f"{tuple(entity_features.shape)}"
            )
        count = entity_features.size(0)
        if adjacency.shape != (count, count):
            raise ValueError(
                f"GeoEE adjacency must be {(count, count)}, got {tuple(adjacency.shape)}"
            )
        if node_geometry.shape != (count, 8):
            raise ValueError(
                "GeoEE node geometry mismatch: "
                f"{tuple(node_geometry.shape)} for {count} entities"
            )
        if count == 0:
            return entity_features.new_zeros(entity_features.shape)

        query = self.query(entity_features).view(
            count, self.num_heads, self.head_dim
        )
        key = self.key(entity_features).view(
            count, self.num_heads, self.head_dim
        )
        value = self.value(entity_features).view(
            count, self.num_heads, self.head_dim
        )
        attended_chunks: list[torch.Tensor] = []
        for start in range(0, count, self.query_chunk_size):
            end = min(start + self.query_chunk_size, count)
            args = (
                query[start:end],
                key,
                value,
                adjacency[start:end],
                node_geometry,
                torch.tensor(start, device=entity_features.device),
            )
            if self.training and torch.is_grad_enabled():
                attended_chunk = checkpoint(
                    self._attend_query_chunk,
                    *args,
                    use_reentrant=False,
                )
            else:
                attended_chunk = self._attend_query_chunk(*args)
            attended_chunks.append(attended_chunk)
        attended = torch.cat(attended_chunks, dim=0).reshape(
            count, self.num_heads * self.head_dim
        )
        output = self.output(attended)
        return output * (adjacency > 0).any(dim=1, keepdim=True).to(output.dtype)


__all__ = ["OBBGeometryEncoder", "GeometryAwareEEAttention"]
