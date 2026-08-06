#!/usr/bin/env python3
"""Render spatially aligned GT/PPG/RSGP scene graphs for paper cases.

The satellite crop is shown only once.  GT, PPG, and RSGP are then rendered
on clean canvases using the same node coordinates, IDs, and class colors.  All
GT entities whose centers lie in the crop are retained. Exact method-specific
triplet differences are emphasized by edge status alone. Predicate text is
intentionally omitted because the figure compares graph structure and final
prediction correctness rather than individual predicate semantics.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import torch
from PIL import Image, ImageDraw

try:
    from tools import render_qualitative_cases as q
except ImportError:  # Direct execution from tools/.
    import render_qualitative_cases as q


RGB = Tuple[int, int, int]
Triple = Tuple[int, int, int]
Pair = Tuple[int, int]

GREEN: RGB = (37, 160, 75)
BLUE: RGB = (32, 105, 220)
PURPLE: RGB = (155, 70, 190)
RED: RGB = (210, 55, 55)
DARK_GRAY: RGB = (92, 98, 108)
ORANGE: RGB = (242, 145, 25)
BLACK: RGB = (24, 26, 30)
WHITE: RGB = (255, 255, 255)


def _class_color(label: int) -> RGB:
    hue = (int(label) * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.48, 0.86)
    return int(red * 255), int(green * 255), int(blue * 255)


def _point(
    center: Sequence[float],
    crop: Tuple[int, int, int, int],
    size: int,
    margin: int,
) -> Tuple[float, float]:
    x0, y0, x1, y1 = crop
    return (
        margin + (float(center[0]) - x0) * (size - 2 * margin) / max(x1 - x0, 1),
        margin + (float(center[1]) - y0) * (size - 2 * margin) / max(y1 - y0, 1),
    )


def _minimum_pair_distance(
    positions: Mapping[int, Tuple[float, float]],
) -> float:
    nodes = sorted(positions)
    if len(nodes) < 2:
        return float("inf")
    return min(
        math.hypot(
            positions[left][0] - positions[right][0],
            positions[left][1] - positions[right][1],
        )
        for index, left in enumerate(nodes)
        for right in nodes[index + 1 :]
    )


def _spread_positions(
    positions: Mapping[int, Tuple[float, float]],
    *,
    size: int,
    margin: int,
    min_distance: float,
) -> Dict[int, Tuple[float, float]]:
    """Resolve collisions while retaining the projected spatial layout."""

    nodes = sorted(positions)
    if len(nodes) < 2 or min_distance <= 0:
        return {node: tuple(positions[node]) for node in nodes}
    anchors = {node: tuple(positions[node]) for node in nodes}
    current = {node: [float(value) for value in positions[node]] for node in nodes}
    lower, upper = float(margin), float(size - margin)

    def collision_pass(anchor_strength: float) -> int:
        displacement = {node: [0.0, 0.0] for node in nodes}
        conflicts = 0
        for left_index, left in enumerate(nodes):
            for right in nodes[left_index + 1 :]:
                dx = current[right][0] - current[left][0]
                dy = current[right][1] - current[left][1]
                distance = math.hypot(dx, dy)
                if distance >= min_distance:
                    continue
                conflicts += 1
                if distance < 1e-6:
                    angle = ((left * 92821 + right * 68917) % 360) * math.pi / 180
                    ux, uy = math.cos(angle), math.sin(angle)
                else:
                    ux, uy = dx / distance, dy / distance
                push = 0.52 * (min_distance - distance + 0.25)
                displacement[left][0] -= ux * push
                displacement[left][1] -= uy * push
                displacement[right][0] += ux * push
                displacement[right][1] += uy * push
        max_step = 0.28 * min_distance
        for node in nodes:
            displacement[node][0] += anchor_strength * (
                anchors[node][0] - current[node][0]
            )
            displacement[node][1] += anchor_strength * (
                anchors[node][1] - current[node][1]
            )
            length = math.hypot(*displacement[node])
            if length > max_step:
                displacement[node][0] *= max_step / length
                displacement[node][1] *= max_step / length
            current[node][0] = min(
                upper, max(lower, current[node][0] + displacement[node][0])
            )
            current[node][1] = min(
                upper, max(lower, current[node][1] + displacement[node][1])
            )
        return conflicts

    # Preserve the geographical layout first, then finish with pure collision
    # resolution so the requested minimum spacing is actually enforced.
    for iteration in range(180):
        strength = 0.018 * max(0.0, 1.0 - iteration / 140.0)
        if collision_pass(strength) == 0:
            break
    for _ in range(80):
        if collision_pass(0.0) == 0:
            break
    return {node: (values[0], values[1]) for node, values in current.items()}


def _hub_aware_positions(
    projected: Mapping[int, Tuple[float, float]],
    directed_pairs: Iterable[Pair],
    *,
    size: int,
    margin: int,
    min_distance: float,
    star_ratio: float = 0.90,
) -> Tuple[Dict[int, Tuple[float, float]], Dict[str, object]]:
    """Locally open crowded hub sectors without replacing spatial layout."""

    nodes = sorted(projected)
    spread = _spread_positions(
        projected,
        size=size,
        margin=margin,
        min_distance=min_distance,
    )
    adjacency: Dict[int, Set[int]] = {node: set() for node in nodes}
    for subj, obj in set(directed_pairs):
        if subj == obj or subj not in adjacency or obj not in adjacency:
            continue
        adjacency[subj].add(obj)
        adjacency[obj].add(subj)
    if len(nodes) < 3:
        return spread, {"mode": "spatial_relaxed", "hub": None, "hub_ratio": 0.0}

    hub = max(nodes, key=lambda node: (len(adjacency[node]), -node))
    hub_ratio = len(adjacency[hub]) / max(len(nodes) - 1, 1)
    if len(adjacency[hub]) < 8 or hub_ratio < star_ratio:
        return spread, {
            "mode": "spatial_relaxed",
            "hub": int(hub),
            "hub_ratio": hub_ratio,
        }

    hub_origin = spread[hub]
    angular_neighbors = sorted(
        (
            math.atan2(
                spread[node][1] - hub_origin[1],
                spread[node][0] - hub_origin[0],
            )
            % (2.0 * math.pi),
            node,
        )
        for node in adjacency[hub]
    )
    # Cut at the largest empty sector, then resolve only angular collisions.
    # Each node is allowed to move at most 32 logical pixels along its original
    # hub-centered circle, so the correspondence to the satellite crop remains.
    gaps = []
    for index, (angle, _) in enumerate(angular_neighbors):
        next_angle = angular_neighbors[(index + 1) % len(angular_neighbors)][0]
        if index == len(angular_neighbors) - 1:
            next_angle += 2.0 * math.pi
        gaps.append((next_angle - angle, index))
    _, cut_index = max(gaps)
    ordered = (
        angular_neighbors[cut_index + 1 :] + angular_neighbors[: cut_index + 1]
    )
    angles: List[float] = []
    radii: List[float] = []
    previous = None
    for angle, node in ordered:
        if previous is not None:
            while angle <= previous:
                angle += 2.0 * math.pi
        angles.append(angle)
        radii.append(max(math.dist(spread[node], hub_origin), min_distance))
        previous = angle
    original_angles = list(angles)
    min_angle = math.radians(6.0)
    max_tangent_shift = 32.0 * (size / 700.0)
    lower = [
        angle - min(math.radians(22.0), max_tangent_shift / radius)
        for angle, radius in zip(original_angles, radii)
    ]
    upper = [
        angle + min(math.radians(22.0), max_tangent_shift / radius)
        for angle, radius in zip(original_angles, radii)
    ]
    for _ in range(120):
        conflicts = 0
        for index in range(len(angles) - 1):
            deficit = min_angle - (angles[index + 1] - angles[index])
            if deficit <= 1e-6:
                continue
            conflicts += 1
            left_room = angles[index] - lower[index]
            right_room = upper[index + 1] - angles[index + 1]
            move_left = min(deficit / 2.0, left_room)
            move_right = min(deficit - move_left, right_room)
            remaining = deficit - move_left - move_right
            if remaining > 0:
                extra_left = min(remaining, left_room - move_left)
                move_left += extra_left
                remaining -= extra_left
            if remaining > 0:
                move_right += min(remaining, right_room - move_right)
            angles[index] -= move_left
            angles[index + 1] += move_right
        if conflicts == 0:
            break

    relaxed = dict(spread)
    for angle, radius, (_, node) in zip(angles, radii, ordered):
        relaxed[node] = (
            hub_origin[0] + radius * math.cos(angle),
            hub_origin[1] + radius * math.sin(angle),
        )
    relaxed = _spread_positions(
        relaxed,
        size=size,
        margin=margin,
        min_distance=min_distance,
    )
    displacements = [
        math.dist(projected[node], relaxed[node]) for node in nodes
    ]
    return relaxed, {
        "mode": "spatial_hub_relaxed",
        "hub": int(hub),
        "hub_ratio": hub_ratio,
        "max_displacement": max(displacements, default=0.0),
        "mean_displacement": (
            sum(displacements) / len(displacements) if displacements else 0.0
        ),
    }


def _adjust_image_440_layout(
    positions: Mapping[int, Tuple[float, float]],
    *,
    size: int,
    margin: int,
    min_distance: float,
) -> Tuple[Dict[int, Tuple[float, float]], Dict[str, object]]:
    """Open image 440's two hubs and its crowded upper/left sectors.

    This is a visualization-only adjustment shared by the GT, PPG, and RSGP
    panels. The right sector is intentionally frozen to preserve the already
    readable part of the spatial layout.
    """

    first_hub, second_hub = 231, 238
    if first_hub not in positions or second_hub not in positions:
        return dict(positions), {"applied": False}

    adjusted = dict(positions)
    first = positions[first_hub]
    second = positions[second_hub]
    midpoint = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
    hub_dx, hub_dy = first[0] - second[0], first[1] - second[1]
    hub_distance = max(math.hypot(hub_dx, hub_dy), 1.0)
    logical_scale = size / 700.0
    vertical_sign = 1.0 if hub_dy >= 0.0 else -1.0
    hub_shift = 10.0 * logical_scale
    adjusted[first_hub] = (
        first[0],
        first[1] + vertical_sign * hub_shift,
    )

    sectors: Dict[str, List[int]] = {
        "right_frozen": [],
        "upper": [],
        "left": [],
    }
    for node, point in positions.items():
        if node in (first_hub, second_hub):
            continue
        dx, dy = point[0] - midpoint[0], point[1] - midpoint[1]
        radius = math.hypot(dx, dy)
        if radius <= 1e-6:
            continue
        # Freeze the right-facing fan, including its shallow upper/lower arms.
        if dx >= 0.0 and abs(dx) >= 0.55 * abs(dy):
            sectors["right_frozen"].append(node)
            continue
        if dy < 0.0:
            sectors["upper"].append(node)
        else:
            sectors["left"].append(node)

    lower, upper = float(margin), float(size - margin)

    def bounded_scale(
        values: Sequence[float], center: float, desired: float
    ) -> float:
        feasible = float(desired)
        for value in values:
            delta = value - center
            if delta > 1e-6:
                feasible = min(feasible, (upper - center) / delta)
            elif delta < -1e-6:
                feasible = min(feasible, (lower - center) / delta)
        return max(1.0, feasible)

    upper_nodes = sectors["upper"]
    upper_center_x = (
        sum(positions[node][0] for node in upper_nodes) / len(upper_nodes)
        if upper_nodes
        else midpoint[0]
    )
    upper_horizontal_scale = bounded_scale(
        [positions[node][0] for node in upper_nodes], upper_center_x, 1.10
    )
    upper_shift = min(
        8.0 * logical_scale,
        min((positions[node][1] - lower for node in upper_nodes), default=0.0),
    )
    for node in upper_nodes:
        point = positions[node]
        adjusted[node] = (
            upper_center_x + (point[0] - upper_center_x) * upper_horizontal_scale,
            point[1] - upper_shift,
        )

    left_nodes = sectors["left"]
    left_center_y = (
        sum(positions[node][1] for node in left_nodes) / len(left_nodes)
        if left_nodes
        else midpoint[1]
    )
    left_vertical_scale = bounded_scale(
        [positions[node][1] for node in left_nodes], left_center_y, 1.10
    )
    left_shift = min(
        6.0 * logical_scale,
        min((positions[node][0] - lower for node in left_nodes), default=0.0),
    )
    for node in left_nodes:
        point = positions[node]
        adjusted[node] = (
            point[0] - left_shift,
            left_center_y + (point[1] - left_center_y) * left_vertical_scale,
        )

    before_collision_resolution = dict(adjusted)
    adjusted = _spread_positions(
        adjusted,
        size=size,
        margin=margin,
        min_distance=min_distance,
    )
    right_displacements = [
        math.dist(before_collision_resolution[node], adjusted[node])
        for node in sectors["right_frozen"]
    ]

    return adjusted, {
        "applied": True,
        "hub_nodes": [first_hub, second_hub],
        "hub_distance_before": hub_distance,
        "hub_distance_after": math.dist(
            adjusted[first_hub], adjusted[second_hub]
        ),
        "hub_shift_logical": {
            str(first_hub): hub_shift / logical_scale,
            str(second_hub): 0.0,
        },
        "hub_separation_axis": "lower_hub_vertical",
        "upper_horizontal_scale": upper_horizontal_scale,
        "upper_shift_logical": upper_shift / logical_scale,
        "left_vertical_scale": left_vertical_scale,
        "left_shift_logical": left_shift / logical_scale,
        "sector_counts": {name: len(nodes) for name, nodes in sectors.items()},
        "right_sector_frozen": True,
        "right_sector_max_collision_correction_logical": (
            max(right_displacements, default=0.0) / logical_scale
        ),
    }


def _build_pair_route_curvatures(
    pairs: Iterable[Pair], *, spacing: float
) -> Dict[Pair, float]:
    """Separate opposite directions after predicates are collapsed by pair."""

    groups: Dict[Pair, List[Pair]] = {}
    for pair in sorted(set(pairs)):
        subj, obj = pair
        if subj == obj:
            continue
        groups.setdefault((min(subj, obj), max(subj, obj)), []).append(pair)
    routes: Dict[Pair, float] = {}
    for (first, _), group in sorted(groups.items()):
        global_lanes = [
            (index - (len(group) - 1) / 2.0) * spacing
            for index in range(len(group))
        ]
        for pair, global_lane in zip(sorted(group), global_lanes):
            routes[pair] = global_lane if pair[0] == first else -global_lane
    return routes


def _displayed_prediction_pairs(
    predictions: Sequence[Mapping[str, object]],
    local_gt: Sequence[Triple],
    focus: Sequence[Mapping[str, object]],
    *,
    unmatched_limit: int,
) -> Set[Pair]:
    """Return exactly the directed pairs that a prediction panel will draw.

    Curvature must be derived from visible edges only. Using every local
    top-k prediction can bend an otherwise isolated visible edge merely
    because its hidden reverse direction exists below the unmatched-output
    display cap.
    """

    local_gt_set = set(local_gt)
    output_pairs: Set[Pair] = set()
    rows_by_pair: Dict[Pair, List[Mapping[str, object]]] = {}
    matched_pairs: Set[Pair] = set()
    for row in predictions:
        triple = _triple(row)
        pair = (triple[0], triple[1])
        output_pairs.add(pair)
        rows_by_pair.setdefault(pair, []).append(row)
        if triple in local_gt_set:
            matched_pairs.add(pair)

    unmatched_pairs = [
        pair for pair in rows_by_pair if pair not in matched_pairs
    ]
    focus_pairs = {
        (int(row["triple"][0]), int(row["triple"][1])) for row in focus
    }
    missing_focus_pairs = {pair for pair in focus_pairs if pair not in output_pairs}
    return (
        matched_pairs
        | set(unmatched_pairs[: max(int(unmatched_limit), 0)])
        | missing_focus_pairs
    )


def _quadratic_point(
    p0: Tuple[float, float],
    control: Tuple[float, float],
    p1: Tuple[float, float],
    t: float,
) -> Tuple[float, float]:
    one_minus_t = 1.0 - t
    return (
        one_minus_t * one_minus_t * p0[0]
        + 2.0 * one_minus_t * t * control[0]
        + t * t * p1[0],
        one_minus_t * one_minus_t * p0[1]
        + 2.0 * one_minus_t * t * control[1]
        + t * t * p1[1],
    )


def _dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Tuple[float, float]],
    color: RGB,
    *,
    width: int,
    dash: float,
    gap: float,
) -> None:
    on, remaining = True, dash
    for start, end in zip(points, points[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        segment_length = math.hypot(dx, dy)
        if segment_length <= 1e-6:
            continue
        consumed = 0.0
        while consumed < segment_length:
            step = min(remaining, segment_length - consumed)
            t0, t1 = consumed / segment_length, (consumed + step) / segment_length
            if on:
                draw.line(
                    [
                        (start[0] + dx * t0, start[1] + dy * t0),
                        (start[0] + dx * t1, start[1] + dy * t1),
                    ],
                    fill=color,
                    width=width,
                )
            consumed += step
            remaining -= step
            if remaining <= 1e-6:
                on = not on
                remaining = dash if on else gap


def _arrow(
    draw: ImageDraw.ImageDraw,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    color: RGB,
    *,
    width: int,
    radius: float,
    dashed: bool = False,
    crossed: bool = False,
    curvature: float = 0.0,
    scale: int = 1,
) -> Tuple[float, float]:
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    distance = max(math.hypot(dx, dy), 1.0)
    ux, uy = dx / distance, dy / distance
    nx, ny = -uy, ux
    control = (
        (p0[0] + p1[0]) / 2.0 + nx * 2.0 * curvature,
        (p0[1] + p1[1]) / 2.0 + ny * 2.0 * curvature,
    )
    trim = min(radius / distance, 0.38)
    sample_count = 36
    points = [
        _quadratic_point(
            p0,
            control,
            p1,
            trim + (1.0 - 2.0 * trim) * index / sample_count,
        )
        for index in range(sample_count + 1)
    ]
    if dashed:
        _dashed_polyline(
            draw,
            points,
            color,
            width=width,
            dash=9 * scale,
            gap=6 * scale,
        )
    else:
        draw.line(points, fill=color, width=width, joint="curve")
    start, end = points[0], points[-1]
    tangent_x, tangent_y = end[0] - points[-2][0], end[1] - points[-2][1]
    tangent_distance = max(math.hypot(tangent_x, tangent_y), 1.0)
    ux, uy = tangent_x / tangent_distance, tangent_y / tangent_distance
    nx, ny = -uy, ux
    arrow_size = 7 * scale + width
    triangle = [
        end,
        (
            end[0] - ux * arrow_size + nx * arrow_size * 0.55,
            end[1] - uy * arrow_size + ny * arrow_size * 0.55,
        ),
        (
            end[0] - ux * arrow_size - nx * arrow_size * 0.55,
            end[1] - uy * arrow_size - ny * arrow_size * 0.55,
        ),
    ]
    draw.polygon(triangle, fill=color)
    midpoint = _quadratic_point(p0, control, p1, 0.5)
    if crossed:
        # A screen-aligned bare X can resemble an unrelated perpendicular edge
        # when the arrow happens to share one of its diagonals. Isolate the
        # symbol with a white circular knockout and rotate the X in the local
        # tangent/normal frame of the relation edge.
        marker_radius = 8 * scale
        draw.ellipse(
            (
                midpoint[0] - marker_radius,
                midpoint[1] - marker_radius,
                midpoint[0] + marker_radius,
                midpoint[1] + marker_radius,
            ),
            fill=WHITE,
            outline=color,
            width=2 * scale,
        )
        chord_x, chord_y = p1[0] - p0[0], p1[1] - p0[1]
        chord_length = max(math.hypot(chord_x, chord_y), 1.0)
        tangent = (chord_x / chord_length, chord_y / chord_length)
        normal = (-tangent[1], tangent[0])
        diagonal_a = (
            (tangent[0] + normal[0]) / math.sqrt(2.0),
            (tangent[1] + normal[1]) / math.sqrt(2.0),
        )
        diagonal_b = (
            (tangent[0] - normal[0]) / math.sqrt(2.0),
            (tangent[1] - normal[1]) / math.sqrt(2.0),
        )
        marker = 4.5 * scale
        draw.line(
            [
                (
                    midpoint[0] - diagonal_a[0] * marker,
                    midpoint[1] - diagonal_a[1] * marker,
                ),
                (
                    midpoint[0] + diagonal_a[0] * marker,
                    midpoint[1] + diagonal_a[1] * marker,
                ),
            ],
            fill=color,
            width=2 * scale,
        )
        draw.line(
            [
                (
                    midpoint[0] - diagonal_b[0] * marker,
                    midpoint[1] - diagonal_b[1] * marker,
                ),
                (
                    midpoint[0] + diagonal_b[0] * marker,
                    midpoint[1] + diagonal_b[1] * marker,
                ),
            ],
            fill=color,
            width=2 * scale,
        )
    return midpoint


def _local_predictions(
    predictions: Sequence[Mapping[str, object]], nodes: Set[int]
) -> List[Mapping[str, object]]:
    return [
        row
        for row in predictions
        if int(row["pair"][0]) in nodes and int(row["pair"][1]) in nodes
    ]


def _triple(row: Mapping[str, object]) -> Triple:
    return int(row["pair"][0]), int(row["pair"][1]), int(row["predicate"])


def _fit_node_label_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    base_logical_size: int,
    radius: int,
    scale: int,
):
    """Fit a stroked node ID inside a fixed-radius node circle."""

    stroke_width = max(int(scale), 1)
    max_extent = max(2 * int(radius) - 3 * int(scale), 4 * int(scale))
    minimum_size = 6
    for logical_size in range(int(base_logical_size), minimum_size - 1, -1):
        font = q._font(logical_size * scale)
        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font,
            stroke_width=stroke_width,
        )
        if bbox[2] - bbox[0] <= max_extent and bbox[3] - bbox[1] <= max_extent:
            return font, bbox
    font = q._font(minimum_size * scale)
    return font, draw.textbbox(
        (0, 0),
        text,
        font=font,
        stroke_width=stroke_width,
    )


def _draw_nodes(
    draw: ImageDraw.ImageDraw,
    nodes: Sequence[int],
    positions: Mapping[int, Tuple[float, float]],
    labels: torch.Tensor,
    *,
    radius: int,
    scale: int,
) -> None:
    base_logical_size = 10 if len(nodes) > 35 else 12
    for node in nodes:
        x, y = positions[node]
        label = int(labels[node]) if node < len(labels) else -1
        color = _class_color(label)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline=BLACK,
            width=2 * scale,
        )
        text = str(node)
        if len(nodes) > 35:
            font, bbox = _fit_node_label_font(
                draw,
                text,
                base_logical_size=base_logical_size,
                radius=radius,
                scale=scale,
            )
        else:
            # Preserve the established appearance of sparse cases such as 748,
            # whose larger node circles already contain their labels.
            font = q._font(base_logical_size * scale)
            bbox = draw.textbbox(
                (0, 0), text, font=font, stroke_width=scale
            )
        draw.text(
            (x - (bbox[0] + bbox[2]) / 2, y - (bbox[1] + bbox[3]) / 2),
            text,
            fill=WHITE,
            font=font,
            stroke_width=scale,
            stroke_fill=BLACK,
        )


def _render_image_panel(
    source: Image.Image,
    artifact: Mapping[str, object],
    crop: Tuple[int, int, int, int],
    nodes: Sequence[int],
    *,
    size: int,
    scale: int,
) -> Image.Image:
    panel = source.crop(crop).resize((size, size), Image.Resampling.LANCZOS)
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    target = artifact["target"]
    boxes = q._scale_boxes_to_raw(artifact)
    centers = q._centers(boxes, str(target["mode"]))
    labels = q._tensor(target, "labels", (0,)).long().reshape(-1)
    class_names = artifact.get("class_names", [])
    angle_unit = str(target.get("box_angle_unit", "radian"))
    for node in nodes:
        if str(target["mode"]) == "xywha":
            polygon = [
                q._crop_point(point, crop, size)
                for point in q._obb_polygon(boxes[node].tolist(), angle_unit)
            ]
            draw.line(
                polygon + [polygon[0]],
                fill=(*ORANGE, 245),
                width=3 * scale,
            )
        center = q._crop_point(centers[node], crop, size)
        label = int(labels[node]) if node < len(labels) else -1
        name = str(class_names[label]) if 0 <= label < len(class_names) else str(label)
        text = f"{node}:{name}"
        font = q._font((10 if len(nodes) > 35 else 12) * scale)
        text_bbox = draw.textbbox(
            (0, 0), text, font=font, stroke_width=2 * scale
        )
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = min(
            max(center[0] + 3 * scale, 3 * scale),
            size - text_width - 5 * scale,
        )
        text_y = min(
            max(center[1] + 3 * scale, 3 * scale),
            size - text_height - 5 * scale,
        )
        draw.text(
            (text_x, text_y),
            text,
            fill=(*BLACK, 255),
            font=font,
            stroke_width=2 * scale,
            stroke_fill=(255, 255, 255, 235),
        )
    return Image.alpha_composite(panel.convert("RGBA"), overlay).convert("RGB")


def _render_graph_panel(
    *,
    kind: str,
    nodes: Sequence[int],
    positions: Mapping[int, Tuple[float, float]],
    labels: torch.Tensor,
    local_gt: Sequence[Triple],
    predictions: Sequence[Mapping[str, object]],
    focus: Sequence[Mapping[str, object]],
    route_curvatures: Mapping[Pair, float],
    role: str,
    size: int,
    unmatched_limit: int,
    scale: int,
) -> Tuple[Image.Image, Dict[str, int]]:
    panel = Image.new("RGB", (size, size), (249, 250, 252))
    draw = ImageDraw.Draw(panel)
    radius = (10 if len(nodes) > 35 else 13) * scale
    local_gt_set = set(local_gt)
    focus_triples = {tuple(row["triple"]) for row in focus}
    focus_pairs = {(triple[0], triple[1]) for triple in focus_triples}
    focus_color = BLUE if role == "success" else PURPLE
    edge_width = 2 * scale

    if kind == "GT":
        # Predicate semantics are not annotated in this structural figure, so
        # multiple predicates on one directed entity pair share one arrow.
        gt_pairs = {(subj, obj) for subj, obj, _ in local_gt_set}
        # Draw focus edges last so the comparison remains visible without
        # changing the style or omitting any GT relation.
        ordered_pairs = sorted(gt_pairs - focus_pairs) + sorted(gt_pairs & focus_pairs)
        for pair in ordered_pairs:
            subj, obj = pair
            is_focus = pair in focus_pairs
            _arrow(
                draw,
                positions[subj],
                positions[obj],
                focus_color if is_focus else DARK_GRAY,
                width=edge_width,
                radius=radius,
                curvature=route_curvatures.get(pair, 0.0),
                scale=scale,
            )
        summary = {
            "local_output_count": 0,
            "local_matched_output_count": len(local_gt),
            "displayed_unmatched_output_count": 0,
        }
    else:
        exact_predictions = {_triple(row) for row in predictions}
        output_pairs = {(triple[0], triple[1]) for triple in exact_predictions}
        rows_by_pair: Dict[Pair, List[Mapping[str, object]]] = {}
        for row in predictions:
            triple = _triple(row)
            rows_by_pair.setdefault((triple[0], triple[1]), []).append(row)
        matched = [row for row in predictions if _triple(row) in local_gt_set]
        matched_pairs = {
            (_triple(row)[0], _triple(row)[1]) for row in matched
        }
        unmatched_pairs = [
            pair for pair in rows_by_pair if pair not in matched_pairs
        ]
        # GT-unmatched output pairs are capped and drawn behind correct pairs.
        for pair in unmatched_pairs[: max(int(unmatched_limit), 0)]:
            subj, obj = pair
            _arrow(
                draw,
                positions[subj],
                positions[obj],
                RED,
                width=edge_width,
                radius=radius,
                dashed=True,
                curvature=route_curvatures.get(pair, 0.0),
                scale=scale,
            )
        for pair in sorted(matched_pairs):
            subj, obj = pair
            pair_triples = {_triple(row) for row in rows_by_pair[pair]}
            color = focus_color if pair_triples & focus_triples else GREEN
            _arrow(
                draw,
                positions[subj],
                positions[obj],
                color,
                width=edge_width,
                radius=radius,
                curvature=route_curvatures.get(pair, 0.0),
                scale=scale,
            )
        # With predicate text omitted, a comparison reference is meaningful
        # only when its directed pair is absent entirely from this output.
        for pair in sorted(focus_pairs):
            if pair in output_pairs:
                continue
            subj, obj = pair
            _arrow(
                draw,
                positions[subj],
                positions[obj],
                focus_color,
                width=edge_width,
                radius=radius,
                dashed=True,
                crossed=True,
                curvature=route_curvatures.get(pair, 0.0),
                scale=scale,
            )
        summary = {
            "local_output_count": len(predictions),
            "local_matched_output_count": len(matched),
            "displayed_unmatched_output_count": min(
                len(unmatched_pairs), unmatched_limit
            ),
            "displayed_directed_pair_count": len(matched_pairs)
            + min(len(unmatched_pairs), unmatched_limit),
        }

    _draw_nodes(draw, nodes, positions, labels, radius=radius, scale=scale)
    return panel, summary


def _select_focus(
    *,
    gt: torch.Tensor,
    local_gt_indices: Set[int],
    priority_indices: Set[int],
    source_predictions: Sequence[Mapping[str, object]],
    source_matches: Sequence[Set[int]],
    predicate_names: Sequence[str],
    ppg_triples: Set[Triple],
    rsgp_triples: Set[Triple],
    limit: int,
) -> List[Dict[str, object]]:
    selected: List[int] = []
    for index, _ in enumerate(source_predictions):
        for gt_index in sorted(
            source_matches[index] & priority_indices & local_gt_indices
        ):
            if gt_index not in selected:
                selected.append(gt_index)
                if len(selected) >= limit:
                    break
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for gt_index in sorted(priority_indices & local_gt_indices):
            if gt_index not in selected:
                selected.append(gt_index)
                if len(selected) >= limit:
                    break
    rows: List[Dict[str, object]] = []
    seen: Set[Triple] = set()
    for gt_index in selected:
        triple = tuple(int(v) for v in gt[gt_index].tolist())
        if triple in seen:
            continue
        seen.add(triple)
        predicate = triple[2]
        predicate_name = (
            str(predicate_names[predicate])
            if 0 <= predicate < len(predicate_names)
            else f"r{predicate}"
        )
        rows.append(
            {
                "tag": f"F{len(rows) + 1}",
                "gt_index": int(gt_index),
                "triple": list(triple),
                "predicate_name": predicate_name,
                "present_in_ppg": triple in ppg_triples,
                "present_in_rsgp": triple in rsgp_triples,
            }
        )
    return rows


def render_spatial_scene_graph_case(
    ppg: Mapping[str, object],
    rsgp: Mapping[str, object],
    output_path: Path,
    *,
    topk: int = 2000,
    panel_size: int = 700,
    render_scale: int = 2,
    gutter: int = 24,
    min_node_spacing: int = 58,
    edge_route_spacing: int = 16,
    crop_fraction: float = 0.10,
    crop_override: Tuple[int, int, int, int] | None = None,
    focus_limit: int = 6,
    unmatched_limit: int = 8,
    require_active_filtering: bool = True,
) -> Dict[str, object]:
    if int(ppg["image_id"]) != int(rsgp["image_id"]):
        raise ValueError("PPG and RSGP artifacts must describe the same image")
    filter_audit: Dict[str, Dict[str, object]] = {}
    for expected, artifact in (("PPG", ppg), ("RSGP", rsgp)):
        prediction = artifact["prediction"]
        semantic = int(q._tensor(prediction, "sema_rel_pair_idxs", (0, 2)).reshape(-1, 2).size(0))
        final = int(q._tensor(prediction, "rel_pair_idxs", (0, 2)).reshape(-1, 2).size(0))
        actual = str(artifact.get("filter_method", "")).upper()
        active = semantic > final > 0
        filter_audit[expected.lower()] = {
            "declared_method": actual,
            "semantic_candidate_count": semantic,
            "final_candidate_count": final,
            "actively_filtered": active,
        }
        if require_active_filtering and (actual != expected or not active):
            raise ValueError(
                f"Image {artifact['image_id']} does not exercise {expected}: "
                f"method={actual!r}, semantic={semantic}, final={final}"
            )

    ppg_predictions = q.graph_predictions(ppg, topk)
    rsgp_predictions = q.graph_predictions(rsgp, topk)
    ppg_matches_by_prediction = q.prediction_gt_matches(ppg, ppg_predictions)
    rsgp_matches_by_prediction = q.prediction_gt_matches(rsgp, rsgp_predictions)
    ppg_matches = set().union(*ppg_matches_by_prediction) if ppg_matches_by_prediction else set()
    rsgp_matches = set().union(*rsgp_matches_by_prediction) if rsgp_matches_by_prediction else set()
    rescued, lost = rsgp_matches - ppg_matches, ppg_matches - rsgp_matches
    gt = q._tensor(ppg["target"], "relation_triplets", (0, 3)).long().reshape(-1, 3)
    gt_pairs = {tuple(int(v) for v in row[:2]) for row in gt.tolist()}
    denominator = max(len(gt_pairs), 1)
    ppg_recall, rsgp_recall = len(ppg_matches) / denominator, len(rsgp_matches) / denominator
    role = "success" if rsgp_recall > ppg_recall else ("failure" if rsgp_recall < ppg_recall else "control")
    priority = rescued if role == "success" else (lost if role == "failure" else ppg_matches | rsgp_matches)
    crop = crop_override or q.choose_crop(ppg, priority, crop_fraction=crop_fraction)

    target = ppg["target"]
    boxes = q._scale_boxes_to_raw(ppg)
    centers = q._centers(boxes, str(target["mode"]))
    labels = q._tensor(target, "labels", (0,)).long().reshape(-1)
    nodes = sorted(index for index, center in enumerate(centers) if q._inside(center, crop))
    node_set = set(nodes)
    local_gt_indices = {
        index
        for index, row in enumerate(gt.tolist())
        if int(row[0]) in node_set and int(row[1]) in node_set
    }
    local_gt = [tuple(int(v) for v in gt[index].tolist()) for index in sorted(local_gt_indices)]
    local_predicate_ids = sorted({predicate for _, _, predicate in local_gt})
    local_ppg = _local_predictions(ppg_predictions, node_set)
    local_rsgp = _local_predictions(rsgp_predictions, node_set)
    ppg_triples, rsgp_triples = {_triple(row) for row in local_ppg}, {_triple(row) for row in local_rsgp}
    source_predictions = rsgp_predictions if role == "success" else ppg_predictions
    source_matches = rsgp_matches_by_prediction if role == "success" else ppg_matches_by_prediction
    focus = _select_focus(
        gt=gt,
        local_gt_indices=local_gt_indices,
        priority_indices=priority,
        source_predictions=source_predictions,
        source_matches=source_matches,
        predicate_names=ppg.get("predicate_names", []),
        ppg_triples=ppg_triples,
        rsgp_triples=rsgp_triples,
        limit=focus_limit,
    )

    scale = max(int(render_scale), 1)
    work_panel_size = int(panel_size) * scale
    margin = 42 * scale
    projected_positions = {
        node: _point(centers[node], crop, work_panel_size, margin) for node in nodes
    }
    gt_directed_pairs = {(subj, obj) for subj, obj, _ in local_gt}
    positions, layout_strategy = _hub_aware_positions(
        projected_positions,
        gt_directed_pairs,
        size=work_panel_size,
        margin=margin,
        min_distance=max(int(min_node_spacing), 0) * scale,
    )
    case_layout_adjustment: Dict[str, object] = {"applied": False}
    if int(ppg["image_id"]) == 440:
        positions, case_layout_adjustment = _adjust_image_440_layout(
            positions,
            size=work_panel_size,
            margin=margin,
            min_distance=max(int(min_node_spacing), 0) * scale,
        )
    route_spacing = max(int(edge_route_spacing), 0) * scale
    # Route each panel from only the edges it actually displays. A reverse
    # prediction in another panel must not bend an otherwise straight GT edge.
    gt_route_curvatures = _build_pair_route_curvatures(
        gt_directed_pairs,
        spacing=0.5 * route_spacing,
    )
    ppg_display_pairs = _displayed_prediction_pairs(
        local_ppg,
        local_gt,
        focus,
        unmatched_limit=unmatched_limit,
    )
    rsgp_display_pairs = _displayed_prediction_pairs(
        local_rsgp,
        local_gt,
        focus,
        unmatched_limit=unmatched_limit,
    )
    ppg_route_curvatures = _build_pair_route_curvatures(
        ppg_display_pairs,
        spacing=0.75 * route_spacing,
    )
    rsgp_route_curvatures = _build_pair_route_curvatures(
        rsgp_display_pairs,
        spacing=0.75 * route_spacing,
    )
    with Image.open(str(ppg["image_path"])) as raw:
        source = raw.convert("RGB")
    image_panel = _render_image_panel(
        source,
        ppg,
        crop,
        nodes,
        size=work_panel_size,
        scale=scale,
    )
    gt_panel, gt_summary = _render_graph_panel(
        kind="GT",
        nodes=nodes,
        positions=positions,
        labels=labels,
        local_gt=local_gt,
        predictions=(),
        focus=focus,
        route_curvatures=gt_route_curvatures,
        role=role,
        size=work_panel_size,
        unmatched_limit=unmatched_limit,
        scale=scale,
    )
    ppg_panel, ppg_summary = _render_graph_panel(
        kind="PPG",
        nodes=nodes,
        positions=positions,
        labels=labels,
        local_gt=local_gt,
        predictions=local_ppg,
        focus=focus,
        route_curvatures=ppg_route_curvatures,
        role=role,
        size=work_panel_size,
        unmatched_limit=unmatched_limit,
        scale=scale,
    )
    rsgp_panel, rsgp_summary = _render_graph_panel(
        kind="RSGP",
        nodes=nodes,
        positions=positions,
        labels=labels,
        local_gt=local_gt,
        predictions=local_rsgp,
        focus=focus,
        route_curvatures=rsgp_route_curvatures,
        role=role,
        size=work_panel_size,
        unmatched_limit=unmatched_limit,
        scale=scale,
    )

    header, footer = 70 * scale, 92 * scale
    outer = 16 * scale
    gap = max(int(gutter), 0) * scale
    figure_width = 2 * outer + 4 * work_panel_size + 3 * gap
    figure_height = header + work_panel_size + footer
    figure = Image.new("RGB", (figure_width, figure_height), (235, 239, 244))
    draw = ImageDraw.Draw(figure)
    panel_x = [
        outer + column * (work_panel_size + gap) for column in range(4)
    ]
    for x, panel in zip(panel_x, (image_panel, gt_panel, ppg_panel, rsgp_panel)):
        draw.rounded_rectangle(
            (
                x - 5 * scale,
                8 * scale,
                x + work_panel_size + 5 * scale,
                header + work_panel_size + 5 * scale,
            ),
            radius=9 * scale,
            fill=WHITE,
            outline=(150, 158, 170),
            width=2 * scale,
        )
        figure.paste(panel, (x, header))
    local_gt_pairs = {(subj, obj) for subj, obj, _ in local_gt}
    titles = (
        f"(a) image {ppg['image_id']}: {len(nodes)} GT entities",
        f"(b) GT: {len(local_gt)} relations / {len(local_gt_pairs)} pairs",
        f"(c) PPG: {ppg_summary['local_matched_output_count']}/{ppg_summary['local_output_count']} matched",
        f"(d) RSGP: {rsgp_summary['local_matched_output_count']}/{rsgp_summary['local_output_count']} matched",
    )
    for x, title in zip(panel_x, titles):
        draw.text(
            (x + 12 * scale, 20 * scale),
            title,
            fill=BLACK,
            font=q._font(19 * scale),
        )
    footer_y = header + work_panel_size
    legend = (
        "same node position/ID in (b-d); one arrow per directed pair; gray: GT relation; "
        "green: correct output; blue: RSGP-rescued; purple: PPG-only; red dashed: GT-unmatched output; "
        "blue/purple dashed circled x: correct only in the other method (reference, not output)"
    )
    draw.text(
        (outer, footer_y + 16 * scale),
        legend,
        fill=BLACK,
        font=q._font(15 * scale),
    )
    stats = (
        f"actual graph-constrained top-{topk}; image-level Triplet R: "
        f"PPG={100 * ppg_recall:.2f}%, RSGP={100 * rsgp_recall:.2f}%, "
        f"delta={100 * (rsgp_recall - ppg_recall):+.2f} points; "
        f"only the top {unmatched_limit} local GT-unmatched outputs are drawn"
    )
    draw.text(
        (outer, footer_y + 44 * scale),
        stats,
        fill=BLACK,
        font=q._font(15 * scale),
    )
    predicate_names = ppg.get("predicate_names", [])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.save(output_path, dpi=(300, 300))
    return {
        "schema_version": 2,
        "layout": "spatially_aligned_scene_graph",
        "image_id": int(ppg["image_id"]),
        "role": role,
        "crop_xyxy_raw": list(crop),
        "topk": int(topk),
        "render_scale": scale,
        "pixel_size": [figure_width, figure_height],
        "layout_diagnostics": {
            "mode": layout_strategy["mode"],
            "hub_node": layout_strategy["hub"],
            "hub_ratio": layout_strategy["hub_ratio"],
            "max_projection_displacement_logical": (
                max(
                    (math.dist(projected_positions[node], positions[node]) for node in nodes),
                    default=0.0,
                )
                / scale
            ),
            "mean_projection_displacement_logical": (
                sum(
                    math.dist(projected_positions[node], positions[node]) for node in nodes
                )
                / max(len(nodes), 1)
                / scale
            ),
            "requested_min_node_spacing_logical": int(min_node_spacing),
            "achieved_min_node_spacing_logical": (
                _minimum_pair_distance(positions) / scale if len(nodes) > 1 else None
            ),
            "edge_route_spacing_logical": {
                "gt": 0.5 * int(edge_route_spacing),
                "prediction": 0.75 * int(edge_route_spacing),
            },
            "curved_directed_pairs": {
                "gt": [
                    list(pair)
                    for pair, curvature in sorted(gt_route_curvatures.items())
                    if abs(curvature) > 1e-6
                ],
                "ppg": [
                    list(pair)
                    for pair, curvature in sorted(ppg_route_curvatures.items())
                    if abs(curvature) > 1e-6
                ],
                "rsgp": [
                    list(pair)
                    for pair, curvature in sorted(rsgp_route_curvatures.items())
                    if abs(curvature) > 1e-6
                ],
            },
            "predicate_text_rendered": False,
            "adaptive_node_label_font": len(nodes) > 35,
            "node_radius_logical": 10 if len(nodes) > 35 else 13,
            "case_layout_adjustment": case_layout_adjustment,
            "node_positions_logical": {
                str(node): [positions[node][0] / scale, positions[node][1] / scale]
                for node in nodes
            },
            "parallel_predicates_collapsed_by_directed_pair": True,
            "uniform_edge_width_logical": 2,
        },
        "filter_audit": filter_audit,
        "global_triplet_recall": {"ppg": ppg_recall, "rsgp": rsgp_recall},
        "local_gt_entity_ids": nodes,
        "local_gt_entity_count": len(nodes),
        "local_gt_relation_count": len(local_gt),
        "local_gt_pair_count": len(local_gt_pairs),
        "local_gt_pairs": [list(pair) for pair in sorted(local_gt_pairs)],
        "local_gt_predicates": [
            {
                "predicate": predicate,
                "name": (
                    str(predicate_names[predicate])
                    if 0 <= predicate < len(predicate_names)
                    else f"r{predicate}"
                ),
            }
            for predicate in local_predicate_ids
        ],
        "ppg_summary": ppg_summary,
        "rsgp_summary": rsgp_summary,
        "focus_relations": focus,
        "figure": str(output_path),
    }


def _parse_overrides(value: str) -> Dict[int, Tuple[int, int, int, int]]:
    result: Dict[int, Tuple[int, int, int, int]] = {}
    if not value.strip():
        return result
    for item in value.split(";"):
        image_token, separator, crop_token = item.strip().partition(":")
        if not separator:
            raise ValueError("crop overrides use image_id:x0,y0,x1,y1")
        crop = tuple(int(token.strip()) for token in crop_token.split(","))
        if len(crop) != 4:
            raise ValueError("each crop override must contain four coordinates")
        result[int(image_token)] = crop
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppg-dir", type=Path, required=True)
    parser.add_argument("--rsgp-dir", type=Path, required=True)
    parser.add_argument("--image-ids", default="440,748,235")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=2000)
    parser.add_argument(
        "--panel-size",
        type=int,
        default=700,
        help="Logical width/height of each panel before supersampling.",
    )
    parser.add_argument(
        "--render-scale",
        type=int,
        default=2,
        help="Integer supersampling/output scale; 2 yields a paper-ready 5K PNG.",
    )
    parser.add_argument(
        "--gutter",
        type=int,
        default=24,
        help="Logical gap between adjacent panels.",
    )
    parser.add_argument(
        "--min-node-spacing",
        type=int,
        default=58,
        help="Minimum logical center distance after spatial-layout relaxation.",
    )
    parser.add_argument(
        "--edge-route-spacing",
        type=int,
        default=16,
        help="Logical separation between parallel/reverse curved edge lanes.",
    )
    parser.add_argument("--crop-fraction", type=float, default=0.10)
    parser.add_argument("--crop-overrides", default="235:6325,4650,6710,5035")
    parser.add_argument("--focus-limit", type=int, default=6)
    parser.add_argument("--unmatched-limit", type=int, default=8)
    parser.add_argument("--allow-no-truncation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = _parse_overrides(args.crop_overrides)
    rows = []
    for image_id in (
        int(token.strip()) for token in args.image_ids.split(",") if token.strip()
    ):
        ppg_path = args.ppg_dir / f"{image_id:04d}.pt"
        rsgp_path = args.rsgp_dir / f"{image_id:04d}.pt"
        if not ppg_path.is_file() or not rsgp_path.is_file():
            raise FileNotFoundError(f"Missing paired artifacts: {ppg_path}, {rsgp_path}")
        row = render_spatial_scene_graph_case(
            q.load_artifact(ppg_path),
            q.load_artifact(rsgp_path),
            args.output_dir / f"{image_id:04d}_spatial_scene_graph.png",
            topk=args.topk,
            panel_size=args.panel_size,
            render_scale=args.render_scale,
            gutter=args.gutter,
            min_node_spacing=args.min_node_spacing,
            edge_route_spacing=args.edge_route_spacing,
            crop_fraction=args.crop_fraction,
            crop_override=overrides.get(image_id),
            focus_limit=args.focus_limit,
            unmatched_limit=args.unmatched_limit,
            require_active_filtering=not args.allow_no_truncation,
        )
        rows.append(row)
        print(
            f"Rendered spatial scene graph {image_id}: role={row['role']}, "
            f"entities={row['local_gt_entity_count']}, "
            f"GT relations={row['local_gt_relation_count']}",
            flush=True,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scene_graph_manifest.json").write_text(
        json.dumps({"schema_version": 1, "cases": rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
