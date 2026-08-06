#!/usr/bin/env python3
"""Render large-image qualitative comparisons from paired PPG/RSGP artifacts.

Each output uses a full-image locator panel and two identical local crops.  The
crop is selected from method-specific final correct-triplet differences,
avoiding an unreadable full-image graph.  The local panels visualize actual
top-k model predictions rather than a candidate-coverage or GT-only graph:
matched predictions are solid and predictions unmatched by the available GT
annotations are dashed.  The JSON sidecar records every displayed prediction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont


Image.MAX_IMAGE_PIXELS = None

RGB = Tuple[int, int, int]
Pair = Tuple[int, int]
Triple = Tuple[int, int, int]

GREEN: RGB = (38, 170, 78)
BLUE: RGB = (30, 115, 230)
RED: RGB = (220, 55, 55)
GRAY: RGB = (135, 135, 135)
ORANGE: RGB = (245, 150, 35)
CYAN: RGB = (30, 195, 205)
PURPLE: RGB = (165, 80, 200)
BLACK: RGB = (20, 20, 20)
WHITE: RGB = (255, 255, 255)


def _font(size: int):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def load_artifact(path: Path) -> Dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.1
        return torch.load(path, map_location="cpu")


def _tensor(payload: Mapping[str, object], key: str, shape: Tuple[int, ...]) -> torch.Tensor:
    value = payload.get(key)
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.zeros(shape, dtype=torch.float32)


def graph_predictions(artifact: Mapping[str, object], topk: int) -> List[Dict[str, object]]:
    prediction = artifact["prediction"]
    pairs = _tensor(prediction, "rel_pair_idxs", (0, 2)).long().reshape(-1, 2)
    rel_scores = _tensor(prediction, "pred_rel_scores", (0, 1)).float()
    if pairs.numel() == 0 or rel_scores.numel() == 0 or rel_scores.size(1) <= 1:
        return []
    length = min(int(pairs.size(0)), int(rel_scores.size(0)))
    pairs = pairs[:length]
    foreground_score, foreground_idx = rel_scores[:length, 1:].max(dim=1)
    predicates = foreground_idx + 1
    obj_scores = _tensor(prediction, "pred_scores", (0,)).float().reshape(-1)
    if obj_scores.numel() > int(pairs.max().item()):
        overall = (
            foreground_score
            * obj_scores[pairs[:, 0]].clamp_min(0.0)
            * obj_scores[pairs[:, 1]].clamp_min(0.0)
        )
    else:
        overall = foreground_score
    order = torch.argsort(overall, descending=True)[: min(int(topk), length)]
    return [
        {
            "pair": (int(pairs[idx, 0]), int(pairs[idx, 1])),
            "predicate": int(predicates[idx]),
            "score": float(overall[idx]),
            "global_rank": rank + 1,
        }
        for rank, idx in enumerate(order.tolist())
    ]


def matched_gt_indices(
    artifact: Mapping[str, object], predictions: Sequence[Mapping[str, object]]
) -> Set[int]:
    target = artifact["target"]
    gt = _tensor(target, "relation_triplets", (0, 3)).long().reshape(-1, 3)
    triple_to_indices: Dict[Triple, List[int]] = {}
    for index, row in enumerate(gt.tolist()):
        triple_to_indices.setdefault(tuple(int(v) for v in row), []).append(index)
    matched: Set[int] = set()
    for row in predictions:
        pair = row["pair"]
        triple = (int(pair[0]), int(pair[1]), int(row["predicate"]))
        matched.update(triple_to_indices.get(triple, ()))
    return matched


def prediction_gt_matches(
    artifact: Mapping[str, object], predictions: Sequence[Mapping[str, object]]
) -> List[Set[int]]:
    """Map each final PredCls prediction to the GT relation rows it matches."""

    target = artifact["target"]
    gt = _tensor(target, "relation_triplets", (0, 3)).long().reshape(-1, 3)
    triple_to_indices: Dict[Triple, Set[int]] = {}
    for index, row in enumerate(gt.tolist()):
        triple_to_indices.setdefault(tuple(int(v) for v in row), set()).add(index)
    result: List[Set[int]] = []
    for row in predictions:
        pair = row["pair"]
        triple = (int(pair[0]), int(pair[1]), int(row["predicate"]))
        result.append(set(triple_to_indices.get(triple, set())))
    return result


def _scale_boxes_to_raw(artifact: Mapping[str, object]) -> torch.Tensor:
    target = artifact["target"]
    boxes = _tensor(target, "bbox", (0, 5)).float().clone()
    rel_w, rel_h = (float(v) for v in target["size"])
    raw_w = float(artifact["raw_width"])
    raw_h = float(artifact["raw_height"])
    if boxes.numel() == 0:
        return boxes
    sx, sy = raw_w / max(rel_w, 1.0), raw_h / max(rel_h, 1.0)
    if str(target["mode"]) == "xywha":
        boxes[:, 0] *= sx
        boxes[:, 1] *= sy
        boxes[:, 2] *= sx
        boxes[:, 3] *= sy
    else:
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
    return boxes


def _centers(boxes: torch.Tensor, mode: str) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 2))
    if mode == "xywha":
        return boxes[:, :2]
    return 0.5 * (boxes[:, :2] + boxes[:, 2:4])


def choose_crop(
    artifact: Mapping[str, object],
    priority_indices: Set[int],
    *,
    crop_fraction: float = 0.30,
) -> Tuple[int, int, int, int]:
    target = artifact["target"]
    gt = _tensor(target, "relation_triplets", (0, 3)).long().reshape(-1, 3)
    boxes = _scale_boxes_to_raw(artifact)
    centers = _centers(boxes, str(target["mode"]))
    raw_w, raw_h = int(artifact["raw_width"]), int(artifact["raw_height"])
    side = int(round(min(raw_w, raw_h) * float(crop_fraction)))
    side = max(min(side, raw_w, raw_h), min(1200, raw_w, raw_h))
    if gt.numel() == 0 or centers.numel() == 0:
        return ((raw_w - side) // 2, (raw_h - side) // 2, (raw_w + side) // 2, (raw_h + side) // 2)

    midpoints = 0.5 * (centers[gt[:, 0]] + centers[gt[:, 1]])
    priority = sorted(index for index in priority_indices if 0 <= index < len(gt))
    candidate_indices = priority or list(range(len(gt)))
    best = None
    half = side * 0.5
    for index in candidate_indices:
        cx, cy = (float(v) for v in midpoints[index])
        x0 = max(0.0, min(cx - half, raw_w - side))
        y0 = max(0.0, min(cy - half, raw_h - side))
        inside_all = (
            (midpoints[:, 0] >= x0)
            & (midpoints[:, 0] <= x0 + side)
            & (midpoints[:, 1] >= y0)
            & (midpoints[:, 1] <= y0 + side)
        )
        if priority:
            priority_tensor = torch.tensor(priority, dtype=torch.long)
            inside_priority = int(inside_all[priority_tensor].sum().item())
        else:
            inside_priority = int(inside_all.sum().item())
        score = 20.0 * inside_priority + float(inside_all.sum().item())
        candidate = (score, -x0 - y0, int(round(x0)), int(round(y0)))
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    x0, y0 = best[2], best[3]
    return (x0, y0, min(x0 + side, raw_w), min(y0 + side, raw_h))


def _obb_polygon(box: Sequence[float], angle_unit: str) -> List[Tuple[float, float]]:
    cx, cy, width, height, angle = (float(v) for v in box[:5])
    if str(angle_unit).lower().startswith("deg"):
        angle = math.radians(angle)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    points = []
    for dx, dy in ((-width / 2, -height / 2), (width / 2, -height / 2), (width / 2, height / 2), (-width / 2, height / 2)):
        points.append((cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a))
    return points


def _crop_point(point: Sequence[float], crop: Tuple[int, int, int, int], panel_size: int) -> Tuple[float, float]:
    x0, y0, x1, y1 = crop
    return (
        (float(point[0]) - x0) * panel_size / max(x1 - x0, 1),
        (float(point[1]) - y0) * panel_size / max(y1 - y0, 1),
    )


def _inside(point: Sequence[float], crop: Tuple[int, int, int, int]) -> bool:
    return crop[0] <= float(point[0]) <= crop[2] and crop[1] <= float(point[1]) <= crop[3]


def _dashed_line(draw: ImageDraw.ImageDraw, p0, p1, fill: RGB, width: int = 3, dash: int = 10) -> None:
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    distance = max(math.hypot(dx, dy), 1.0)
    ux, uy = dx / distance, dy / distance
    position = 0.0
    while position < distance:
        end = min(position + dash, distance)
        draw.line(
            [(p0[0] + ux * position, p0[1] + uy * position), (p0[0] + ux * end, p0[1] + uy * end)],
            fill=fill,
            width=width,
        )
        position += 2 * dash


def _final_pair_set(artifact: Mapping[str, object]) -> Set[Pair]:
    prediction = artifact["prediction"]
    pairs = _tensor(prediction, "rel_pair_idxs", (0, 2)).long().reshape(-1, 2)
    return {tuple(int(v) for v in row) for row in pairs.tolist()}


def _prediction_rows_for_crop(
    artifact: Mapping[str, object],
    crop: Tuple[int, int, int, int],
    predictions: Sequence[Mapping[str, object]],
    prediction_matches: Sequence[Set[int]],
    priority_indices: Set[int],
    limit: int,
) -> List[int]:
    """Select a readable sample of actual local top-k predictions.

    Correct predictions explain the downstream recall difference.  A quarter
    of the display budget is reserved for the highest-ranked predictions that
    are unmatched by STAR annotations, so the panel is not a GT-only recall
    visualization.  We deliberately call those outputs *unmatched* rather
    than false positives because scene-graph annotations are not exhaustive.
    """

    target = artifact["target"]
    centers = _centers(_scale_boxes_to_raw(artifact), str(target["mode"]))
    eligible = [
        index
        for index, row in enumerate(predictions)
        if int(row["pair"][0]) < len(centers)
        and int(row["pair"][1]) < len(centers)
        and _inside(centers[int(row["pair"][0])], crop)
        and _inside(centers[int(row["pair"][1])], crop)
    ]
    correct = [index for index in eligible if prediction_matches[index]]
    unmatched = [index for index in eligible if not prediction_matches[index]]
    priority_correct = [
        index
        for index in correct
        if prediction_matches[index] & priority_indices
    ]
    priority_set = set(priority_correct)
    other_correct = [index for index in correct if index not in priority_set]

    unmatched_quota = min(len(unmatched), max(2, int(limit) // 4))
    correct_quota = max(int(limit) - unmatched_quota, 0)
    selected = (priority_correct + other_correct)[:correct_quota]
    selected.extend(unmatched[:unmatched_quota])
    if len(selected) < int(limit):
        used = set(selected)
        selected.extend(index for index in eligible if index not in used)
    return selected[: int(limit)]


def _render_crop_panel(
    source: Image.Image,
    artifact: Mapping[str, object],
    crop: Tuple[int, int, int, int],
    *,
    method: str,
    predictions: Sequence[Mapping[str, object]],
    prediction_matches: Sequence[Set[int]],
    ppg_matches: Set[int],
    priority_indices: Set[int],
    comparison_lost_indices: Set[int],
    missing_reference_indices: Sequence[int],
    panel_size: int,
    edge_limit: int,
) -> Tuple[
    Image.Image,
    List[Dict[str, object]],
    Dict[str, int],
    List[Dict[str, object]],
]:
    panel = source.crop(crop).resize(
        (panel_size, panel_size), Image.Resampling.LANCZOS
    )
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    target = artifact["target"]
    boxes = _scale_boxes_to_raw(artifact)
    centers = _centers(boxes, str(target["mode"]))
    labels = _tensor(target, "labels", (0,)).long().reshape(-1)
    class_names = artifact.get("class_names", [])
    predicate_names = artifact.get("predicate_names", [])
    angle_unit = str(target.get("box_angle_unit", "radian"))
    local_indices = [
        index
        for index, row in enumerate(predictions)
        if int(row["pair"][0]) < len(centers)
        and int(row["pair"][1]) < len(centers)
        and _inside(centers[int(row["pair"][0])], crop)
        and _inside(centers[int(row["pair"][1])], crop)
    ]
    rows = _prediction_rows_for_crop(
        artifact,
        crop,
        predictions,
        prediction_matches,
        priority_indices,
        edge_limit,
    )

    displayed = []
    involved_nodes: Set[int] = set()
    for index in rows:
        prediction_row = predictions[index]
        subj, obj = (int(v) for v in prediction_row["pair"])
        predicate = int(prediction_row["predicate"])
        involved_nodes.update((subj, obj))
        p0 = _crop_point(centers[subj], crop, panel_size)
        p1 = _crop_point(centers[obj], crop, panel_size)
        gt_matches = prediction_matches[index]
        current_hit = bool(gt_matches)
        rescued_hit = method == "RSGP" and bool(gt_matches - ppg_matches)
        ppg_only_hit = method == "PPG" and bool(
            gt_matches & comparison_lost_indices
        )
        if ppg_only_hit:
            color = PURPLE
        elif rescued_hit:
            color = BLUE
        elif current_hit:
            color = GREEN
        else:
            color = RED
        if current_hit:
            draw.line([p0, p1], fill=(*color, 235), width=4)
        else:
            _dashed_line(draw, p0, p1, color, width=3, dash=10)
        midpoint = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        predicate_name = (
            str(predicate_names[predicate])
            if 0 <= predicate < len(predicate_names)
            else f"r{predicate}"
        )
        short_name = predicate_name if len(predicate_name) <= 20 else predicate_name[:18] + "…"
        draw.text(midpoint, short_name, fill=(*color, 255), font=_font(13), stroke_width=2, stroke_fill=(255, 255, 255, 230))
        displayed.append(
            {
                "prediction_index": index,
                "global_rank": int(prediction_row["global_rank"]),
                "subject": subj,
                "object": obj,
                "predicate": predicate,
                "predicate_name": predicate_name,
                "score": float(prediction_row["score"]),
                "matched_gt_indices": sorted(gt_matches),
                "matched_gt": current_hit,
                "rsgp_rescued": rescued_hit,
                "ppg_only_correct": ppg_only_hit,
            }
        )

    # In a failure case, explicitly pair the PPG-only correct predictions with
    # their absence in the RSGP panel.  These dashed purple lines are reference
    # GT triplets, not claimed RSGP outputs; the sidecar records them separately
    # from ``rsgp_displayed_predictions`` to keep that distinction auditable.
    gt = _tensor(target, "relation_triplets", (0, 3)).long().reshape(-1, 3)
    missing_references: List[Dict[str, object]] = []
    for gt_index in missing_reference_indices:
        if not (0 <= int(gt_index) < len(gt)):
            continue
        subj, obj, predicate = (int(v) for v in gt[int(gt_index)].tolist())
        if subj >= len(centers) or obj >= len(centers):
            continue
        if not (_inside(centers[subj], crop) and _inside(centers[obj], crop)):
            continue
        involved_nodes.update((subj, obj))
        p0 = _crop_point(centers[subj], crop, panel_size)
        p1 = _crop_point(centers[obj], crop, panel_size)
        _dashed_line(draw, p0, p1, PURPLE, width=4, dash=12)
        midpoint = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        marker = 6
        draw.line(
            [
                (midpoint[0] - marker, midpoint[1] - marker),
                (midpoint[0] + marker, midpoint[1] + marker),
            ],
            fill=(*PURPLE, 255),
            width=3,
        )
        draw.line(
            [
                (midpoint[0] - marker, midpoint[1] + marker),
                (midpoint[0] + marker, midpoint[1] - marker),
            ],
            fill=(*PURPLE, 255),
            width=3,
        )
        predicate_name = (
            str(predicate_names[predicate])
            if 0 <= predicate < len(predicate_names)
            else f"r{predicate}"
        )
        short_name = (
            predicate_name if len(predicate_name) <= 17 else predicate_name[:15] + "…"
        )
        draw.text(
            (midpoint[0] + 7, midpoint[1] + 7),
            f"missing: {short_name}",
            fill=(*PURPLE, 255),
            font=_font(13),
            stroke_width=2,
            stroke_fill=(255, 255, 255, 235),
        )
        missing_references.append(
            {
                "gt_index": int(gt_index),
                "subject": subj,
                "object": obj,
                "predicate": predicate,
                "predicate_name": predicate_name,
                "is_model_prediction": False,
                "meaning": "PPG-correct triplet absent from RSGP top-k output",
            }
        )

    for node in sorted(involved_nodes):
        if node >= len(boxes):
            continue
        if str(target["mode"]) == "xywha":
            polygon = [
                _crop_point(point, crop, panel_size)
                for point in _obb_polygon(boxes[node].tolist(), angle_unit)
            ]
            draw.line(polygon + [polygon[0]], fill=(*ORANGE, 245), width=3)
        center = _crop_point(centers[node], crop, panel_size)
        label_id = int(labels[node]) if node < len(labels) else -1
        label_name = str(class_names[label_id]) if 0 <= label_id < len(class_names) else str(label_id)
        draw.text(
            (center[0] + 3, center[1] + 3),
            f"{node}:{label_name}",
            fill=(*BLACK, 255),
            font=_font(12),
            stroke_width=2,
            stroke_fill=(255, 255, 255, 240),
        )
    summary = {
        "local_prediction_count": len(local_indices),
        "local_matched_prediction_count": sum(
            bool(prediction_matches[index]) for index in local_indices
        ),
        "displayed_prediction_count": len(displayed),
        "displayed_matched_prediction_count": sum(
            bool(row["matched_gt"]) for row in displayed
        ),
        "displayed_missing_reference_count": len(missing_references),
    }
    return (
        Image.alpha_composite(panel.convert("RGBA"), overlay).convert("RGB"),
        displayed,
        summary,
        missing_references,
    )


def _overview_panel(source: Image.Image, crop, panel_size: int) -> Image.Image:
    image = source.copy()
    image.thumbnail((panel_size, panel_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (panel_size, panel_size), WHITE)
    ox, oy = (panel_size - image.width) // 2, (panel_size - image.height) // 2
    canvas.paste(image, (ox, oy))
    sx, sy = image.width / source.width, image.height / source.height
    rectangle = (
        ox + crop[0] * sx,
        oy + crop[1] * sy,
        ox + crop[2] * sx,
        oy + crop[3] * sy,
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(rectangle, outline=RED, width=5)
    draw.text((18, 18), "Full-image locator", fill=BLACK, font=_font(22), stroke_width=2, stroke_fill=WHITE)
    return canvas


def render_case(
    ppg: Mapping[str, object],
    rsgp: Mapping[str, object],
    output_path: Path,
    *,
    topk: int,
    panel_size: int,
    edge_limit: int,
    crop_fraction: float = 0.30,
    control_crop_fraction: float = 0.12,
    crop_override: Tuple[int, int, int, int] | None = None,
    require_active_filtering: bool = False,
) -> Dict[str, object]:
    if int(ppg["image_id"]) != int(rsgp["image_id"]):
        raise ValueError("PPG and RSGP artifacts must describe the same image")
    filter_audit: Dict[str, Dict[str, object]] = {}
    for expected_method, artifact in (("PPG", ppg), ("RSGP", rsgp)):
        prediction = artifact["prediction"]
        semantic_count = int(
            _tensor(prediction, "sema_rel_pair_idxs", (0, 2)).reshape(-1, 2).size(0)
        )
        final_count = int(
            _tensor(prediction, "rel_pair_idxs", (0, 2)).reshape(-1, 2).size(0)
        )
        actual_method = str(artifact.get("filter_method", "")).upper()
        actively_filtered = semantic_count > final_count > 0
        filter_audit[expected_method.lower()] = {
            "declared_method": actual_method,
            "semantic_candidate_count": semantic_count,
            "final_candidate_count": final_count,
            "actively_filtered": actively_filtered,
        }
        if require_active_filtering and actual_method != expected_method:
            raise ValueError(
                f"Expected {expected_method} artifact, got filter_method={actual_method!r}"
            )
        if require_active_filtering and not actively_filtered:
            raise ValueError(
                f"Image {artifact['image_id']} does not exercise {expected_method} "
                f"filtering: semantic={semantic_count}, final={final_count}"
            )
    ppg_predictions = graph_predictions(ppg, topk)
    rsgp_predictions = graph_predictions(rsgp, topk)
    ppg_prediction_matches = prediction_gt_matches(ppg, ppg_predictions)
    rsgp_prediction_matches = prediction_gt_matches(rsgp, rsgp_predictions)
    ppg_matches = set().union(*ppg_prediction_matches) if ppg_prediction_matches else set()
    rsgp_matches = set().union(*rsgp_prediction_matches) if rsgp_prediction_matches else set()
    rescued = rsgp_matches - ppg_matches
    lost = ppg_matches - rsgp_matches
    gt = _tensor(ppg["target"], "relation_triplets", (0, 3)).long().reshape(-1, 3)
    gt_pairs = {tuple(int(v) for v in row[:2]) for row in gt.tolist()}
    denom = max(len(gt_pairs), 1)
    ppg_recall = len(ppg_matches) / denom
    rsgp_recall = len(rsgp_matches) / denom
    recall_delta = rsgp_recall - ppg_recall
    if recall_delta > 0:
        role, priority = "success", rescued
    elif recall_delta < 0:
        role, priority = "failure", lost
    else:
        role, priority = "control", ppg_matches | rsgp_matches
    # A control image has no rescued/lost relation to localize.  Using the
    # normal crop size therefore tends to include a wide spread of common
    # relations and makes small remote-sensing objects unreadable.  Select a
    # tighter, locally dense window for controls while leaving success/failure
    # cases unchanged.
    selected_crop_fraction = (
        float(control_crop_fraction) if role == "control" else float(crop_fraction)
    )
    crop = (
        tuple(int(value) for value in crop_override)
        if crop_override is not None
        else choose_crop(ppg, priority, crop_fraction=selected_crop_fraction)
    )

    with Image.open(str(ppg["image_path"])) as raw:
        source = raw.convert("RGB")
    overview = _overview_panel(source, crop, panel_size)
    # Failure panels use a small, paired set of PPG-only correct triplets.  The
    # same triplets are then overlaid as explicitly marked *missing references*
    # in the RSGP panel, making the failure visible without misrepresenting a
    # GT edge as an RSGP prediction.
    ppg_edge_limit = min(edge_limit, 8) if role == "failure" else edge_limit
    ppg_panel, ppg_displayed, ppg_local, _ = _render_crop_panel(
        source,
        ppg,
        crop,
        method="PPG",
        predictions=ppg_predictions,
        prediction_matches=ppg_prediction_matches,
        ppg_matches=ppg_matches,
        priority_indices=priority,
        comparison_lost_indices=lost,
        missing_reference_indices=(),
        panel_size=panel_size,
        edge_limit=ppg_edge_limit,
    )
    displayed_lost = sorted(
        {
            int(gt_index)
            for row in ppg_displayed
            for gt_index in row["matched_gt_indices"]
            if int(gt_index) in lost
        }
    )[:6]
    rsgp_edge_limit = min(edge_limit, 6) if role == "failure" else edge_limit
    rsgp_panel, rsgp_displayed, rsgp_local, rsgp_missing_references = _render_crop_panel(
        source,
        rsgp,
        crop,
        method="RSGP",
        predictions=rsgp_predictions,
        prediction_matches=rsgp_prediction_matches,
        ppg_matches=ppg_matches,
        priority_indices=priority,
        comparison_lost_indices=lost,
        missing_reference_indices=displayed_lost if role == "failure" else (),
        panel_size=panel_size,
        edge_limit=rsgp_edge_limit,
    )

    header, footer = 62, 82
    figure = Image.new("RGB", (panel_size * 3, panel_size + header + footer), WHITE)
    figure.paste(overview, (0, header))
    figure.paste(ppg_panel, (panel_size, header))
    figure.paste(rsgp_panel, (panel_size * 2, header))
    draw = ImageDraw.Draw(figure)
    titles = (
        f"(a) image {ppg['image_id']} / selected crop",
        f"(b) PPG, Triplet R@{topk}={100*ppg_recall:.2f}%",
        f"(c) RSGP, Triplet R@{topk}={100*rsgp_recall:.2f}%",
    )
    for column, title in enumerate(titles):
        draw.text((column * panel_size + 16, 18), title, fill=BLACK, font=_font(21))
    if role == "failure":
        legend = (
            "purple solid: PPG-only correct output   purple dashed x: same triplet "
            "missing under RSGP (reference, not output)   green: other matched output"
        )
    else:
        legend = "green: matched output   blue: RSGP-rescued matched output   red dashed: GT-unmatched output   orange: OBB"
    draw.text((18, header + panel_size + 20), legend, fill=BLACK, font=_font(16))
    draw.text(
        (18, header + panel_size + 50),
        (
            f"actual top-{topk} outputs; local matched/predicted: "
            f"PPG={ppg_local['local_matched_prediction_count']}/{ppg_local['local_prediction_count']}, "
            f"RSGP={rsgp_local['local_matched_prediction_count']}/{rsgp_local['local_prediction_count']}; "
            f"rescued={len(rescued)}; lost={len(lost)}; paired lost shown={len(rsgp_missing_references)}; "
            f"ΔR={100*recall_delta:+.2f} points"
        ),
        fill=BLACK,
        font=_font(16),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.save(output_path)
    return {
        "image_id": int(ppg["image_id"]),
        "role": role,
        "crop_fraction": selected_crop_fraction,
        "crop_xyxy_raw": list(crop),
        "topk": int(topk),
        "filter_audit": filter_audit,
        "gt_pair_count": len(gt_pairs),
        "ppg_match_count": len(ppg_matches),
        "rsgp_match_count": len(rsgp_matches),
        "ppg_triplet_recall": ppg_recall,
        "rsgp_triplet_recall": rsgp_recall,
        "delta_triplet_recall": recall_delta,
        "rescued_gt_indices": sorted(rescued),
        "lost_gt_indices": sorted(lost),
        "ppg_local_prediction_summary": ppg_local,
        "rsgp_local_prediction_summary": rsgp_local,
        "ppg_displayed_predictions": ppg_displayed,
        "rsgp_displayed_predictions": rsgp_displayed,
        "rsgp_missing_triplet_references": rsgp_missing_references,
        "figure": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppg-dir", type=Path, required=True)
    parser.add_argument("--rsgp-dir", type=Path, required=True)
    parser.add_argument("--image-ids", default="440,748,235")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=2000)
    parser.add_argument("--panel-size", type=int, default=800)
    parser.add_argument("--edge-limit", type=int, default=18)
    parser.add_argument("--crop-fraction", type=float, default=0.30)
    parser.add_argument("--control-crop-fraction", type=float, default=0.12)
    parser.add_argument(
        "--allow-no-truncation",
        action="store_true",
        help="Allow cases where semantic candidates already fit the final budget.",
    )
    parser.add_argument(
        "--crop-override",
        default="",
        help="Optional raw-image crop as x0,y0,x1,y1; intended for final paper composition.",
    )
    parser.add_argument(
        "--crop-overrides",
        default="",
        help=(
            "Optional per-image crops as "
            "image_id:x0,y0,x1,y1;image_id:x0,y0,x1,y1."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_ids = [int(token.strip()) for token in args.image_ids.split(",") if token.strip()]
    crop_override = None
    if args.crop_override.strip():
        values = tuple(
            int(token.strip())
            for token in args.crop_override.split(",")
            if token.strip()
        )
        if len(values) != 4:
            raise ValueError("--crop-override must contain x0,y0,x1,y1")
        crop_override = values
        if len(image_ids) != 1:
            raise ValueError("--crop-override requires exactly one --image-ids value")
    crop_overrides: Dict[int, Tuple[int, int, int, int]] = {}
    if args.crop_overrides.strip():
        for item in args.crop_overrides.split(";"):
            image_token, separator, crop_token = item.strip().partition(":")
            if not separator:
                raise ValueError(
                    "--crop-overrides entries must use image_id:x0,y0,x1,y1"
                )
            values = tuple(
                int(token.strip())
                for token in crop_token.split(",")
                if token.strip()
            )
            if len(values) != 4:
                raise ValueError(
                    "--crop-overrides entries must contain x0,y0,x1,y1"
                )
            crop_overrides[int(image_token)] = values
    rows = []
    for image_id in image_ids:
        ppg_path = args.ppg_dir / f"{image_id:04d}.pt"
        rsgp_path = args.rsgp_dir / f"{image_id:04d}.pt"
        if not ppg_path.is_file() or not rsgp_path.is_file():
            raise FileNotFoundError(f"Missing paired artifacts for image {image_id}: {ppg_path}, {rsgp_path}")
        row = render_case(
            load_artifact(ppg_path),
            load_artifact(rsgp_path),
            args.output_dir / f"{image_id:04d}_ppg_vs_rsgp.png",
            topk=args.topk,
            panel_size=args.panel_size,
            edge_limit=args.edge_limit,
            crop_fraction=args.crop_fraction,
            control_crop_fraction=args.control_crop_fraction,
            crop_override=crop_override or crop_overrides.get(image_id),
            require_active_filtering=not args.allow_no_truncation,
        )
        rows.append(row)
        print(
            f"Rendered image {image_id}: role={row['role']}, "
            f"PPG={100*row['ppg_triplet_recall']:.2f}, "
            f"RSGP={100*row['rsgp_triplet_recall']:.2f}",
            flush=True,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps({"schema_version": 1, "cases": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
