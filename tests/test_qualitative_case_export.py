import math
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from tools.eval_once import _parse_image_ids, _restrict_dataset_to_image_ids
from tools.render_qualitative_cases import render_case
from tools.render_spatial_scene_graph_cases import (
    _adjust_image_440_layout,
    _build_pair_route_curvatures,
    _displayed_prediction_pairs,
    _fit_node_label_font,
    _hub_aware_positions,
    _minimum_pair_distance,
    _spread_positions,
    render_spatial_scene_graph_case,
)


def _artifact(image_path: Path, relation_predictions):
    boxes = torch.tensor(
        [
            [100.0, 100.0, 40.0, 20.0, 0.0],
            [220.0, 120.0, 35.0, 18.0, 0.1],
            [350.0, 300.0, 45.0, 22.0, -0.2],
            [470.0, 330.0, 50.0, 25.0, 0.0],
        ]
    )
    pairs = torch.tensor([row[:2] for row in relation_predictions], dtype=torch.long)
    scores = torch.zeros((len(relation_predictions), 4), dtype=torch.float32)
    for index, (_, _, predicate) in enumerate(relation_predictions):
        scores[index, predicate] = 0.9
    return {
        "schema_version": 1,
        "image_id": 7,
        "image_path": str(image_path),
        "raw_width": 600,
        "raw_height": 500,
        "class_names": ["bg", "airplane", "taxiway"],
        "predicate_names": ["bg", "connect", "over", "around"],
        "prediction": {
            "bbox": boxes,
            "size": (600, 500),
            "mode": "xywha",
            "pred_scores": torch.ones(4),
            "rel_pair_idxs": pairs,
            "pred_rel_scores": scores,
        },
        "target": {
            "bbox": boxes,
            "size": (600, 500),
            "mode": "xywha",
            "box_angle_unit": "radian",
            "labels": torch.tensor([1, 2, 1, 2]),
            "relation_triplets": torch.tensor([[0, 1, 1], [2, 3, 2]]),
        },
    }


def test_parse_and_restrict_image_ids():
    class Dataset:
        split = "test"
        records = [
            {"image_index": 1},
            {"image_index": 7},
            {"image_index": 9},
        ]

    dataset = Dataset()
    assert _parse_image_ids("7,1,7") == [7, 1]
    _restrict_dataset_to_image_ids(dataset, [7, 1])
    assert [row["image_index"] for row in dataset.records] == [7, 1]


def test_render_case_selects_rescued_relation(tmp_path):
    image_path = tmp_path / "0007.png"
    Image.new("RGB", (600, 500), (210, 215, 205)).save(image_path)
    ppg = _artifact(image_path, [(0, 1, 1)])
    rsgp = _artifact(image_path, [(0, 1, 1), (2, 3, 2)])
    output = tmp_path / "case.png"

    row = render_case(ppg, rsgp, output, topk=20, panel_size=320, edge_limit=8)

    assert output.is_file()
    assert row["role"] == "success"
    assert row["rsgp_match_count"] > row["ppg_match_count"]
    assert row["rescued_gt_indices"] == [1]
    assert row["rsgp_displayed_predictions"]
    assert any(
        prediction["matched_gt"] and prediction["rsgp_rescued"]
        for prediction in row["rsgp_displayed_predictions"]
    )


def test_render_case_marks_actual_unmatched_predictions(tmp_path):
    image_path = tmp_path / "0007.png"
    Image.new("RGB", (600, 500), (210, 215, 205)).save(image_path)
    ppg = _artifact(image_path, [(0, 1, 3)])
    rsgp = _artifact(image_path, [(0, 1, 1), (2, 3, 3)])

    row = render_case(
        ppg,
        rsgp,
        tmp_path / "actual_predictions.png",
        topk=20,
        panel_size=320,
        edge_limit=8,
    )

    assert any(
        not prediction["matched_gt"]
        for prediction in row["ppg_displayed_predictions"]
    )
    assert row["ppg_local_prediction_summary"]["local_prediction_count"] == 1


def test_failure_case_pairs_ppg_only_output_with_missing_rsgp_reference(tmp_path):
    image_path = tmp_path / "0007.png"
    Image.new("RGB", (600, 500), (210, 215, 205)).save(image_path)
    ppg = _artifact(image_path, [(0, 1, 1), (2, 3, 2)])
    rsgp = _artifact(image_path, [(0, 1, 1)])

    row = render_case(
        ppg,
        rsgp,
        tmp_path / "failure.png",
        topk=20,
        panel_size=320,
        edge_limit=8,
    )

    assert row["role"] == "failure"
    assert row["lost_gt_indices"] == [1]
    assert any(
        prediction["ppg_only_correct"]
        for prediction in row["ppg_displayed_predictions"]
    )
    assert row["rsgp_missing_triplet_references"] == [
        {
            "gt_index": 1,
            "subject": 2,
            "object": 3,
            "predicate": 2,
            "predicate_name": "over",
            "is_model_prediction": False,
            "meaning": "PPG-correct triplet absent from RSGP top-k output",
        }
    ]


def test_spatial_scene_graph_keeps_aligned_gt_nodes_and_exact_focus(tmp_path):
    image_path = tmp_path / "0007.png"
    Image.new("RGB", (600, 500), (210, 215, 205)).save(image_path)
    ppg = _artifact(image_path, [(0, 1, 1), (2, 3, 2)])
    rsgp = _artifact(image_path, [(0, 1, 1)])

    row = render_spatial_scene_graph_case(
        ppg,
        rsgp,
        tmp_path / "spatial_graph.png",
        topk=20,
        panel_size=260,
        render_scale=1,
        crop_fraction=1.0,
        focus_limit=4,
        require_active_filtering=False,
    )

    assert row["layout"] == "spatially_aligned_scene_graph"
    assert row["local_gt_entity_ids"] == [0, 1, 2, 3]
    assert row["local_gt_relation_count"] == 2
    assert row["focus_relations"][0]["triple"] == [2, 3, 2]
    assert row["focus_relations"][0]["present_in_ppg"] is True
    assert row["focus_relations"][0]["present_in_rsgp"] is False
    assert row["layout_diagnostics"]["predicate_text_rendered"] is False
    assert row["layout_diagnostics"]["uniform_edge_width_logical"] == 2


def test_scene_graph_layout_separates_nodes_and_parallel_routes():
    positions = {0: (80.0, 100.0), 1: (82.0, 100.0), 2: (84.0, 100.0)}
    spread = _spread_positions(
        positions,
        size=240,
        margin=30,
        min_distance=48.0,
    )
    assert _minimum_pair_distance(spread) >= 47.5

    pair_routes = _build_pair_route_curvatures({(0, 1), (1, 0)}, spacing=16.0)
    assert pair_routes[(0, 1)] != -pair_routes[(1, 0)]


def test_hidden_reverse_prediction_does_not_curve_visible_edge():
    predictions = [
        {"pair": (0, 1), "predicate": 1},
        {"pair": (1, 0), "predicate": 3},
    ]
    displayed = _displayed_prediction_pairs(
        predictions,
        [(0, 1, 1)],
        (),
        unmatched_limit=0,
    )
    assert displayed == {(0, 1)}
    assert _build_pair_route_curvatures(displayed, spacing=16.0)[(0, 1)] == 0.0

    displayed_with_reverse = _displayed_prediction_pairs(
        predictions,
        [(0, 1, 1)],
        (),
        unmatched_limit=1,
    )
    assert displayed_with_reverse == {(0, 1), (1, 0)}
    routes = _build_pair_route_curvatures(displayed_with_reverse, spacing=16.0)
    assert routes[(0, 1)] != 0.0
    assert routes[(1, 0)] != 0.0


def test_dense_three_digit_node_label_fits_inside_circle():
    draw = ImageDraw.Draw(Image.new("RGB", (100, 100), "white"))
    _, bbox = _fit_node_label_font(
        draw,
        "238",
        base_logical_size=10,
        radius=20,
        scale=2,
    )
    max_extent = 2 * 20 - 3 * 2
    assert bbox[2] - bbox[0] <= max_extent
    assert bbox[3] - bbox[1] <= max_extent


def test_image_440_layout_opens_hubs_and_preserves_right_sector():
    positions = {
        231: (100.0, 110.0),
        238: (100.0, 90.0),
        10: (160.0, 100.0),  # right sector: fixed
        20: (60.0, 30.0),
        21: (100.0, 30.0),
        30: (40.0, 120.0),
        31: (40.0, 160.0),
    }
    adjusted, diagnostics = _adjust_image_440_layout(
        positions,
        size=200,
        margin=10,
        min_distance=10.0,
    )
    assert diagnostics["applied"] is True
    assert math.dist(adjusted[231], adjusted[238]) > math.dist(
        positions[231], positions[238]
    )
    assert adjusted[10] == positions[10]
    assert abs(adjusted[20][0] - adjusted[21][0]) > abs(
        positions[20][0] - positions[21][0]
    )
    assert adjusted[30][0] < positions[30][0]


def test_scene_graph_uses_bounded_hub_relaxation_for_complete_star():
    projected = {0: (120.0, 120.0)}
    projected.update(
        {node: (150.0 + node, 100.0 + node) for node in range(1, 11)}
    )
    positions, diagnostics = _hub_aware_positions(
        projected,
        {(0, node) for node in range(1, 11)},
        size=500,
        margin=40,
        min_distance=48.0,
    )
    assert diagnostics["mode"] == "spatial_hub_relaxed"
    assert diagnostics["hub"] == 0
    assert _minimum_pair_distance(positions) >= 47.5


def test_control_case_uses_tighter_crop(tmp_path):
    image_path = tmp_path / "0007.png"
    Image.new("RGB", (600, 500), (210, 215, 205)).save(image_path)
    ppg = _artifact(image_path, [(0, 1, 1), (2, 3, 2)])
    rsgp = _artifact(image_path, [(0, 1, 1), (2, 3, 2)])

    row = render_case(
        ppg,
        rsgp,
        tmp_path / "control.png",
        topk=20,
        panel_size=320,
        edge_limit=8,
        crop_fraction=0.30,
        control_crop_fraction=0.12,
    )

    assert row["role"] == "control"
    assert row["crop_fraction"] == 0.12


def test_active_filter_audit_rejects_no_truncation_case(tmp_path):
    image_path = tmp_path / "0007.png"
    Image.new("RGB", (600, 500), (210, 215, 205)).save(image_path)
    ppg = _artifact(image_path, [(0, 1, 1)])
    rsgp = _artifact(image_path, [(0, 1, 1)])
    ppg["filter_method"] = "PPG"
    rsgp["filter_method"] = "RSGP"

    with pytest.raises(ValueError, match="does not exercise PPG filtering"):
        render_case(
            ppg,
            rsgp,
            tmp_path / "no_filtering.png",
            topk=20,
            panel_size=320,
            edge_limit=8,
            require_active_filtering=True,
        )
