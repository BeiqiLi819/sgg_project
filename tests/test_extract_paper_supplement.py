from pathlib import Path
import json

from tools.extract_paper_supplement import (
    build_ablation_rows,
    parse_predicate_log,
    select_qualitative_cases,
)


def _write_log(path: Path, offset: float) -> None:
    lines = ["header", "Per-Relation Recall:"]
    for predicate_id in range(1, 59):
        value = min(0.01 * predicate_id + offset, 1.0)
        lines.append(
            f"predicate {predicate_id:<2} | count {predicate_id:4d} | "
            f"@1000 {value:.4f} | @1500 {value:.4f} | @2000 {value:.4f}"
        )
    lines.extend(["Pair Accuracy:", "A | @2000 0.5"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_predicate_log_and_build_ablation_rows(tmp_path):
    paths = {}
    for name, offset in (("Base", 0.0), ("D", 0.01), ("DL", 0.02), ("Full", 0.03)):
        path = tmp_path / f"{name}.log"
        _write_log(path, offset)
        paths[name] = path

    runs = {name: parse_predicate_log(path) for name, path in paths.items()}
    rows = build_ablation_rows(runs)

    assert len(rows) == 58
    assert rows[0]["predicate_id"] == 1
    assert rows[0]["predicate"] == "predicate 1"
    assert rows[0]["count"] == 1
    assert abs(rows[0]["delta_D_minus_Base@2000"] - 0.01) < 1e-8
    assert abs(rows[0]["delta_DL_minus_D@2000"] - 0.01) < 1e-8
    assert abs(rows[0]["delta_Full_minus_DL@2000"] - 0.01) < 1e-8


def test_parse_predicate_log_uses_last_complete_block(tmp_path):
    path = tmp_path / "repeated.log"
    _write_log(path, 0.0)
    first = path.read_text(encoding="utf-8")
    second_path = tmp_path / "second.log"
    _write_log(second_path, 0.1)
    path.write_text(first + second_path.read_text(encoding="utf-8"), encoding="utf-8")

    rows = parse_predicate_log(path)

    assert abs(rows[0]["2000"] - 0.11) < 1e-8


def test_qualitative_selection_requires_exact_filtered_predictions(tmp_path):
    def write_metrics(path, recalls):
        rows = []
        for image_id, recall in recalls.items():
            rows.append(
                {
                    "image_id": image_id,
                    "gt_rel_count": 2,
                    "gt_pair_count": 2,
                    "pred_pair_count": 10,
                    "semantic_candidate_count": 20,
                    "pressure_bucket": "overload_low",
                    "triplet_match_count": int(recall * 10),
                    "triplet_recall": recall,
                }
            )
        path.write_text(json.dumps({"metrics": {"per-image": rows}}))

    ppg_json = tmp_path / "ppg.json"
    rsgp_json = tmp_path / "rsgp.json"
    write_metrics(ppg_json, {1: 0.1, 2: 0.2, 3: 0.3})
    write_metrics(rsgp_json, {1: 0.2, 2: 0.4, 3: 0.1})
    image_data = tmp_path / "images.json"
    image_data.write_text(
        json.dumps([{"width": 10, "height": 10} for _ in range(4)])
    )
    dict_path = tmp_path / "dict.json"
    dict_path.write_text(json.dumps({"label_to_idx": {}, "predicate_to_idx": {}}))

    cases = select_qualitative_cases(
        ppg_json,
        rsgp_json,
        tmp_path,
        image_data,
        tmp_path / "missing.h5",
        dict_path,
        preferred_image_ids=(1, 2, 3),
    )

    assert [case["role"] for case in cases] == [
        "success",
        "second_filtered_success",
        "failure",
    ]
    assert all(case["comparison_is_exact"] for case in cases)
    assert all(case["ppg"]["semantic_candidate_count"] > case["ppg"]["pred_pair_count"] for case in cases)
