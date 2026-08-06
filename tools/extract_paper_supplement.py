#!/usr/bin/env python3
"""Extract paper-ready predicate tables and qualitative case candidates.

The current evaluator prints class-wise recall to ``test.log``.  Older final
JSON files predate the structured ``per-predicate-recall`` and ``per-image``
fields, so this utility deliberately supports both sources:

* class-wise values are parsed from the authoritative logs;
* qualitative cases are paired from saved per-image final-prediction records;
* future evaluations can use the complete ``per-image`` JSON records directly.

All default inputs point to the frozen PredCls paper experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/paper_submission/supplement"
DEFAULT_PAPER_DOC = ROOT / "docs/per_predicate_supplement.md"

DEFAULT_RUNS = {
    "Base": ROOT / "outputs/paper_submission/predcls/unified/test/ppg/test.log",
    "D": ROOT / "outputs/paper_submission/predcls/dual/test/ppg/test.log",
    "DL": ROOT / "outputs/paper_submission/predcls/dual_la/test/ppg/test.log",
    "Full": ROOT
    / "outputs/paper_submission/predcls/dual_la/rsgp_component_test/rsgp_full/test.log",
}
DEFAULT_PPG_JSON = (
    ROOT / "outputs/paper_submission/predcls/dual_la/test/ppg/test_metrics.json"
)
DEFAULT_RSGP_JSON = (
    ROOT
    / "outputs/paper_submission/predcls/dual_la/rsgp_component_test/rsgp_full/test_metrics.json"
)
DEFAULT_CASE_PPG_JSONS = (
    ROOT / "outputs/paper_submission/qualitative_figures/artifacts/ppg/metrics.json",
    ROOT
    / "outputs/paper_submission/qualitative_case_search/preview/artifacts/ppg/metrics.json",
)
DEFAULT_CASE_RSGP_JSONS = (
    ROOT / "outputs/paper_submission/qualitative_figures/artifacts/rsgp/metrics.json",
    ROOT
    / "outputs/paper_submission/qualitative_case_search/preview/artifacts/rsgp/metrics.json",
)
DEFAULT_IMAGE_ROOT = Path(
    "/home/ubuntu/research/ssd/RSDatasets/STAR_SGG/STAR_img"
)
DEFAULT_IMAGE_DATA = Path(
    "/home/ubuntu/research/ssd/RSDatasets/STAR_SGG/STAR_image_data_v1.json"
)
DEFAULT_ROIDB = Path(
    "/home/ubuntu/research/ssd/RSDatasets/STAR_SGG/STAR-SGG-with-attri.h5"
)
DEFAULT_DICT = Path(
    "/home/ubuntu/research/ssd/RSDatasets/STAR_SGG/STAR-SGG-dicts-with-attri.json"
)

PREDICATE_ROW = re.compile(
    r"^(?P<name>.*?)\s+\| count\s+(?P<count>\d+)\s+"
    r"\| @1000\s+(?P<r1000>[0-9.]+)\s+"
    r"\| @1500\s+(?P<r1500>[0-9.]+)\s+"
    r"\| @2000\s+(?P<r2000>[0-9.]+)\s*$"
)


def _load_json(path: Path) -> MutableMapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_predicate_log(path: Path) -> List[Dict[str, object]]:
    """Return the final Per-Relation Recall block from an evaluation log."""
    blocks: List[List[Dict[str, object]]] = []
    active: Optional[List[Dict[str, object]]] = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip() == "Per-Relation Recall:":
            active = []
            blocks.append(active)
            continue
        if active is None:
            continue
        if raw_line.strip() == "Pair Accuracy:":
            active = None
            continue
        match = PREDICATE_ROW.match(raw_line)
        if match is None:
            continue
        active.append(
            {
                "name": match.group("name").strip(),
                "count": int(match.group("count")),
                "1000": float(match.group("r1000")),
                "1500": float(match.group("r1500")),
                "2000": float(match.group("r2000")),
            }
        )
    blocks = [block for block in blocks if block]
    if not blocks:
        raise ValueError(f"No Per-Relation Recall block found in {path}")
    rows = blocks[-1]
    if len(rows) != 58:
        raise ValueError(f"Expected 58 predicates in {path}, found {len(rows)}")
    return rows


def _validate_runs(runs: Mapping[str, List[Dict[str, object]]]) -> None:
    reference_name = next(iter(runs))
    reference = runs[reference_name]
    reference_names = [str(row["name"]) for row in reference]
    for run_name, rows in runs.items():
        names = [str(row["name"]) for row in rows]
        if names != reference_names:
            raise ValueError(
                f"Predicate order mismatch between {reference_name} and {run_name}"
            )
        for predicate_id, (ref_row, row) in enumerate(zip(reference, rows), start=1):
            if int(ref_row["count"]) != int(row["count"]):
                raise ValueError(
                    f"Predicate count mismatch for id={predicate_id} ({row['name']}): "
                    f"{reference_name}={ref_row['count']} vs {run_name}={row['count']}"
                )


def build_ablation_rows(
    runs: Mapping[str, List[Dict[str, object]]]
) -> List[Dict[str, object]]:
    _validate_runs(runs)
    rows: List[Dict[str, object]] = []
    reference = runs["Base"]
    for index, ref_row in enumerate(reference):
        row: Dict[str, object] = {
            "predicate_id": index + 1,
            "predicate": ref_row["name"],
            "count": ref_row["count"],
        }
        for run_name, values in runs.items():
            for k in (1000, 1500, 2000):
                row[f"{run_name}_R@{k}"] = float(values[index][str(k)])
        row["delta_D_minus_Base@2000"] = row["D_R@2000"] - row["Base_R@2000"]
        row["delta_DL_minus_D@2000"] = row["DL_R@2000"] - row["D_R@2000"]
        row["delta_Full_minus_DL@2000"] = row["Full_R@2000"] - row["DL_R@2000"]
        row["delta_Full_minus_Base@2000"] = row["Full_R@2000"] - row["Base_R@2000"]
        rows.append(row)
    return rows


def _coverage_by_predicate(metrics_path: Path) -> Dict[int, float]:
    data = _load_json(metrics_path)
    metrics = data.get("metrics", {})
    stage = metrics.get("predicate-candidate-stage-coverage", {}).get("final", {})
    return {int(predicate): float(value) for predicate, value in stage.items()}


def build_filter_rows(
    ppg: List[Dict[str, object]],
    rsgp: List[Dict[str, object]],
    ppg_json: Path,
    rsgp_json: Path,
) -> List[Dict[str, object]]:
    _validate_runs({"PPG": ppg, "RSGP": rsgp})
    ppg_coverage = _coverage_by_predicate(ppg_json)
    rsgp_coverage = _coverage_by_predicate(rsgp_json)
    rows: List[Dict[str, object]] = []
    for predicate_id, (ppg_row, rsgp_row) in enumerate(zip(ppg, rsgp), start=1):
        row: Dict[str, object] = {
            "predicate_id": predicate_id,
            "predicate": ppg_row["name"],
            "count": ppg_row["count"],
        }
        for k in (1000, 1500, 2000):
            ppg_value = float(ppg_row[str(k)])
            rsgp_value = float(rsgp_row[str(k)])
            row[f"PPG_R@{k}"] = ppg_value
            row[f"RSGP_R@{k}"] = rsgp_value
            row[f"delta_R@{k}"] = rsgp_value - ppg_value
        row["PPG_final_pair_coverage"] = ppg_coverage.get(predicate_id, 0.0)
        row["RSGP_final_pair_coverage"] = rsgp_coverage.get(predicate_id, 0.0)
        row["delta_final_pair_coverage"] = (
            row["RSGP_final_pair_coverage"] - row["PPG_final_pair_coverage"]
        )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: object) -> str:
    return f"{100.0 * float(value):.2f}"


def _signed_pct(value: object) -> str:
    return f"{100.0 * float(value):+.2f}"


def _write_ablation_markdown(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "# Per-predicate PredCls ablation (@2000)",
        "",
        "All values except `Count` are percentages. Base/D/DL use PPG; Full uses the",
        "validation-selected statistical RSGP Hybrid 9000/1000. The same fixed test",
        "split and evaluator are used for all rows.",
        "",
        "| ID | Predicate | Count | Base | D | DL | Full | D−Base | DL−D | Full−DL |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {predicate_id} | {predicate} | {count} | {base} | {d} | {dl} | {full} | "
            "{d_base} | {dl_d} | {full_dl} |".format(
                **row,
                base=_pct(row["Base_R@2000"]),
                d=_pct(row["D_R@2000"]),
                dl=_pct(row["DL_R@2000"]),
                full=_pct(row["Full_R@2000"]),
                d_base=_signed_pct(row["delta_D_minus_Base@2000"]),
                dl_d=_signed_pct(row["delta_DL_minus_D@2000"]),
                full_dl=_signed_pct(row["delta_Full_minus_DL@2000"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_filter_markdown(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    improved = sum(float(row["delta_R@2000"]) > 0.00005 for row in rows)
    unchanged = sum(abs(float(row["delta_R@2000"])) <= 0.00005 for row in rows)
    regressed = len(rows) - improved - unchanged
    top_gains = sorted(rows, key=lambda row: float(row["delta_R@2000"]), reverse=True)[:10]
    top_drops = sorted(rows, key=lambda row: float(row["delta_R@2000"]))[:10]
    lines = [
        "# Per-predicate PPG vs statistical RSGP",
        "",
        "The relation checkpoint is identical; only the test-time pair filter changes.",
        f"At @2000, {improved}/58 predicates improve, {unchanged}/58 are unchanged, "
        f"and {regressed}/58 decrease (tolerance: 0.005 percentage points).",
        "",
        "## Largest gains at @2000",
        "",
        "| ID | Predicate | Count | PPG | RSGP | Δ | PPG pair cov. | RSGP pair cov. |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_gains:
        lines.append(
            f"| {row['predicate_id']} | {row['predicate']} | {row['count']} | "
            f"{_pct(row['PPG_R@2000'])} | {_pct(row['RSGP_R@2000'])} | "
            f"{_signed_pct(row['delta_R@2000'])} | "
            f"{_pct(row['PPG_final_pair_coverage'])} | "
            f"{_pct(row['RSGP_final_pair_coverage'])} |"
        )
    lines.extend(
        [
            "",
            "## Largest decreases at @2000",
            "",
            "| ID | Predicate | Count | PPG | RSGP | Δ | PPG pair cov. | RSGP pair cov. |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top_drops:
        lines.append(
            f"| {row['predicate_id']} | {row['predicate']} | {row['count']} | "
            f"{_pct(row['PPG_R@2000'])} | {_pct(row['RSGP_R@2000'])} | "
            f"{_signed_pct(row['delta_R@2000'])} | "
            f"{_pct(row['PPG_final_pair_coverage'])} | "
            f"{_pct(row['RSGP_final_pair_coverage'])} |"
        )
    lines.extend(
        [
            "",
            "The complete 58-class, three-cutoff table is in",
            "`per_predicate_filter_ppg_vs_rsgp.csv` and its JSON counterpart.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metrics_rows(path: Path) -> List[Dict[str, object]]:
    metrics = _load_json(path).get("metrics", {})
    rows = metrics.get("per-image")
    if isinstance(rows, list) and rows:
        return [dict(row) for row in rows]
    return [dict(row) for row in metrics.get("failure-cases", [])]


def _index_rows(rows: Iterable[Mapping[str, object]]) -> Dict[int, Dict[str, object]]:
    return {int(row["image_id"]): dict(row) for row in rows}


def _dataset_metadata(
    image_ids: Sequence[int],
    image_root: Path,
    image_data_path: Path,
    roidb_path: Path,
    dict_path: Path,
) -> Dict[int, Dict[str, object]]:
    image_data = _load_json(image_data_path)
    info = _load_json(dict_path)
    class_names = {int(v): str(k) for k, v in info["label_to_idx"].items()}
    predicate_names = {int(v): str(k) for k, v in info["predicate_to_idx"].items()}
    result: Dict[int, Dict[str, object]] = {}
    try:
        import h5py  # type: ignore

        with h5py.File(roidb_path, "r") as handle:
            labels = handle["labels"][:, 0]
            predicates = handle["predicates"][:, 0]
            first_box = handle["img_to_first_box"][:]
            last_box = handle["img_to_last_box"][:]
            first_rel = handle["img_to_first_rel"][:]
            last_rel = handle["img_to_last_rel"][:]
            for image_id in image_ids:
                meta = image_data[image_id]
                b0, b1 = int(first_box[image_id]), int(last_box[image_id])
                r0, r1 = int(first_rel[image_id]), int(last_rel[image_id])
                object_counts = Counter(int(v) for v in labels[b0 : b1 + 1])
                predicate_counts = (
                    Counter(int(v) for v in predicates[r0 : r1 + 1])
                    if r0 >= 0 and r1 >= r0
                    else Counter()
                )
                result[image_id] = {
                    "image_path": str(image_root / f"{image_id:04d}.png"),
                    "raw_width": int(meta["width"]),
                    "raw_height": int(meta["height"]),
                    "object_count": int(sum(object_counts.values())),
                    "dominant_objects": [
                        {"name": class_names.get(idx, str(idx)), "count": count}
                        for idx, count in object_counts.most_common(6)
                    ],
                    "dominant_predicates": [
                        {"name": predicate_names.get(idx, str(idx)), "count": count}
                        for idx, count in predicate_counts.most_common(6)
                    ],
                }
    except (ImportError, OSError, KeyError):
        for image_id in image_ids:
            meta = image_data[image_id]
            result[image_id] = {
                "image_path": str(image_root / f"{image_id:04d}.png"),
                "raw_width": int(meta["width"]),
                "raw_height": int(meta["height"]),
            }
    return result


def select_qualitative_cases(
    ppg_json: Path,
    rsgp_json: Path,
    image_root: Path,
    image_data_path: Path,
    roidb_path: Path,
    dict_path: Path,
    extra_ppg_jsons: Sequence[Path] = (),
    extra_rsgp_jsons: Sequence[Path] = (),
    preferred_image_ids: Sequence[int] = (440, 748, 235),
) -> List[Dict[str, object]]:
    ppg_rows = _index_rows(_metrics_rows(ppg_json))
    rsgp_rows = _index_rows(_metrics_rows(rsgp_json))
    for extra_ppg_json in extra_ppg_jsons:
        if extra_ppg_json.is_file():
            ppg_rows.update(_index_rows(_metrics_rows(extra_ppg_json)))
    for extra_rsgp_json in extra_rsgp_jsons:
        if extra_rsgp_json.is_file():
            rsgp_rows.update(_index_rows(_metrics_rows(extra_rsgp_json)))
    shared_ids = sorted(set(ppg_rows) & set(rsgp_rows))
    if not shared_ids:
        raise ValueError("PPG and RSGP JSON files have no paired per-image records")

    def delta(image_id: int) -> float:
        return float(rsgp_rows[image_id]["triplet_recall"]) - float(
            ppg_rows[image_id]["triplet_recall"]
        )

    overloaded = [
        image_id
        for image_id in shared_ids
        if str(ppg_rows[image_id].get("pressure_bucket")) != "no_truncation"
    ]
    preferred = [int(image_id) for image_id in preferred_image_ids]
    if preferred and all(image_id in shared_ids for image_id in preferred):
        roles = ("success", "second_filtered_success", "failure")
        rationales = (
            "Exact paired high-load case with a positive final-triplet gain.",
            "Exact readable high-load case with a large final-triplet gain.",
            "Exact readable case where graph reallocation removes correct triplets.",
        )
        chosen = list(zip(roles, preferred, rationales))
    else:
        improving = sorted(
            (image_id for image_id in overloaded if delta(image_id) > 0.0),
            key=delta,
            reverse=True,
        )
        degrading = sorted(
            (image_id for image_id in overloaded if delta(image_id) < 0.0),
            key=delta,
        )
        if len(improving) < 2 or not degrading:
            raise ValueError(
                "Qualitative selection requires two improving and one degrading "
                "paired cases where semantic candidates exceed the final budget"
            )
        chosen = [
            ("success", improving[0], "Largest exact filtered final-triplet gain."),
            (
                "second_filtered_success",
                improving[1],
                "Second exact filtered final-triplet gain.",
            ),
            ("failure", degrading[0], "Largest exact filtered final-triplet decrease."),
        ]

    metadata = _dataset_metadata(
        [image_id for _, image_id, _ in chosen],
        image_root,
        image_data_path,
        roidb_path,
        dict_path,
    )
    cases: List[Dict[str, object]] = []
    for role, image_id, rationale in chosen:
        ppg_row = ppg_rows[image_id]
        rsgp_row = rsgp_rows[image_id]
        for method_name, row in (("PPG", ppg_row), ("RSGP", rsgp_row)):
            semantic_count = int(row.get("semantic_candidate_count", 0))
            final_count = int(row.get("pred_pair_count", 0))
            if semantic_count <= final_count:
                raise ValueError(
                    f"Image {image_id} does not exercise {method_name} filtering: "
                    f"semantic={semantic_count}, final={final_count}"
                )
        case: Dict[str, object] = {
            "role": role,
            "image_id": image_id,
            "rationale": rationale,
            **metadata.get(image_id, {}),
            "ppg": ppg_row,
            "rsgp": rsgp_row,
            "delta_triplet_recall": delta(image_id),
            "comparison_is_exact": True,
        }
        cases.append(case)
    return cases


def _write_cases_markdown(path: Path, cases: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "# Qualitative case selection",
        "",
        "Cases are selected from the same Dual+LA checkpoint under PPG and the final",
        "statistical RSGP. `Triplet R` is the per-image graph-constrained recall at",
        "the largest cutoff (@2000).",
        "",
        "| Role | Image ID | Pressure | Semantic pairs | GT pairs | PPG Triplet R | RSGP Triplet R | Δ |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        ppg = case["ppg"]
        if bool(case["comparison_is_exact"]):
            rsgp_value = _pct(case["rsgp"]["triplet_recall"])
            delta_value = _signed_pct(case["delta_triplet_recall"])
        else:
            rsgp_value = "≥" + _pct(case["rsgp_triplet_recall_lower_bound"])
            delta_value = "≥" + _signed_pct(case["delta_triplet_recall_lower_bound"])
        lines.append(
            f"| {case['role']} | {case['image_id']} | {ppg['pressure_bucket']} | "
            f"{ppg['semantic_candidate_count']} | {ppg['gt_pair_count']} | "
            f"{_pct(ppg['triplet_recall'])} | {rsgp_value} | {delta_value} |"
        )
    lines.extend(["", "## Recommended use", ""])
    for case in cases:
        dominant_objects = ", ".join(
            f"{item['name']} ({item['count']})"
            for item in case.get("dominant_objects", [])[:4]
        )
        dominant_predicates = ", ".join(
            f"{item['name']} ({item['count']})"
            for item in case.get("dominant_predicates", [])[:4]
        )
        lines.extend(
            [
                f"### {case['role']}: image {case['image_id']}",
                "",
                str(case["rationale"]),
                "",
                f"- Source: `{case.get('image_path', '')}`",
                f"- Raw size: {case.get('raw_width', '?')} × {case.get('raw_height', '?')}",
                f"- Dominant objects: {dominant_objects or 'not available'}",
                f"- Dominant predicates: {dominant_predicates or 'not available'}",
                "",
            ]
        )
    lines.extend(
        [
            "All selected cases are exact paired final-triplet comparisons. Each has more",
            "semantic-valid pairs than final candidate pairs, so both PPG and RSGP actively",
            "execute filtering. No-truncation images are reserved for the pressure table and",
            "must not be presented as qualitative evidence of one filter outperforming another.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_thumbnails(cases: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - optional presentation dependency
        raise RuntimeError("Pillow is required for --render-thumbnails") from exc

    Image.MAX_IMAGE_PIXELS = None
    output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        source = Path(str(case.get("image_path", "")))
        if not source.is_file():
            continue
        with Image.open(source) as raw:
            image = raw.convert("RGB")
        raw_size = image.size
        image.thumbnail((1400, 900))
        canvas = Image.new("RGB", (1400, 960), "white")
        canvas.paste(image, ((1400 - image.width) // 2, 50))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (16, 15),
            f"{case['role']} | image_id={case['image_id']} | raw={raw_size[0]}x{raw_size[1]}",
            fill="black",
        )
        canvas.save(output_dir / f"{int(case['image_id']):04d}_{case['role']}.jpg", quality=90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-log", type=Path, default=DEFAULT_RUNS["Base"])
    parser.add_argument("--dual-log", type=Path, default=DEFAULT_RUNS["D"])
    parser.add_argument("--dual-la-log", type=Path, default=DEFAULT_RUNS["DL"])
    parser.add_argument("--full-log", type=Path, default=DEFAULT_RUNS["Full"])
    parser.add_argument("--ppg-json", type=Path, default=DEFAULT_PPG_JSON)
    parser.add_argument("--rsgp-json", type=Path, default=DEFAULT_RSGP_JSON)
    parser.add_argument(
        "--case-ppg-json",
        type=Path,
        action="append",
        default=[],
        help="Additional exact per-image PPG metrics; may be repeated.",
    )
    parser.add_argument(
        "--case-rsgp-json",
        type=Path,
        action="append",
        default=[],
        help="Additional exact per-image RSGP metrics; may be repeated.",
    )
    parser.add_argument("--qualitative-image-ids", default="440,748,235")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--image-data", type=Path, default=DEFAULT_IMAGE_DATA)
    parser.add_argument("--roidb", type=Path, default=DEFAULT_ROIDB)
    parser.add_argument("--dict-file", type=Path, default=DEFAULT_DICT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--paper-doc", type=Path, default=DEFAULT_PAPER_DOC)
    parser.add_argument("--render-thumbnails", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_ppg_jsons = args.case_ppg_json or list(DEFAULT_CASE_PPG_JSONS)
    case_rsgp_jsons = args.case_rsgp_json or list(DEFAULT_CASE_RSGP_JSONS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_paths = {
        "Base": args.base_log,
        "D": args.dual_log,
        "DL": args.dual_la_log,
        "Full": args.full_log,
    }
    runs = {name: parse_predicate_log(path) for name, path in run_paths.items()}
    ablation_rows = build_ablation_rows(runs)
    filter_rows = build_filter_rows(
        runs["DL"], runs["Full"], args.ppg_json, args.rsgp_json
    )
    cases = select_qualitative_cases(
        args.ppg_json,
        args.rsgp_json,
        args.image_root,
        args.image_data,
        args.roidb,
        args.dict_file,
        extra_ppg_jsons=case_ppg_jsons,
        extra_rsgp_jsons=case_rsgp_jsons,
        preferred_image_ids=[
            int(token.strip())
            for token in args.qualitative_image_ids.split(",")
            if token.strip()
        ],
    )

    _write_csv(args.output_dir / "per_predicate_ablation.csv", ablation_rows)
    _write_csv(args.output_dir / "per_predicate_filter_ppg_vs_rsgp.csv", filter_rows)
    (args.output_dir / "per_predicate_ablation.json").write_text(
        json.dumps({"sources": {k: str(v) for k, v in run_paths.items()}, "rows": ablation_rows}, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "per_predicate_filter_ppg_vs_rsgp.json").write_text(
        json.dumps(
            {
                "ppg_log": str(args.dual_la_log),
                "rsgp_log": str(args.full_log),
                "ppg_metrics": str(args.ppg_json),
                "rsgp_metrics": str(args.rsgp_json),
                "rows": filter_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "qualitative_cases.json").write_text(
        json.dumps(
            {
                "ppg_metrics": str(args.ppg_json),
                "rsgp_metrics": str(args.rsgp_json),
                "case_ppg_metrics": [str(path) for path in case_ppg_jsons],
                "case_rsgp_metrics": [str(path) for path in case_rsgp_jsons],
                "selection_basis": "paired final graph-constrained triplet predictions",
                "cases": cases,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    ablation_markdown = args.output_dir / "per_predicate_ablation.md"
    _write_ablation_markdown(ablation_markdown, ablation_rows)
    args.paper_doc.parent.mkdir(parents=True, exist_ok=True)
    args.paper_doc.write_text(
        ablation_markdown.read_text(encoding="utf-8"), encoding="utf-8"
    )
    _write_filter_markdown(
        args.output_dir / "per_predicate_filter_ppg_vs_rsgp.md", filter_rows
    )
    _write_cases_markdown(args.output_dir / "qualitative_cases.md", cases)
    if args.render_thumbnails:
        _render_thumbnails(cases, args.output_dir / "case_thumbnails")

    print(f"Wrote per-predicate tables and case selection to {args.output_dir}")


if __name__ == "__main__":
    main()
