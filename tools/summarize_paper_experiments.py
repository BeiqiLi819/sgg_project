#!/usr/bin/env python3
"""Summarize the controlled paper runs without touching training/test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="outputs/paper_submission",
        help="Root produced by the paper protocol scripts.",
    )
    parser.add_argument(
        "--output",
        default="outputs/paper_submission/summary.json",
    )
    return parser.parse_args()


def _load_metrics(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("metrics", {})


def _metric(metrics: dict, name: str, k: int):
    values = metrics.get(name, {})
    return values.get(str(k), values.get(k))


def main():
    args = parse_args()
    root = Path(args.root)
    predcls_root = root / "predcls"
    rows = {
        "base_unified_ppg": predcls_root / "unified/test/ppg/test_metrics.json",
        "dual_ppg": predcls_root / "dual/test/ppg/test_metrics.json",
        "dual_la_ppg": predcls_root / "dual_la/test/ppg/test_metrics.json",
        "dual_la_ppn": predcls_root / "dual_la/test/ppn/test_metrics.json",
        # The formal Full result is the validation-selected Hybrid 9000/1000
        # run.  The older dual_la/test/rsgp_statistical JSON used 8000/2000.
        "dual_la_rsgp": (
            predcls_root
            / "dual_la/rsgp_component_test/rsgp_full/test_metrics.json"
        ),
        "source_rpcm_audit_ppg": (
            predcls_root / "base_original/test/ppg/test_metrics.json"
        ),
    }
    summary = {}
    for row_name, path in rows.items():
        metrics = _load_metrics(path)
        row = {"path": str(path), "available": metrics is not None}
        if metrics is None:
            summary[row_name] = row
            continue
        for metric_name in ("R", "mR", "HR"):
            for k in (1000, 1500, 2000):
                row[f"{metric_name}@{k}"] = _metric(metrics, metric_name, k)
        for graph_name in (
            "avg_candidate_node_coverage",
            "avg_gt_relation_node_coverage",
            "avg_degree_gini",
            "max_degree",
            "avg_label_pair_entropy",
        ):
            row[graph_name] = metrics.get("candidate-graph", {}).get(graph_name)
        row["gt_pair_coverage"] = metrics.get(
            "candidate-stage-coverage", {}
        ).get("final")
        row["candidate_pressure"] = metrics.get("candidate-pressure", {})
        summary[row_name] = row

    cross_task = {}
    for task_name, task_dir in (
        ("sgcls", "sgcls"),
        ("sgdet", "sgdet_rpcm_budget"),
    ):
        task_rows = {}
        for protocol in ("legacy", "pred"):
            for method in ("ppg", "rsgp"):
                result_dir = "rsgp_statistical" if method == "rsgp" else method
                path = (
                    root
                    / task_dir
                    / "test"
                    / protocol
                    / result_dir
                    / "test_metrics.json"
                )
                metrics = _load_metrics(path)
                row = {
                    "path": str(path),
                    "available": metrics is not None,
                }
                if metrics is not None:
                    row.update(
                        {
                            "R": metrics.get("R", {}),
                            "mR": metrics.get("mR", {}),
                            "HMR": metrics.get("HR", {}),
                            "candidate_graph": metrics.get("candidate-graph", {}),
                        }
                    )
                task_rows[f"{protocol}_{method}"] = row
        cross_task[task_name] = task_rows

    missing_cross_task = [
        f"{task_name}.{row_name}"
        for task_name, task_rows in cross_task.items()
        for row_name, row in task_rows.items()
        if not row["available"]
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "protocol": {
                    "checkpoint_selection": "validation",
                    "final_reporting": "test",
                    "runs_per_configuration": 1,
                    "pair_budget": 10000,
                },
                "predcls": summary,
                "cross_task": cross_task,
                "completion": {
                    "predcls_complete": all(row["available"] for row in summary.values()),
                    "missing_cross_task": missing_cross_task,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
