#!/usr/bin/env python3
"""Audit/train Hard-Quota-free STAR-Q5 whole-image proposal learning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgg.modeling.roi_heads.q5_budgeted_coverage import (
    positive_group_coverage_score_gradient,
    released_score_anchor_gradient,
)
from sgg.modeling.roi_heads.residual_ppg_scorer import (
    SetConditionedPPGResidual,
    module_state_digest,
    save_set_conditioned_checkpoint,
)
from sgg.modeling.roi_heads.set_context import (
    Q5_SET_CONTEXT_FEATURE_NAMES,
    q5_set_context_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "anchor", "budgeted-coverage"), required=True)
    parser.add_argument("--context-cache", type=Path, required=True)
    parser.add_argument("--positive-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1029)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--budget", type=int, default=10_000)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    parser.add_argument("--coverage-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=-1)
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_shards(meta: dict):
    for row in meta["shards"]:
        yield torch.load(row["path"], map_location="cpu", weights_only=False)


def context_for_shard(shard: dict, meta: dict, device: torch.device, budget: int):
    return q5_set_context_features(
        base_scores=shard["base_scores"].to(device=device, dtype=torch.float32),
        group_ids=shard["group_ids"].to(device=device, dtype=torch.long),
        within_group_percentile=shard["within_group_percentile"].to(
            device=device, dtype=torch.float32
        ),
        group_counts=meta["group_counts"].to(device=device),
        candidate_count=int(meta["candidate_count"]),
        active_group_count=int(meta["active_group_count"]),
        num_groups=int(meta["num_classes"]) ** 2,
        budget=budget,
    )


@torch.no_grad()
def materialize_scores(model, meta, device, budget):
    students, bases = [], []
    for shard in iter_shards(meta):
        base = shard["base_scores"].to(device=device, dtype=torch.float32)
        students.append(model.scores(base, context_for_shard(shard, meta, device, budget)))
        bases.append(base)
    empty = torch.zeros(0, device=device)
    return torch.cat(students) if students else empty, torch.cat(bases) if bases else empty


def backward_stream(model, meta, device, budget, gradient):
    offset = 0
    for shard in iter_shards(meta):
        base = shard["base_scores"].to(device=device, dtype=torch.float32)
        scores = model.scores(base, context_for_shard(shard, meta, device, budget))
        end = offset + scores.numel()
        scores.backward(gradient[offset:end])
        offset = end
    if offset != gradient.numel():
        raise RuntimeError("Q5 streamed gradient did not cover complete image")


def parameter_grad_norm(model) -> float:
    return math.sqrt(
        sum(
            float(parameter.grad.detach().float().square().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    if args.coverage_weight != 1.0:
        raise ValueError("STAR-Q5 freezes coverage-weight=1.0; no search is allowed")
    if args.temperature != 5.0:
        raise ValueError("STAR-Q5 freezes SoftTopK temperature=5.0")
    if args.epsilon != 1.0e-6:
        raise ValueError("STAR-Q5 freezes epsilon=1e-6")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    context_manifest = json.loads((args.context_cache / "manifest.json").read_text())
    positive_manifest = json.loads((args.positive_cache / "manifest.json").read_text())
    if context_manifest.get("split") != "train" or bool(
        context_manifest.get("contains_relation_gt", True)
    ):
        raise ValueError("Q5 context cache must be target-free train")
    if bool(context_manifest.get("contains_hard_quota_allocation", True)) or bool(
        context_manifest.get("hard_quota_fields_consumed", True)
    ):
        raise ValueError("Q5 context contains forbidden HardQuota signal")
    if positive_manifest.get("split") != "train" or not bool(
        positive_manifest.get("contains_relation_gt", False)
    ):
        raise ValueError("Q5 positives must be train-only annotated pairs")
    if bool(positive_manifest.get("contains_predicate_labels", True)) or bool(
        positive_manifest.get("unannotated_pairs_are_negative_labels", True)
    ):
        raise ValueError("Q5 forbids predicate or unannotated-negative supervision")
    if positive_manifest["q5_context_digest"] != context_manifest["q5_context_digest"]:
        raise ValueError("Q5 context/positive cache digests differ")
    context_rows = {int(row["image_id"]): row for row in context_manifest["images"]}
    positive_rows = {int(row["image_id"]): row for row in positive_manifest["images"]}
    if set(context_rows) != set(positive_rows):
        raise ValueError("Q5 context/positive image sets differ")
    rows = [context_rows[image_id] for image_id in sorted(context_rows)]
    if args.limit >= 0:
        rows = rows[: args.limit]
    device = torch.device(args.device)
    model = SetConditionedPPGResidual(
        hidden_dim=32, feature_names=Q5_SET_CONTEXT_FEATURE_NAMES
    ).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "audit":
        selected_rows = rows[:2]
        if rows:
            selected_rows.append(max(rows, key=lambda row: int(row["candidate_count"])))
        selected_rows = list({int(row["image_id"]): row for row in selected_rows}.values())
        audits = []
        nonzero_gradient = False
        for row in selected_rows:
            meta = torch.load(args.context_cache / row["path"], map_location="cpu", weights_only=False)
            positives = torch.load(
                args.positive_cache / positive_rows[int(row["image_id"])]["path"],
                map_location="cpu", weights_only=False,
            )
            student, base = materialize_scores(model, meta, device, args.budget)
            identity = float((student - base).abs().max()) if student.numel() else 0.0
            coverage_loss, coverage_gradient, diagnostics = (
                positive_group_coverage_score_gradient(
                    student_scores=student,
                    positive_indices=positives["positive_indices"],
                    positive_group_ids=positives["positive_group_ids"],
                    budget=args.budget,
                    temperature=args.temperature,
                    epsilon=args.epsilon,
                )
            )
            model.zero_grad(set_to_none=True)
            backward_stream(model, meta, device, args.budget, coverage_gradient)
            grad = parameter_grad_norm(model)
            nonzero_gradient |= grad > 0
            audits.append(
                {
                    "image_id": int(row["image_id"]),
                    "candidate_count": int(meta["candidate_count"]),
                    "identity_max_abs": identity,
                    "coverage_loss": float(coverage_loss),
                    "soft_mass_error": abs(
                        float(diagnostics["soft_mass"]) - int(meta["effective_budget"])
                    ),
                    "positive_pairs": int(diagnostics["positive_pairs"]),
                    "positive_groups": int(diagnostics["positive_groups"]),
                    "residual_parameter_grad_norm": grad,
                }
            )
        payload = {
            "schema_version": 1,
            "passed": all(row["identity_max_abs"] == 0 for row in audits)
            and nonzero_gradient,
            "hard_quota_training_signal": False,
            "rpcm_supervision": False,
            "predicate_supervision": False,
            "unannotated_negative_supervision": False,
            "temperature": args.temperature,
            "epsilon": args.epsilon,
            "coverage_weight": args.coverage_weight,
            "images": audits,
            "test_untouched": True,
        }
        atomic_json(args.output_dir / "q5_numerical_audit.json", payload)
        print(json.dumps(payload, indent=2))
        if not payload["passed"]:
            raise RuntimeError("STAR-Q5 numerical audit failed")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    initial_digest = module_state_digest(model)
    rng = random.Random(args.seed)
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(rows)))
        rng.shuffle(order)
        totals = {
            key: 0.0
            for key in (
                "total", "anchor", "coverage", "grad", "soft_mass_error",
                "micro_positive_coverage", "macro_group_coverage",
                "minimum_group_coverage",
            )
        }
        supervised = 0
        for position, index in enumerate(order, 1):
            row = rows[index]
            image_id = int(row["image_id"])
            meta = torch.load(args.context_cache / row["path"], map_location="cpu", weights_only=False)
            positives = torch.load(
                args.positive_cache / positive_rows[image_id]["path"],
                map_location="cpu", weights_only=False,
            )
            student, base = materialize_scores(model, meta, device, args.budget)
            anchor_loss, anchor_gradient = released_score_anchor_gradient(student, base)
            coverage_loss = student.new_zeros(())
            coverage_gradient = torch.zeros_like(student)
            diagnostics = {
                "positive_pairs": student.new_zeros((), dtype=torch.long),
                "micro_positive_coverage": student.new_zeros(()),
                "macro_group_coverage": student.new_zeros(()),
                "minimum_group_coverage": student.new_zeros(()),
                "soft_mass": student.new_tensor(float(meta["effective_budget"])),
            }
            if args.mode == "budgeted-coverage":
                coverage_loss, coverage_gradient, diagnostics = (
                    positive_group_coverage_score_gradient(
                        student_scores=student,
                        positive_indices=positives["positive_indices"],
                        positive_group_ids=positives["positive_group_ids"],
                        budget=args.budget,
                        temperature=args.temperature,
                        epsilon=args.epsilon,
                    )
                )
            total_loss = anchor_loss + float(args.coverage_weight) * coverage_loss
            score_gradient = anchor_gradient + float(args.coverage_weight) * coverage_gradient
            if not torch.isfinite(total_loss) or not torch.isfinite(score_gradient).all():
                raise FloatingPointError(f"Non-finite Q5 objective on image {image_id}")
            optimizer.zero_grad(set_to_none=True)
            backward_stream(model, meta, device, args.budget, score_gradient)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
            optimizer.step()
            totals["total"] += float(total_loss)
            totals["anchor"] += float(anchor_loss)
            totals["coverage"] += float(coverage_loss)
            totals["grad"] += norm
            totals["soft_mass_error"] += abs(
                float(diagnostics["soft_mass"]) - int(meta["effective_budget"])
            )
            if int(diagnostics["positive_pairs"]) > 0:
                supervised += 1
                for key in (
                    "micro_positive_coverage", "macro_group_coverage",
                    "minimum_group_coverage",
                ):
                    totals[key] += float(diagnostics[key])
            if position % 25 == 0 or position == len(order):
                print(
                    f"[STAR-Q5 {args.mode}] seed={args.seed} epoch={epoch}/{args.epochs} "
                    f"image={position}/{len(order)} total={float(total_loss):.6g} "
                    f"anchor={float(anchor_loss):.6g} coverage={float(coverage_loss):.6g}",
                    flush=True,
                )
            del student, base, anchor_gradient, coverage_gradient, score_gradient
        denominator = max(len(order), 1)
        epoch_row = {
            "epoch": epoch,
            "total": totals["total"] / denominator,
            "anchor": totals["anchor"] / denominator,
            "coverage": totals["coverage"] / denominator,
            "grad": totals["grad"] / denominator,
            "soft_mass_error": totals["soft_mass_error"] / denominator,
            "micro_positive_coverage": totals["micro_positive_coverage"] / max(supervised, 1),
            "macro_group_coverage": totals["macro_group_coverage"] / max(supervised, 1),
            "minimum_group_coverage": totals["minimum_group_coverage"] / max(supervised, 1),
            "supervised_images": supervised,
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row), flush=True)
        torch.cuda.empty_cache()

    checkpoint = args.output_dir / "residual_final.pth"
    metadata = {
        "round": "STAR-Q5",
        "mode": args.mode,
        "seed": args.seed,
        "released_ppg_frozen": True,
        "hard_quota_training_signal": False,
        "rpcm_supervision": False,
        "predicate_supervision": False,
        "unannotated_negative_supervision": False,
        "objective": (
            "released score mean-MSE anchor"
            if args.mode == "anchor"
            else "anchor - mean_positive_group log(eps + soft_topk_coverage)"
        ),
        "budget": args.budget,
        "temperature": args.temperature,
        "epsilon": args.epsilon,
        "coverage_weight": args.coverage_weight,
        "optimizer": {
            "name": "Adam", "lr": args.lr, "weight_decay": 0.0,
            "epochs": args.epochs, "gradient_clip_l2": args.grad_clip,
        },
        "context_cache_digest": context_manifest["q5_context_digest"],
        "positive_cache_digest": positive_manifest["positive_digest"],
        "released_ppg_sha256": context_manifest["ppg_checkpoint_sha256"],
        "initial_state_digest": initial_digest,
        "final_state_digest": module_state_digest(model),
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "test_untouched": True,
    }
    save_set_conditioned_checkpoint(checkpoint, model, metadata=metadata)
    summary = {
        **metadata,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    atomic_json(args.output_dir / "q5_training_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
