#!/usr/bin/env python3
"""Export ordinary global Top-10k graphs from a STAR-Q5 residual scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgg.modeling.roi_heads.ppg_budget_adaptation import stable_descending_order
from sgg.modeling.roi_heads.residual_ppg_scorer import SetConditionedPPGResidual
from sgg.modeling.roi_heads.set_context import (
    Q5_SET_CONTEXT_FEATURE_NAMES,
    q5_set_context_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--budget", type=int, default=10_000)
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def context(shard, meta, device, budget):
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


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    manifest = json.loads((args.context_cache / "manifest.json").read_text())
    if manifest.get("split") not in {"train", "val"} or bool(
        manifest.get("contains_relation_gt", True)
    ):
        raise ValueError("Q5 export requires target-free train/val context")
    if bool(manifest.get("contains_hard_quota_allocation", True)) or bool(
        manifest.get("hard_quota_fields_consumed", True)
    ):
        raise ValueError("Q5 export context contains HardQuota signal")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SetConditionedPPGResidual.from_checkpoint_payload(payload).to(args.device).eval()
    metadata = payload.get("metadata", {})
    if metadata.get("released_ppg_sha256") != manifest["ppg_checkpoint_sha256"]:
        raise ValueError("Q5 residual/context PPG hashes differ")
    if list(payload["architecture"].get("feature_names", [])) != list(
        Q5_SET_CONTEXT_FEATURE_NAMES
    ):
        raise ValueError("Checkpoint is not a Q5 HardQuota-free context model")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows, diagnostics = [], []
    digest = hashlib.sha256()
    total_seconds = 0.0
    total_edges = 0
    cuda_peak = 0
    for ordinal, row in enumerate(manifest["images"], 1):
        meta = torch.load(args.context_cache / row["path"], map_location="cpu", weights_only=False)
        pair_parts, base_parts, score_parts, residual_parts = [], [], [], []
        torch.cuda.reset_peak_memory_stats()
        for shard_row in meta["shards"]:
            shard = torch.load(shard_row["path"], map_location="cpu", weights_only=False)
            base = shard["base_scores"].to(args.device, dtype=torch.float32)
            features = context(shard, meta, torch.device(args.device), args.budget)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                correction = model(features)
            torch.cuda.synchronize()
            total_seconds += time.perf_counter() - started
            pair_parts.append(shard["pairs"].long())
            base_parts.append(base.cpu())
            residual_parts.append(correction.cpu())
            score_parts.append((base + correction).cpu())
        pairs = torch.cat(pair_parts)
        base_scores = torch.cat(base_parts)
        scores = torch.cat(score_parts)
        residuals = torch.cat(residual_parts)
        order = stable_descending_order(scores, pairs).cpu()
        selected_indices = order[: min(args.budget, order.numel())]
        selected = pairs[selected_indices]
        base_order = stable_descending_order(base_scores, pairs).cpu()
        base_top = set(base_order[: min(args.budget, base_order.numel())].tolist())
        current_top = set(selected_indices.tolist())
        overlap = len(base_top & current_top)
        current_rank = torch.empty_like(order)
        current_rank[order] = torch.arange(order.numel())
        path = args.output_dir / f"{int(meta['image_id']):04d}.pt"
        torch.save(
            {
                "schema_version": 1,
                "image_id": int(meta["image_id"]),
                "split": manifest["split"],
                "filter_method": "STAR_Q5_HARD_QUOTA_FREE_TOPK",
                "prediction": {
                    "rel_pair_idxs": selected,
                    "selected_original_indices": selected_indices.to(torch.int32),
                    "student_rank": current_rank.to(torch.int32),
                },
            },
            path,
        )
        output_rows.append({"image_id": int(meta["image_id"]), "path": path.name})
        diagnostics.append(
            {
                "image_id": int(meta["image_id"]),
                "candidate_count": int(meta["candidate_count"]),
                "top10k_overlap": overlap,
                "top10k_jaccard": overlap / max(len(base_top | current_top), 1),
                "replaced_edges": len(current_top - base_top),
                "residual_abs_mean": float(residuals.abs().mean()) if residuals.numel() else 0.0,
                "residual_min": float(residuals.min()) if residuals.numel() else 0.0,
                "residual_max": float(residuals.max()) if residuals.numel() else 0.0,
            }
        )
        digest.update(int(meta["image_id"]).to_bytes(8, "little", signed=True))
        digest.update(selected.contiguous().numpy().tobytes())
        total_edges += int(scores.numel())
        cuda_peak = max(cuda_peak, int(torch.cuda.max_memory_allocated()))
        print(
            f"[STAR-Q5 export] {ordinal}/{len(manifest['images'])} "
            f"image={meta['image_id']} replaced={len(current_top-base_top)}",
            flush=True,
        )
        torch.cuda.empty_cache()
    export = {
        "schema_version": 1,
        "split": manifest["split"],
        "variant": metadata.get("mode"),
        "filter_method": "STAR_Q5_HARD_QUOTA_FREE_TOPK",
        "inference_rule": "Released PPG + Q5 residual -> ordinary global Top-10000",
        "budget": args.budget,
        "images": output_rows,
        "pair_digest": digest.hexdigest(),
        "released_ppg_sha256": manifest["ppg_checkpoint_sha256"],
        "residual_checkpoint": str(args.checkpoint),
        "residual_checkpoint_sha256": sha256(args.checkpoint),
        "hard_quota_training_signal": False,
        "test_untouched": True,
    }
    atomic_json(args.output_dir / "manifest.json", export)
    atomic_json(
        args.output_dir / "selection_diagnostics.json",
        {
            "images": diagnostics,
            "macro": {
                key: float(np.mean([row[key] for row in diagnostics]))
                for key in (
                    "top10k_overlap", "top10k_jaccard", "replaced_edges",
                    "residual_abs_mean", "residual_min", "residual_max",
                )
            },
        },
    )
    atomic_json(
        args.output_dir / "efficiency.json",
        {
            "residual_scoring_seconds": total_seconds,
            "residual_scoring_edges": total_edges,
            "residual_edges_per_second": total_edges / max(total_seconds, 1e-12),
            "cuda_peak_bytes": cuda_peak,
            "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        },
    )
    print(json.dumps(export, indent=2))


if __name__ == "__main__":
    main()
