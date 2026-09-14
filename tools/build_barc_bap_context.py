#!/usr/bin/env python3
"""Build Hard-Quota-free Q5 context metadata over existing target-free shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgg.modeling.roi_heads.set_context import Q5_SET_CONTEXT_FEATURE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--source-context", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    source_manifest = json.loads((args.source_context / "manifest.json").read_text())
    if source_manifest.get("split") != args.split or bool(
        source_manifest.get("contains_relation_gt", True)
    ):
        raise ValueError("Q5 source context must be target-free and match split")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    digest = hashlib.sha256()
    for ordinal, row in enumerate(source_manifest["images"], 1):
        source_meta_path = args.source_context / row["path"]
        source_meta = torch.load(
            source_meta_path, map_location="cpu", weights_only=False
        )
        # Deliberately consume only candidate-derived statistics.  In
        # particular, source_meta["quota_targets"] is never read or copied.
        image_id = int(source_meta["image_id"])
        group_counts = source_meta["group_counts"].to(torch.int32)
        shards = [
            {
                "path": str((source_meta_path.parent / shard["path"]).resolve()),
                "start": int(shard["start"]),
                "end": int(shard["end"]),
                "rows": int(shard["rows"]),
            }
            for shard in source_meta["shards"]
        ]
        meta = {
            "schema_version": 1,
            "kind": "star_q5_hard_quota_free_set_context",
            "split": args.split,
            "image_id": image_id,
            "candidate_count": int(source_meta["candidate_count"]),
            "effective_budget": min(10_000, int(source_meta["candidate_count"])),
            "num_classes": int(source_meta["num_classes"]),
            "active_group_count": int(source_meta["active_group_count"]),
            "group_counts": group_counts,
            "shards": shards,
            "feature_names": list(Q5_SET_CONTEXT_FEATURE_NAMES),
            "contains_relation_gt": False,
            "contains_hard_quota_allocation": False,
            "hard_quota_fields_consumed": False,
        }
        path = args.output_dir / f"{image_id:04d}.pt"
        torch.save(meta, path)
        rows.append(
            {
                "image_id": image_id,
                "path": path.name,
                "candidate_count": int(meta["candidate_count"]),
            }
        )
        digest.update(image_id.to_bytes(8, "little", signed=True))
        digest.update(group_counts.contiguous().numpy().tobytes())
        digest.update(str(source_meta["source_teacher_pair_score_sha256"]).encode())
        print(
            f"[STAR-Q5 context] {ordinal}/{len(source_manifest['images'])} "
            f"image={image_id}",
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "kind": "star_q5_hard_quota_free_set_context",
        "split": args.split,
        "contains_relation_gt": False,
        "contains_hard_quota_allocation": False,
        "hard_quota_fields_consumed": False,
        "images": rows,
        "image_count": len(rows),
        "total_candidates": sum(int(row["candidate_count"]) for row in rows),
        "num_classes": int(source_manifest["num_classes"]),
        "budget": 10_000,
        "feature_names": list(Q5_SET_CONTEXT_FEATURE_NAMES),
        "feature_replacement": (
            "Q4 HardQuota allocation pressure replaced by candidate-only "
            "pressure relative to K/active_groups"
        ),
        "ppg_checkpoint_sha256": source_manifest["ppg_checkpoint_sha256"],
        "teacher_score_digest": source_manifest["teacher_score_digest"],
        "q5_context_digest": digest.hexdigest(),
        "source_target_free_shards": str(args.source_context),
        "test_untouched": True,
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
