"""Export frozen DRR TRAIN evidence for predicate-decision calibration.

The frozen model runs a declared Top-K proposal graph used by the target
inference protocol.  The historical default is Released PPG; the controlled
proposal-matched PredCls experiment uses a frozen BAP TRAIN pair manifest,
while SGCls can recompute the frozen BAP/Q5 scorer online on its GT-box object
universe. SGDet uses the same online scorer on frozen detector proposals and
labels retained proposal pairs by deterministic GT-endpoint matching.
Only TRAIN annotations label retained pairs; validation/test statistics are
never read. Unannotated candidates are never treated as negatives: a bounded
subset is stored only for base-distribution preservation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgg.data.build import build_dataloaders, build_datasets  # noqa: E402
from sgg.engine import Trainer  # noqa: E402
from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector  # noqa: E402
from sgg.structures.boxlist_ops import boxlist_iou  # noqa: E402
from sgg.modeling.roi_heads.sparse_competition_calibration import (  # noqa: E402
    build_group_manifest,
    save_group_manifest,
)
from tools.eval_once import (  # noqa: E402
    apply_runtime_cfg,
    load_model_only_checkpoint,
    load_py_config,
)


CACHE_SCHEMA = "sparse-competition-train-cache-v2"
LEGACY_CACHE_SCHEMA = "sparse-competition-train-cache-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _relation_head(model):
    roi_heads = model.roi_heads
    if hasattr(roi_heads, "relation"):
        return roi_heads.relation
    if isinstance(roi_heads, (dict, torch.nn.ModuleDict)) and "relation" in roi_heads:
        return roi_heads["relation"]
    raise AttributeError("SceneGraphDetector has no relation ROI head")


def _image_id(meta: dict, target) -> int:
    for key in ("source_image_id", "image_id"):
        if key in meta:
            return int(meta[key])
    if target.has_field("image_id"):
        return int(target.get_field("image_id").reshape(-1)[0])
    raise ValueError("cache export requires stable TRAIN image IDs")


def audit_proposal_contract(rel_cfg: dict, proposal_mode: str) -> dict[str, object]:
    """Validate and describe the graph used for frozen TRAIN feature export."""

    mode = str(proposal_mode).strip().lower()
    if mode not in {"released_ppg", "bap", "q5_online"}:
        raise ValueError(
            "proposal_mode must be released_ppg, bap, or q5_online"
        )
    override = str(rel_cfg.get("PAIR_OVERRIDE_ARTIFACT_DIR", "")).strip()
    if mode == "released_ppg":
        if str(rel_cfg.get("TEST_FILTER_METHOD", "")).upper() != "PPG":
            raise ValueError("Released-PPG cache export requires TEST_FILTER_METHOD=PPG")
        if int(rel_cfg.get("PPG_TOPK", 0)) != 10_000:
            raise ValueError("Released-PPG cache export requires Top-10000")
        if override:
            raise ValueError("Released-PPG cache export forbids a frozen pair override")
        return {
            "mode": mode,
            "name": "released_ppg_top10000",
            "pair_override": False,
        }

    if mode == "q5_online":
        if str(rel_cfg.get("TEST_FILTER_METHOD", "")).upper() != "Q5":
            raise ValueError("Online-Q5 cache export requires TEST_FILTER_METHOD=Q5")
        if int(rel_cfg.get("Q5_TOPK", 0)) != 10_000:
            raise ValueError("Online-Q5 cache export requires Top-10000")
        if override:
            raise ValueError("Online-Q5 cache export forbids a frozen pair override")
        checkpoint = Path(str(rel_cfg.get("Q5_CHECKPOINT", "")))
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        return {
            "mode": mode,
            "name": "online_q5_bap_seed1029_top10000",
            "pair_override": False,
            "q5_checkpoint": str(checkpoint.resolve()),
            "q5_checkpoint_sha256": _sha256(checkpoint),
        }

    if int(rel_cfg.get("PPG_TOPK", 0)) != 10_000:
        raise ValueError("BAP pair-manifest cache export requires Top-10000")
    if not override:
        raise ValueError("BAP-matched cache export requires PAIR_OVERRIDE_ARTIFACT_DIR")
    manifest_path = Path(override) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split") != "train":
        raise ValueError("BAP-matched pair manifest must use the TRAIN split")
    if manifest.get("variant") != "budgeted-coverage":
        raise ValueError("BAP-matched pair manifest must be the budgeted-coverage variant")
    if int(manifest.get("budget", -1)) != 10_000:
        raise ValueError("BAP-matched pair manifest must use Top-10000")
    if manifest.get("filter_method") != "STAR_Q5_HARD_QUOTA_FREE_TOPK":
        raise ValueError("BAP-matched pair manifest has an unexpected filter method")
    if not manifest.get("images"):
        raise ValueError("BAP-matched pair manifest is empty")
    expected_hash = str(rel_cfg.get("PAIR_OVERRIDE_MANIFEST_SHA256", "")).strip()
    actual_hash = _sha256(manifest_path)
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError(
            "BAP TRAIN pair manifest hash mismatch: "
            f"expected={expected_hash}, actual={actual_hash}"
        )
    return {
        "mode": mode,
        "name": "bap_seed1029_top10000",
        "pair_override": True,
        "pair_manifest": str(manifest_path.resolve()),
        "pair_manifest_sha256": actual_hash,
        "pair_manifest_images": len(manifest["images"]),
        "inference_rule": manifest.get("inference_rule"),
    }


def label_sgdet_pairs(
    processor,
    proposal,
    target,
    pair_idx: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Label frozen SGDet pairs by the task's class-and-IoU endpoint contract.

    This is the no-sampling analogue of ``_sample_detect_rpcm``. Each GT
    relation endpoint can match multiple detector proposals with the same
    effective proposal label and IoU above the configured foreground
    threshold. Only captured directed pairs receive a foreground predicate;
    every other captured pair remains unannotated for preservation sampling.
    """

    device = pair_idx.device
    pair_count = int(pair_idx.size(0))
    labels = torch.zeros((pair_count,), dtype=torch.long, device=device)
    if (
        pair_count == 0
        or len(proposal) == 0
        or len(target) == 0
        or not target.has_field("relation_triplets")
    ):
        return labels, {
            "proposals": int(len(proposal)),
            "matched_proposals": 0,
            "positive_pairs": 0,
        }

    if proposal.has_field("filter_labels"):
        proposal_labels = proposal.get_field("filter_labels").long().to(device)
    elif proposal.has_field("gt_labels"):
        proposal_labels = proposal.get_field("gt_labels").long().to(device)
    else:
        proposal_labels = proposal.get_field("labels").long().to(device)
    if proposal_labels.numel() != len(proposal):
        raise ValueError("SGDet effective labels do not align with proposals")
    target_labels = target.get_field("labels").long().to(device)
    ious = boxlist_iou(
        target,
        proposal,
        mode=(
            "obb"
            if target.mode == "xywha" and proposal.mode == "xywha"
            else "hbb"
        ),
    ).to(device)
    is_match = (
        (target_labels[:, None] == proposal_labels[None, :])
        & (proposal_labels[None, :] > 0)
        & (ious > float(processor.fg_thres))
    )

    num_proposals = len(proposal)
    pair_rows = torch.full(
        (num_proposals, num_proposals),
        -1,
        dtype=torch.long,
        device=device,
    )
    pair_rows[pair_idx[:, 0].long(), pair_idx[:, 1].long()] = torch.arange(
        pair_count, device=device
    )
    relation_triplets = target.get_field("relation_triplets").long().to(device)
    for gt_subject, gt_object, predicate in relation_triplets:
        subject_matches = torch.nonzero(
            is_match[int(gt_subject)], as_tuple=False
        ).flatten()
        object_matches = torch.nonzero(
            is_match[int(gt_object)], as_tuple=False
        ).flatten()
        if subject_matches.numel() == 0 or object_matches.numel() == 0:
            continue
        subject_grid = subject_matches[:, None].expand(
            -1, object_matches.numel()
        ).reshape(-1)
        object_grid = object_matches[None, :].expand(
            subject_matches.numel(), -1
        ).reshape(-1)
        nonself = subject_grid != object_grid
        rows = pair_rows[subject_grid[nonself], object_grid[nonself]]
        rows = rows[rows >= 0]
        if rows.numel() > 0:
            # Preserve the relation-matrix last-write behavior used elsewhere.
            labels[rows] = int(predicate)

    return labels, {
        "proposals": int(num_proposals),
        "matched_proposals": int(is_match.any(dim=0).sum().item()),
        "positive_pairs": int(labels.gt(0).sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-groups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1029)
    parser.add_argument(
        "--proposal-mode",
        choices=("released_ppg", "bap", "q5_online"),
        default="released_ppg",
        help="Frozen proposal graph used to export relation features.",
    )
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--preserve-hard-budget", type=int, default=128)
    parser.add_argument("--preserve-random-budget", type=int, default=128)
    parser.add_argument(
        "--cache-dtype", choices=("float16", "float32"), default="float16"
    )
    args = parser.parse_args()

    if not args.device.startswith("cuda"):
        raise ValueError("cache export requires CUDA; CPU fallback is disabled")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    if int(args.batch_size) != 1:
        raise ValueError(
            "competition cache export is intentionally per-image; "
            "--batch-size must equal 1"
        )
    if int(args.preserve_hard_budget) < 0 or int(args.preserve_random_budget) < 0:
        raise ValueError("preservation budgets must be non-negative")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"cache output is not empty: {output}")
    (output / "shards").mkdir(parents=True, exist_ok=True)

    _seed(int(args.seed))
    cfg = load_py_config(args.config)
    apply_runtime_cfg(cfg)
    rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
    predictor_name = str(rel_cfg.get("PREDICTOR", "")).upper()
    if predictor_name not in {
        "RPCM_ORIGINAL_LEGACY",
        "ORIGINAL_RPCM_LEGACY",
        "RPCM_NATIVE_LEGACY",
    }:
        raise ValueError("cache export requires the checkpoint-compatible RPCMLegacy path")
    if bool(rel_cfg.get("COMPETITION_CALIBRATION", {}).get("ENABLED", False)):
        raise ValueError("cache must be exported from the uncalibrated base")
    task = str(cfg["MODEL"].get("TASK", "")).lower()
    if task not in {"predcls", "sgcls", "sgdet"}:
        raise ValueError(
            "competition cache export supports PredCls, SGCls, and SGDet; "
            f"got {task!r}"
        )
    proposal_contract = audit_proposal_contract(rel_cfg, args.proposal_mode)

    cfg["DATASETS"]["TRAIN"]["AUGMENT"] = False
    cfg["DATALOADER"]["TRAIN_BATCH_SIZE"] = int(args.batch_size)
    # ``build_dataloaders`` gives the historical RPCM solver batch precedence
    # over DATALOADER.TRAIN_BATCH_SIZE.  Cache inference must be per-image (as
    # val/test is), otherwise RPCMLegacy concatenates up to 16 Top-10k graphs
    # before constructing its dense relation adjacency.
    cfg["SOLVER"]["IMS_PER_BATCH"] = int(args.batch_size)
    cfg["DATALOADER"]["NUM_WORKERS"] = min(
        int(cfg["DATALOADER"].get("NUM_WORKERS", 4)), 4
    )
    datasets = build_datasets(cfg, splits=("train",))
    dataset = datasets["train"]
    metadata = dataset.metadata
    object_names = [metadata.categories[index] for index in sorted(metadata.categories)]
    relation_names = [metadata.predicates[index] for index in sorted(metadata.predicates)]
    cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"] = object_names
    rel_cfg["RELATION_NAMES"] = relation_names
    loaders = build_dataloaders(
        cfg,
        splits=("train",),
        datasets=datasets,
        shuffle_map={"train": False},
    )
    model = SceneGraphDetector(cfg)
    trainer = Trainer(cfg, model, device=args.device, dataloaders=loaders)
    load_model_only_checkpoint(trainer, str(checkpoint_path), legacy_rpcm=False)
    trainer.model.eval()
    head = _relation_head(trainer.model)
    predictor = head.predictor
    if not hasattr(predictor, "set_competition_calibration_capture"):
        raise TypeError("RPCMLegacy does not expose competition-calibration capture")
    predictor.set_competition_calibration_capture(True)

    num_classes = len(relation_names)
    train_class_counts = torch.zeros(num_classes, dtype=torch.long)
    feature_class_counts = torch.zeros(num_classes, dtype=torch.long)
    centroid_sum: torch.Tensor | None = None
    shard_rows: list[dict[str, object]] = []
    total_candidates = 0
    total_positive = 0
    total_preserve = 0
    processed_images = 0
    sgdet_total_proposals = 0
    sgdet_matched_proposals = 0
    sgdet_images_with_positive = 0
    state: dict[str, torch.Tensor] = {}
    cache_dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32

    with torch.inference_mode():
        for shard_id, (images, targets, metas) in enumerate(
            tqdm(loaders["train"], desc="Sparse competition frozen TRAIN cache")
        ):
            if int(args.max_images) >= 0 and processed_images >= int(args.max_images):
                break
            processed_images += len(targets)
            images = images.to(trainer.device)
            moved_targets = trainer._move_targets(targets)
            detector_images, detector_targets = trainer._sgdet_detector_inputs_from_metas(
                metas
            )
            predictions = trainer.model(
                images,
                moved_targets,
                detector_images=detector_images,
                detector_targets=detector_targets,
            )
            state = predictor.competition_calibration_capture_state()
            required = {
                "rel_pair_idxs",
                "relation_features",
                "union_features",
                "base_relation_logits",
            }
            missing = sorted(required.difference(state))
            if missing:
                raise RuntimeError(f"incomplete competition cache capture: {missing}")
            edge_count = int(state["rel_pair_idxs"].size(0))
            for key in ("relation_features", "union_features", "base_relation_logits"):
                if int(state[key].size(0)) != edge_count:
                    raise RuntimeError(f"capture row mismatch for {key}")
            total_candidates += edge_count

            label_parts: list[torch.Tensor] = []
            image_parts: list[torch.Tensor] = []
            offset = 0
            for prediction, target, meta in zip(predictions, moved_targets, metas):
                pairs = prediction.get_field("rel_pair_idxs").long()
                if int(pairs.size(0)) > 10_000:
                    raise RuntimeError(
                        "cache inference violated released PPG Top-10000: "
                        f"image has {int(pairs.size(0))} pairs"
                    )
                captured = state["rel_pair_idxs"][
                    offset : offset + pairs.size(0)
                ].long()
                entity_count = max(len(prediction), 1)
                post_encoded = pairs[:, 0] * entity_count + pairs[:, 1]
                capture_encoded = captured[:, 0] * entity_count + captured[:, 1]
                if (
                    torch.unique(post_encoded).numel() != post_encoded.numel()
                    or torch.unique(capture_encoded).numel() != capture_encoded.numel()
                    or not torch.equal(
                        torch.sort(post_encoded).values,
                        torch.sort(capture_encoded.to(post_encoded.device)).values,
                    )
                ):
                    raise RuntimeError("pre/postprocess pair sets differ")
                relation_matrix = head.samp_processor._get_relation_matrix(target).to(
                    captured.device
                )
                all_gt = relation_matrix.reshape(-1).long()
                all_gt = all_gt[all_gt > 0]
                if all_gt.numel() > 0:
                    train_class_counts += torch.bincount(
                        all_gt.cpu(), minlength=num_classes
                    )[:num_classes]
                if task == "sgdet":
                    captured_labels, match_diagnostics = label_sgdet_pairs(
                        head.samp_processor,
                        prediction,
                        target,
                        captured,
                    )
                    sgdet_total_proposals += int(
                        match_diagnostics["proposals"]
                    )
                    sgdet_matched_proposals += int(
                        match_diagnostics["matched_proposals"]
                    )
                    sgdet_images_with_positive += int(
                        match_diagnostics["positive_pairs"] > 0
                    )
                else:
                    captured_labels, _ = head.samp_processor.label_gtbox_pairs(
                        prediction,
                        target,
                        captured,
                    )
                label_parts.append(captured_labels.long())
                image_parts.append(
                    torch.full(
                        (captured.size(0),),
                        _image_id(meta, target),
                        dtype=torch.long,
                        device=captured.device,
                    )
                )
                offset += int(pairs.size(0))
            labels = torch.cat(label_parts, dim=0)
            image_ids = torch.cat(image_parts, dim=0)
            positive_rows = torch.nonzero(labels > 0, as_tuple=False).flatten()
            unannotated_rows = torch.nonzero(labels == 0, as_tuple=False).flatten()
            preserve_parts: list[torch.Tensor] = []
            if unannotated_rows.numel() > 0 and int(args.preserve_hard_budget) > 0:
                base_logits = state["base_relation_logits"].index_select(
                    0, unannotated_rows
                )
                foreground_confidence = F.softmax(base_logits.float(), dim=1)[
                    :, 1:
                ].max(dim=1).values
                confidence_order = torch.argsort(
                    foreground_confidence, descending=True, stable=True
                )
                hard_count = min(
                    int(args.preserve_hard_budget), int(unannotated_rows.numel())
                )
                hard_rows = unannotated_rows.index_select(
                    0, confidence_order[:hard_count]
                )
                preserve_parts.append(hard_rows)
            else:
                hard_rows = unannotated_rows.new_zeros((0,), dtype=torch.long)

            if unannotated_rows.numel() > hard_rows.numel() and int(
                args.preserve_random_budget
            ) > 0:
                remaining_mask = torch.ones(
                    (unannotated_rows.numel(),),
                    dtype=torch.bool,
                    device=unannotated_rows.device,
                )
                if hard_rows.numel() > 0:
                    remaining_mask &= ~torch.isin(unannotated_rows, hard_rows)
                remaining = unannotated_rows[remaining_mask]
                image_seed = int(image_ids[0].item()) if image_ids.numel() else 0
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(args.seed) * 1_000_003 + image_seed)
                order = torch.randperm(
                    remaining.numel(), generator=generator, device="cpu"
                ).to(device=remaining.device)
                random_count = min(
                    int(args.preserve_random_budget), int(remaining.numel())
                )
                preserve_parts.append(
                    remaining.index_select(0, order[:random_count])
                )
            preserve_rows = (
                torch.cat(preserve_parts, dim=0)
                if preserve_parts
                else labels.new_zeros((0,), dtype=torch.long)
            )
            selected = torch.cat([positive_rows, preserve_rows], dim=0)
            if selected.numel() == 0:
                continue
            selected_labels = labels.index_select(0, selected).long()
            if preserve_rows.numel() > 0:
                selected_labels[positive_rows.numel() :] = -1
            relation_features = state["relation_features"].index_select(
                0, selected
            ).float().cpu()
            if centroid_sum is None:
                centroid_sum = torch.zeros(
                    num_classes, relation_features.size(1), dtype=torch.float64
                )
            positive_labels = selected_labels[: positive_rows.numel()]
            positive_features = relation_features[: positive_rows.numel()]
            for class_id in positive_labels.unique().tolist():
                class_rows = positive_labels.eq(int(class_id))
                centroid_sum[int(class_id)] += positive_features[
                    class_rows.cpu()
                ].double().sum(dim=0)
                feature_class_counts[int(class_id)] += int(class_rows.sum())

            payload = {
                "relation_features": relation_features.to(cache_dtype),
                "union_features": state["union_features"].index_select(
                    0, selected
                ).to(cache_dtype).cpu(),
                "original_logits": state["base_relation_logits"].index_select(
                    0, selected
                ).float().cpu(),
                "labels": selected_labels.to(torch.int16).cpu(),
                "supervision_mask": selected_labels.gt(0).cpu(),
                "image_ids": image_ids.index_select(0, selected).cpu(),
            }
            shard_path = output / "shards" / f"shard_{shard_id:05d}.pt"
            torch.save(payload, shard_path)
            rows = int(selected.numel())
            positive_count = int(positive_rows.numel())
            preserve_count = int(preserve_rows.numel())
            total_positive += positive_count
            total_preserve += preserve_count
            shard_rows.append(
                {
                    "path": str(shard_path.relative_to(output)),
                    "rows": rows,
                    "positive_rows": positive_count,
                    "preserve_rows": preserve_count,
                    "sha256": _sha256(shard_path),
                }
            )

    predictor.set_competition_calibration_capture(False)
    if centroid_sum is None or total_positive <= 0:
        raise RuntimeError("no positive TRAIN relationship survived the declared Top-10000 graph")
    centroids = centroid_sum.float() / feature_class_counts.clamp_min(1).float().unsqueeze(1)
    base_hash = _sha256(checkpoint_path)
    group_manifest = build_group_manifest(
        centroids,
        train_class_counts.tolist(),
        class_feature_counts=feature_class_counts.tolist(),
        relation_names=relation_names,
        union_dim=int(state["union_features"].size(1)),
        num_groups=int(args.num_groups),
        seed=int(args.seed),
        source={
            "dataset": "STAR",
            "split": "train",
            "task": task,
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": base_hash,
            "proposal": proposal_contract["name"],
            "proposal_contract": proposal_contract,
            "candidate_rows": total_candidates,
            "positive_rows": total_positive,
            "preserve_rows": total_preserve,
            "images": processed_images,
        },
    )
    group_path = output / "group_manifest.json"
    save_group_manifest(group_manifest, group_path)
    cache_manifest = {
        "schema": CACHE_SCHEMA,
        "split": "train",
        "task": task,
        "validation_or_test_labels_used": False,
        "base_model_frozen": True,
        "seed": int(args.seed),
        "config": str(Path(args.config).resolve()),
        "base_checkpoint": str(checkpoint_path.resolve()),
        "base_checkpoint_sha256": base_hash,
        "proposal": proposal_contract["name"],
        "proposal_contract": proposal_contract,
        "candidate_rows": total_candidates,
        "positive_rows": total_positive,
        "preserve_rows": total_preserve,
        "images": processed_images,
        "max_images": int(args.max_images),
        "relation_dim": int(state["relation_features"].size(1)),
        "union_dim": int(state["union_features"].size(1)),
        "num_rel_classes": num_classes,
        "cache_dtype": str(args.cache_dtype),
        "class_counts": train_class_counts.tolist(),
        "feature_class_counts": feature_class_counts.tolist(),
        "group_manifest": group_path.name,
        "group_manifest_hash": group_manifest["content_hash"],
        "sampling_contract": (
            "annotated_foreground_correction_plus_unannotated_base_preservation"
        ),
        "sgdet_pair_labeling": (
            {
                "enabled": True,
                "definition": (
                    "GT relation endpoints matched to frozen detector proposals "
                    "by effective class and IoU > relation FG threshold; no sampling"
                ),
                "foreground_iou_threshold": float(head.samp_processor.fg_thres),
                "total_proposals": int(sgdet_total_proposals),
                "matched_proposals": int(sgdet_matched_proposals),
                "images_with_positive_pairs": int(sgdet_images_with_positive),
            }
            if task == "sgdet"
            else {"enabled": False}
        ),
        "unannotated_pairs_used_as_negatives": False,
        "preserve_hard_budget_per_image": int(args.preserve_hard_budget),
        "preserve_random_budget_per_image": int(args.preserve_random_budget),
        "shards": shard_rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(cache_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        {
            "output": str(output),
            "candidate_rows": total_candidates,
            "positive_rows": total_positive,
            "preserve_rows": total_preserve,
            "groups": group_manifest["groups"],
            "group_hash": group_manifest["content_hash"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
