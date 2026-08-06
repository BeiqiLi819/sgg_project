"""Evaluate one checkpoint on one split without hidden experiment defaults.

The shell wrappers ``eval_star_{predcls,sgcls,sgdet}.sh`` provide task-specific
defaults.  This lower-level entry intentionally requires config/checkpoint and
supports ``--max-images`` for deterministic smoke tests.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sgg.config.defaults import get_default_cfg
from sgg.data.build import build_dataloaders, build_datasets
from sgg.engine import Trainer
from sgg.modeling.detectors.class_channel_order import (
    BACKGROUND_LAST,
    INTERNAL_DETECTOR_CLASS_ORDER,
    is_detector_classifier_key,
    reorder_detector_classifier_rows,
)
from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector
from sgg.utils.reproducibility import save_run_manifest


def apply_runtime_cfg(cfg):
    runtime_cfg = cfg.get("RUNTIME", {})
    disable_cudnn = bool(runtime_cfg.get("DISABLE_CUDNN", True))
    torch.backends.cudnn.enabled = not disable_cudnn
    torch.backends.cudnn.benchmark = bool(runtime_cfg.get("CUDNN_BENCHMARK", False)) and not disable_cudnn
    torch.backends.cudnn.deterministic = bool(runtime_cfg.get("CUDNN_DETERMINISTIC", True))


def load_py_config(path: str):
    spec = importlib.util.spec_from_file_location("user_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if hasattr(mod, "cfg"):
        return mod.cfg
    if hasattr(mod, "get_cfg"):
        return mod.get_cfg()
    raise AttributeError("Config file must expose `cfg` or `get_cfg()`.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run one evaluation pass on val/test split.")
    parser.add_argument("--config", type=str, required=True, help="Path to python config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=("val", "test"),
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional JSON file to save metrics.",
    )
    parser.add_argument(
        "--filter-method",
        choices=("PPG", "PPN", "RSGP"),
        default=None,
        help="Optional runtime override for the configured pair filter.",
    )
    parser.add_argument(
        "--pair-filter-checkpoint",
        default="",
        help="Checkpoint override for the selected PPG/PPN filter.",
    )
    parser.add_argument(
        "--checkpoint-load-mode",
        choices=("full", "model-only", "legacy-rpcm"),
        default="full",
        help=(
            "full: load model/optimizer/scheduler through Trainer.load_checkpoint; "
            "model-only: load only model tensors; "
            "legacy-rpcm: model-only plus safe key remaps for original RPCM checkpoints."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=-1,
        help="Optional deterministic prefix limit for smoke evaluation.",
    )
    parser.add_argument(
        "--image-ids",
        type=str,
        default="",
        help="Optional comma-separated source image IDs. The dataset is restricted before loading.",
    )
    parser.add_argument(
        "--artifact-output-dir",
        type=str,
        default="",
        help="Optional directory for per-image prediction/target .pt artifacts.",
    )
    return parser.parse_args()


def _parse_image_ids(value: str) -> list[int]:
    if not str(value).strip():
        return []
    image_ids = []
    seen = set()
    for token in str(value).split(","):
        image_id = int(token.strip())
        if image_id < 0:
            raise ValueError(f"image IDs must be non-negative, got {image_id}")
        if image_id not in seen:
            image_ids.append(image_id)
            seen.add(image_id)
    return image_ids


def _restrict_dataset_to_image_ids(dataset, image_ids: list[int]) -> None:
    if not image_ids:
        return
    records = getattr(dataset, "records", None)
    if records is None:
        raise TypeError("--image-ids requires a dataset exposing source records")
    selected = {}
    for record in records:
        image_id = int(record.get("_source_image_id", record.get("image_index", -1)))
        if image_id in image_ids and image_id not in selected:
            selected[image_id] = record
    missing = [image_id for image_id in image_ids if image_id not in selected]
    if missing:
        raise ValueError(f"Requested image IDs are not present in this split: {missing}")
    dataset.records = [selected[image_id] for image_id in image_ids]
    print(f"Restricted {dataset.split} dataset to image_ids={image_ids}", flush=True)


def _boxlist_tensor(boxlist, field: str, *, default=None):
    if boxlist.has_field(field):
        value = boxlist.get_field(field)
        if torch.is_tensor(value):
            return value.detach().cpu()
        return value
    return default


def _serialize_case_boxlist(boxlist, fields: tuple[str, ...]) -> dict:
    payload = {
        "bbox": boxlist.bbox.detach().cpu(),
        "size": tuple(int(v) for v in boxlist.size),
        "mode": str(boxlist.mode),
    }
    for field in fields:
        value = _boxlist_tensor(boxlist, field)
        if value is not None:
            payload[field] = value
    return payload


def _write_case_artifacts(
    artifact_output_dir: Path,
    artifacts: dict,
    *,
    cfg: dict,
    split: str,
    filter_method: str,
) -> None:
    artifact_output_dir.mkdir(parents=True, exist_ok=True)
    prediction_fields = (
        "labels",
        "pred_labels",
        "pred_scores",
        "scores",
        "predict_logits",
        "filter_labels",
        "rel_pair_idxs",
        "pred_rel_scores",
        "pred_rel_labels",
        "base_rel_pair_idxs",
        "sema_rel_pair_idxs",
        "final_rel_pair_idxs",
        "pruned_rel_pair_idxs",
        "ppn_ranked_pair_idxs",
        "degree_capped_pair_idxs",
        "box_angle_unit",
    )
    target_fields = (
        "labels",
        "relation_triplets",
        "all_relation_triplets",
        "image_id",
        "box_angle_unit",
    )
    rows = []
    image_root = Path(cfg["DATASETS"][split.upper()]["IMAGE_ROOT"])
    class_names = list(cfg["MODEL"]["ROI_BOX_HEAD"].get("CLASS_NAMES", []))
    predicate_names = list(cfg["MODEL"]["ROI_RELATION_HEAD"].get("RELATION_NAMES", []))
    for prediction, target, meta in zip(
        artifacts["predictions"], artifacts["targets"], artifacts["metas"]
    ):
        image_id = int(meta.get("source_image_id", meta.get("image_id", -1)))
        file_name = str(meta.get("file_name", f"{image_id:04d}.png"))
        output_path = artifact_output_dir / f"{image_id:04d}.pt"
        torch.save(
            {
                "schema_version": 1,
                "image_id": image_id,
                "split": split,
                "filter_method": filter_method,
                "image_path": str((image_root / file_name).resolve()),
                "raw_width": int(meta.get("source_width", meta.get("width", target.size[0]))),
                "raw_height": int(meta.get("source_height", meta.get("height", target.size[1]))),
                "prediction": _serialize_case_boxlist(prediction, prediction_fields),
                "target": _serialize_case_boxlist(target, target_fields),
                "class_names": class_names,
                "predicate_names": predicate_names,
            },
            output_path,
        )
        rows.append({"image_id": image_id, "path": str(output_path), "file_name": file_name})
    (artifact_output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split": split,
                "filter_method": filter_method,
                "images": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} qualitative artifacts to {artifact_output_dir}", flush=True)


def _checkpoint_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "module"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value
    return ckpt


def _legacy_rpcm_key_candidates(key: str) -> list[str]:
    candidates = [key]
    if ".patch_embed.projection." in key:
        candidates.append(key.replace(".patch_embed.projection.", ".patch_embed.proj."))
    if key.startswith("roi_heads.relation.PPG."):
        candidates.append("roi_heads.relation.ppg." + key[len("roi_heads.relation.PPG."):])
    if key.startswith("roi_heads.relation.union_feature_extractor.feature_extractor.fc6."):
        candidates.append(
            "roi_heads.relation.union_feature_extractor.head.1."
            + key[len("roi_heads.relation.union_feature_extractor.feature_extractor.fc6."):]
        )
    if key.startswith("roi_heads.relation.box_feature_extractor.fc6."):
        candidates.append(
            "roi_heads.relation.box_feature_extractor.head.1."
            + key[len("roi_heads.relation.box_feature_extractor.fc6."):]
        )
    if key.startswith("roi_heads.box.feature_extractor.fc6."):
        suffix = key[len("roi_heads.box.feature_extractor.fc6."):]
        candidates.extend(
            [
                "roi_head.bbox_head.shared_fcs.0." + suffix,
                "roi_head_d2.bbox_head.shared_fcs.0." + suffix,
            ]
        )
    if key.startswith("roi_heads.box.feature_extractor.fc7."):
        suffix = key[len("roi_heads.box.feature_extractor.fc7."):]
        candidates.extend(
            [
                "roi_head.bbox_head.shared_fcs.1." + suffix,
                "roi_head_d2.bbox_head.shared_fcs.1." + suffix,
            ]
        )
    if key.startswith("roi_heads.box.predictor.cls_score."):
        suffix = key[len("roi_heads.box.predictor.cls_score."):]
        candidates.extend(
            [
                "roi_head.bbox_head.fc_cls." + suffix,
                "roi_head_d2.bbox_head.fc_cls." + suffix,
            ]
        )
    return candidates


def load_model_only_checkpoint(trainer: Trainer, path: str, *, legacy_rpcm: bool = False):
    ckpt = torch.load(path, map_location=trainer.device)
    checkpoint_class_order = (
        ckpt.get("detector_class_channel_order") if isinstance(ckpt, dict) else None
    )
    if checkpoint_class_order is None:
        legacy_note = (
            " Applying original-RPCM legacy remaps, including "
            "background_last->background_first for detector classifiers."
            if legacy_rpcm
            else ""
        )
        print(
            "Checkpoint detector class-channel order: <unmarked>; loading unchanged as "
            "background_first for backward compatibility. Migrate explicitly before using "
            f"an old checkpoint for sgcls/sgdet if needed.{legacy_note}",
            flush=True,
        )
    else:
        print(f"Checkpoint detector class-channel order: {checkpoint_class_order}", flush=True)
    source = _checkpoint_state_dict(ckpt)
    if not isinstance(source, dict):
        raise TypeError(f"Checkpoint does not contain a model state_dict: {path}")

    target = trainer.model.state_dict()
    update = {}
    remapped = {}
    skipped_shape = []
    used_source_keys = set()
    reordered_detector_classifier = []
    for source_key, source_value in source.items():
        if source_key not in target:
            continue
        if not hasattr(source_value, "shape") or source_value.shape != target[source_key].shape:
            skipped_shape.append(
                (
                    source_key,
                    source_key,
                    tuple(source_value.shape) if hasattr(source_value, "shape") else "<no-shape>",
                    tuple(target[source_key].shape),
                )
            )
            continue
        value_to_load = source_value
        # Original RPCM checkpoints may carry both the mmrotate detector key
        # (``roi_head...fc_cls``) and the old relation-model alias
        # (``roi_heads.box.predictor.cls_score``).  The direct mmrotate key
        # wins this loop, so it must receive the same background-last ->
        # background-first conversion as the alias path below.
        if legacy_rpcm and is_detector_classifier_key(source_key):
            value_to_load = reorder_detector_classifier_rows(
                source_value,
                source_order=BACKGROUND_LAST,
                target_order=INTERNAL_DETECTOR_CLASS_ORDER,
            )
            reordered_detector_classifier.append((source_key, source_key))
        update[source_key] = value_to_load
        used_source_keys.add(source_key)

    if legacy_rpcm:
        for source_key, source_value in source.items():
            candidates = _legacy_rpcm_key_candidates(source_key)[1:]
            if not candidates:
                continue
            loaded_targets = []
            for target_key in candidates:
                if target_key not in target or target_key in update:
                    continue
                if not hasattr(source_value, "shape") or source_value.shape != target[target_key].shape:
                    skipped_shape.append(
                        (
                            source_key,
                            target_key,
                            tuple(source_value.shape) if hasattr(source_value, "shape") else "<no-shape>",
                            tuple(target[target_key].shape),
                        )
                    )
                    continue
                value_to_load = source_value
                if (
                    source_key.startswith("roi_heads.box.predictor.cls_score.")
                    and is_detector_classifier_key(target_key)
                ):
                    value_to_load = reorder_detector_classifier_rows(
                        source_value,
                        source_order=BACKGROUND_LAST,
                        target_order=INTERNAL_DETECTOR_CLASS_ORDER,
                    )
                    reordered_detector_classifier.append((source_key, target_key))
                update[target_key] = value_to_load
                loaded_targets.append(target_key)
            if loaded_targets:
                used_source_keys.add(source_key)
                remapped[source_key] = loaded_targets

    merged = dict(target)
    merged.update(update)
    incompatible = trainer.model.load_state_dict(merged, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Internal model-only checkpoint load failed unexpectedly: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    filter_prefixes = (
        "roi_heads.relation.ppg.",
        "roi_heads.relation.ppn.",
    )
    unloaded_target = [
        key for key in target
        if key not in update and not key.startswith(filter_prefixes)
    ]
    unused_source = [
        key for key in source
        if key not in used_source_keys and not key.startswith("roi_heads.relation.PPG_HBB.")
    ]
    print(
        "Model-only checkpoint load:",
        {
            "checkpoint": path,
            "loaded": len(update),
            "remapped": len(remapped),
            "unloaded_target": len(unloaded_target),
            "unused_source": len(unused_source),
            "skipped_shape": len(skipped_shape),
            "mode": "legacy-rpcm" if legacy_rpcm else "model-only",
            "reordered_detector_classifier": len(reordered_detector_classifier),
        },
        flush=True,
    )
    if remapped:
        print("Legacy key remaps:", list(remapped.items())[:20], flush=True)
    if unloaded_target:
        print("Unloaded target keys sample:", unloaded_target[:30], flush=True)
    if unused_source:
        print("Unused source keys sample:", unused_source[:30], flush=True)
    if skipped_shape:
        print("Skipped shape mismatch sample:", skipped_shape[:20], flush=True)
    if reordered_detector_classifier:
        print(
            "Legacy detector classifier rows reordered (background_last->background_first):",
            reordered_detector_classifier,
            flush=True,
        )
    return ckpt


def main():
    args = parse_args()

    cfg = get_default_cfg()
    cfg = load_py_config(args.config)
    apply_runtime_cfg(cfg)

    rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
    if args.filter_method is not None:
        rel_cfg["TEST_FILTER_METHOD"] = str(args.filter_method).upper()
    if args.pair_filter_checkpoint:
        method = str(rel_cfg.get("TEST_FILTER_METHOD", "NONE")).upper()
        if method == "PPN":
            rel_cfg["PPN_MODEL_PATH"] = args.pair_filter_checkpoint
        elif method == "PPG":
            rel_cfg["PPG_MODEL_PATH_OBB"] = args.pair_filter_checkpoint
        elif method == "RSGP":
            rel_cfg["PPN_MODEL_PATH"] = args.pair_filter_checkpoint
        else:
            raise ValueError("--pair-filter-checkpoint requires --filter-method PPN, PPG or RSGP")
    output_parent = Path(args.output).parent if args.output else Path(cfg["SOLVER"].get("OUTPUT_DIR", "outputs/default"))
    filter_method = str(rel_cfg.get("TEST_FILTER_METHOD", "NONE")).upper()
    if filter_method == "PPN":
        filter_path = rel_cfg.get("PPN_MODEL_PATH", "")
    elif filter_method == "PPG":
        filter_path = rel_cfg.get("PPG_MODEL_PATH_OBB", "")
    elif filter_method == "RSGP":
        filter_path = (
            f"ppg={rel_cfg.get('PPG_MODEL_PATH_OBB', '')}, "
            f"ppn={rel_cfg.get('PPN_MODEL_PATH', '')}, "
            f"mode={rel_cfg.get('RSGP_MODE', 'HYBRID')}"
        )
    else:
        filter_path = ""
    print(
        f"Resolved pair filter: method={filter_method}, checkpoint={filter_path or '<none>'}",
        flush=True,
    )

    split = args.split.lower()
    datasets = build_datasets(cfg, splits=(split,))
    image_ids = _parse_image_ids(args.image_ids)
    _restrict_dataset_to_image_ids(datasets[split], image_ids)
    split_meta = datasets[split].metadata
    cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"] = [
        split_meta.categories[i] for i in sorted(split_meta.categories.keys())
    ]
    cfg["MODEL"]["ROI_RELATION_HEAD"]["RELATION_NAMES"] = [
        split_meta.predicates[i] for i in sorted(split_meta.predicates.keys())
    ]
    save_run_manifest(
        cfg,
        output_parent,
        config_path=args.config,
        filename="eval_manifest.json",
    )
    dataloaders = build_dataloaders(
        cfg,
        splits=(split,),
        datasets=datasets,
        shuffle_map={split: False},
    )

    model = SceneGraphDetector(cfg)
    trainer = Trainer(cfg, model, device=args.device, dataloaders=dataloaders)
    if args.checkpoint_load_mode == "full":
        trainer.load_checkpoint(args.checkpoint)
    elif args.checkpoint_load_mode == "model-only":
        load_model_only_checkpoint(trainer, args.checkpoint, legacy_rpcm=False)
    elif args.checkpoint_load_mode == "legacy-rpcm":
        load_model_only_checkpoint(trainer, args.checkpoint, legacy_rpcm=True)
    else:
        raise ValueError(f"Unknown checkpoint load mode: {args.checkpoint_load_mode}")

    if args.artifact_output_dir:
        metrics, _, artifacts = trainer.evaluate_loader(
            dataloaders[split],
            return_artifacts=True,
            max_images=int(args.max_images),
        )
        _write_case_artifacts(
            Path(args.artifact_output_dir),
            artifacts,
            cfg=cfg,
            split=split,
            filter_method=filter_method,
        )
    else:
        metrics, _ = trainer.evaluate_loader(
            dataloaders[split],
            return_result=True,
            max_images=int(args.max_images),
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "config": str(Path(args.config).resolve()),
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "split": split,
                    "max_images": int(args.max_images),
                    "image_ids": image_ids,
                    "metrics": metrics,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
