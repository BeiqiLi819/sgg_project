#!/usr/bin/env python3
"""Convert the original RPCM 6850 checkpoint into a native Dual checkpoint.

The source file uses the SGG-Toolkit module names, background-last detector
classification, and an optimizer tied to those source tensors.  This tool
constructs the current Dual model, applies the audited legacy key mapping once,
and writes a complete current-project model-only checkpoint.  The output can
thereafter be loaded with the ordinary ``full`` checkpoint path; no legacy
runtime remapping is needed.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sgg.data.build import build_datasets  # noqa: E402
from sgg.data.statistics import predicate_histogram  # noqa: E402
from sgg.engine import Trainer  # noqa: E402
from sgg.modeling.detectors.class_channel_order import (  # noqa: E402
    INTERNAL_DETECTOR_CLASS_ORDER,
)
from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector  # noqa: E402


DEFAULT_SOURCE = "/home/ubuntu/research/ssd/RPCM/weights/6850_4135.pth"
DEFAULT_CONFIG = "configs/star_predcls_obb_ablation_dual_rca_train.py"
DEFAULT_OUTPUT = "pretrained/RPCM_6850_Dual_bgfirst.pth"
DEFAULT_SEED = 1029


def load_py_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location("rpcm_6850_conversion_cfg", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "cfg"):
        return module.cfg
    if hasattr(module, "get_cfg"):
        return module.get_cfg()
    raise AttributeError("Config file must expose cfg or get_cfg().")


def resolve_dataset_metadata(cfg: dict) -> None:
    datasets = build_datasets(cfg, splits=("train",))
    metadata = datasets["train"].metadata
    cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"] = [
        metadata.categories[index] for index in sorted(metadata.categories)
    ]
    rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
    rel_cfg["RELATION_NAMES"] = [
        metadata.predicates[index] for index in sorted(metadata.predicates)
    ]
    if not rel_cfg.get("PREDICATE_COUNTS"):
        rel_cfg["PREDICATE_COUNTS"] = predicate_histogram(datasets["train"])


def convert_checkpoint(
    source_path: str | Path,
    output_path: str | Path,
    config_path: str | Path,
    *,
    seed: int = DEFAULT_SEED,
) -> dict:
    source_path = Path(source_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if source_path == output_path:
        raise ValueError("--output must differ from --input; the source is never modified.")

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = load_py_config(config_path)
    rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
    if str(cfg["MODEL"].get("TASK", "")).lower() != "predcls":
        raise ValueError("Conversion config must define MODEL.TASK='predcls'.")
    if str(rel_cfg.get("RPCM_RELATION_GRAPH_MODE", "")).lower() != "dual_view":
        raise ValueError("Conversion config must select RPCM_RELATION_GRAPH_MODE='dual_view'.")
    if not bool(rel_cfg.get("RPCM_LEGACY_6850_EXACT", False)):
        raise ValueError("Conversion config must enable RPCM_LEGACY_6850_EXACT.")

    resolve_dataset_metadata(cfg)
    model = SceneGraphDetector(cfg)
    trainer = Trainer(cfg, model, device="cpu", dataloaders={})
    mapping_report = trainer.load_rpcm_predictor_weights(str(source_path))

    state_dict = trainer.model.state_dict()
    # This strict self-check makes the output invariant explicit: every target
    # key is present and shape-compatible before the multi-GB file is written.
    incompatible = trainer.model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Converted state failed strict validation: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    payload = {
        "epoch": 0,
        "global_step": 0,
        "model": state_dict,
        # Source optimizer momentum is indexed by the old module tensors and
        # must never be restored into the converted model.
        "optimizer": None,
        "scheduler": None,
        "cfg": cfg,
        "detector_class_channel_order": INTERNAL_DETECTOR_CLASS_ORDER,
        "checkpoint_type": "rpcm_6850_converted_dual_model_only",
        "conversion": {
            "source_checkpoint": str(source_path),
            "config": str(config_path),
            "graph_mode": "dual_view",
            "legacy_6850_exact": True,
            "source_optimizer_dropped": True,
            "current_only_initialization_seed": seed,
            "mapping_report": mapping_report,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    return {
        "input": str(source_path),
        "output": str(output_path),
        "config": str(config_path),
        "model_tensors": len(state_dict),
        "loaded_source_tensors": len(mapping_report["loaded"]),
        "loaded_predictor_tensors": len(mapping_report["loaded_predictor"]),
        "unloaded_target_tensors_initialized_by_current_model": len(
            mapping_report["unloaded_target"]
        ),
        "unused_source_tensors": len(mapping_report["unused_source"]),
        "shape_mismatches": len(mapping_report["skipped_shape"]),
        "detector_class_channel_order": INTERNAL_DETECTOR_CLASS_ORDER,
        "optimizer_scheduler_dropped": True,
        "current_only_initialization_seed": seed,
        "size_bytes": output_path.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RPCM weights/6850_4135.pth to a native current-project Dual checkpoint."
    )
    parser.add_argument("--input", default=DEFAULT_SOURCE, help="Original RPCM checkpoint.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Current Dual config.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Converted checkpoint path.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for the few current-only tensors absent from the source checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = convert_checkpoint(args.input, args.output, args.config, seed=args.seed)
    print("Converted RPCM 6850 checkpoint:", report, flush=True)


if __name__ == "__main__":
    main()
