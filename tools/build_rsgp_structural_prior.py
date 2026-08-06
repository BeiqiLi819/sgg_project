from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sgg.data.build import build_dataset
from sgg.data.rsgp_statistics import (
    build_rsgp_structural_prior,
    write_rsgp_structural_prior,
)


def load_py_config(path: str):
    spec = importlib.util.spec_from_file_location("rsgp_prior_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if hasattr(mod, "cfg"):
        return mod.cfg
    if hasattr(mod, "get_cfg"):
        return mod.get_cfg()
    raise AttributeError("Config file must expose `cfg` or `get_cfg()`.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build category-name-agnostic RSGP structural roles and rarity "
            "support from the configured training split."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/star_predcls_obb_ablation_dual_la_train.py",
        help="Python config whose DATASETS.TRAIN entry defines the source split.",
    )
    parser.add_argument(
        "--output",
        default="pretrained/rsgp_structural_prior.json",
        help="Output JSON path.",
    )
    parser.add_argument("--rarity-beta", type=float, default=0.5)
    parser.add_argument("--containment-block-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_py_config(args.config)
    dataset = build_dataset(cfg, "train")
    payload = build_rsgp_structural_prior(
        dataset,
        split="train",
        rarity_beta=args.rarity_beta,
        containment_block_size=args.containment_block_size,
    )
    output_path = write_rsgp_structural_prior(payload, args.output)
    summary = {
        "output": str(output_path),
        "schema_version": payload["schema_version"],
        "source_split": payload["source_split"],
        "dataset_name": payload["dataset_name"],
        "num_images": payload["num_images"],
        "num_classes": payload["num_classes"],
        "num_predicates": payload["num_predicates"],
        "dataset_signature": payload["dataset_signature"],
        "config_hash": payload["config_hash"],
        "payload_hash": payload["payload_hash"],
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
