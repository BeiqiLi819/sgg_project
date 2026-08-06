from __future__ import annotations

import argparse
import importlib.util
import os
import random
from pathlib import Path

import numpy as np
import torch

from sgg.config.defaults import get_default_cfg
from sgg.data.build import build_dataloaders, build_datasets
from sgg.data.statistics import predicate_histogram
from sgg.engine import Trainer
from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector
from sgg.utils.reproducibility import save_run_manifest


def apply_runtime_cfg(cfg):
    runtime_cfg = cfg.get("RUNTIME", {})
    seed = runtime_cfg.get("SEED")
    if seed is not None:
        seed = int(seed)
        # PYTHONHASHSEED only affects a newly started interpreter.  The public
        # 6850 launcher exports it before Python starts; retaining it here
        # records the intended value for direct ``python train.py`` launches.
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    disable_cudnn = bool(runtime_cfg.get("DISABLE_CUDNN", True))
    torch.backends.cudnn.enabled = not disable_cudnn
    torch.backends.cudnn.benchmark = bool(runtime_cfg.get("CUDNN_BENCHMARK", False)) and not disable_cudnn
    torch.backends.cudnn.deterministic = bool(runtime_cfg.get("CUDNN_DETERMINISTIC", True))
    if bool(runtime_cfg.get("STRICT_REPRODUCIBILITY", False)):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if seed is not None:
        print(
            "Runtime reproducibility:",
            {
                "seed": seed,
                "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                "cudnn_enabled": torch.backends.cudnn.enabled,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "deterministic_algorithms": bool(
                    runtime_cfg.get("STRICT_REPRODUCIBILITY", False)
                ),
            },
            flush=True,
        )


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--filter-method",
        type=lambda value: value.upper(),
        choices=("PPG", "PPN", "RSGP"),
        default=None,
        help="Override MODEL.ROI_RELATION_HEAD.TEST_FILTER_METHOD.",
    )
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--init-rpcm", type=str, default="",
        help="Initialize the RPCM relation model from a compatible checkpoint.",
    )
    parser.add_argument("--start-epoch", type=int, default=None, help="Epoch index to resume from, 0-based. Defaults to checkpoint epoch.")
    parser.add_argument("--resume-step", type=int, default=None, help="Override global training step after loading a checkpoint.")
    parser.add_argument(
        "--reset-solver-on-resume",
        action="store_true",
        help="Load model weights from --resume but rebuild optimizer/scheduler from the current config.",
    )
    parser.add_argument(
        "--reset-scheduler-on-resume",
        action="store_true",
        help="Preserve resumed optimizer state but rebuild the LR scheduler from the current config.",
    )
    args = parser.parse_args()
    if args.reset_solver_on_resume and args.reset_scheduler_on_resume:
        raise ValueError(
            "--reset-solver-on-resume and --reset-scheduler-on-resume "
            "are mutually exclusive"
        )

    cfg = get_default_cfg()
    if args.config:
        user_cfg = load_py_config(args.config)
        cfg = user_cfg
    if args.filter_method is not None:
        filter_method = args.filter_method.upper()
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        rel_cfg["TEST_FILTER_METHOD"] = filter_method
        # These fields are retained only for old checkpoints/config readers.
        # Runtime construction uses TEST_FILTER_METHOD as the source of truth.
        rel_cfg["PPG_ENABLED"] = filter_method == "PPG"
        rel_cfg["PPN_ENABLED"] = filter_method == "PPN"
        rel_cfg["RSGP_ENABLED"] = filter_method == "RSGP"
    print(
        "Resolved training pair filter:",
        cfg["MODEL"]["ROI_RELATION_HEAD"]["TEST_FILTER_METHOD"],
        flush=True,
    )
    apply_runtime_cfg(cfg)
    datasets = build_datasets(cfg)
    if "train" in datasets:
        train_meta = datasets["train"].metadata
        cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"] = [
            train_meta.categories[i] for i in sorted(train_meta.categories.keys())
        ]
        cfg["MODEL"]["ROI_RELATION_HEAD"]["RELATION_NAMES"] = [
            train_meta.predicates[i] for i in sorted(train_meta.predicates.keys())
        ]
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        if not rel_cfg.get("PREDICATE_COUNTS"):
            rel_cfg["PREDICATE_COUNTS"] = predicate_histogram(datasets["train"])
    # Rewrite the manifest after dataset metadata and predicate priors have
    # been resolved so the archived config is sufficient to audit the run.
    save_run_manifest(
        cfg,
        cfg["SOLVER"].get("OUTPUT_DIR", "outputs/default"),
        config_path=args.config,
    )
    model = SceneGraphDetector(cfg)
    # print(model)
    dataloaders = build_dataloaders(
        cfg,
        datasets=datasets,
        shuffle_map={"train": True, "val": False, "test": False},
    )
    trainer = Trainer(cfg, model, device=args.device, dataloaders=dataloaders)
    start_epoch = 0
    init_modes = sum(bool(value) for value in (args.init_rpcm, args.resume))
    if init_modes > 1:
        raise ValueError("--init-rpcm and --resume are mutually exclusive")
    if args.init_rpcm:
        trainer.load_rpcm_predictor_weights(args.init_rpcm)
    if args.resume:
        ckpt = trainer.load_checkpoint(args.resume)
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"])
        if args.reset_solver_on_resume:
            solver_step = int(args.resume_step if args.resume_step is not None else trainer.global_step)
            trainer.reset_solver_state(global_step=solver_step)
        elif args.reset_scheduler_on_resume:
            scheduler_step = int(
                args.resume_step
                if args.resume_step is not None
                else trainer.global_step
            )
            trainer.reset_scheduler_state(global_step=scheduler_step)
    if args.start_epoch is not None:
        start_epoch = int(args.start_epoch)
    if args.resume_step is not None:
        trainer.global_step = int(args.resume_step)
    if args.resume or args.start_epoch is not None or args.resume_step is not None:
        print(
            f"Resume state: checkpoint={args.resume or '<none>'}, start_epoch={start_epoch}, global_step={trainer.global_step}",
            flush=True,
        )
    trainer.train(start_epoch=start_epoch)


if __name__ == "__main__":
    main()
