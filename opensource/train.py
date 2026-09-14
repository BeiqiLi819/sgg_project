from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import random

import numpy as np
import torch

from sgg.config.defaults import get_default_cfg
from sgg.data.build import build_dataloaders, build_datasets
from sgg.data.statistics import predicate_histogram
from sgg.engine import Trainer
from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector


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


def seed_runtime(cfg) -> int | None:
    value = cfg.get("RUNTIME", {}).get("SEED")
    if value is None:
        return None
    seed = int(value)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # This is effective for child data-loader processes.  The matched Round-3
    # launcher also exports it before Python starts, covering hash iteration in
    # the main process.
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(
        "Deterministic runtime seed:",
        {"seed": seed, "pythonhashseed": os.environ.get("PYTHONHASHSEED")},
        flush=True,
    )
    return seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--init-rpcm", type=str, default="",
        help="Initialize the RPCM relation model from a compatible checkpoint.",
    )
    parser.add_argument(
        "--init-model-checkpoint",
        type=str,
        default="",
        help=(
            "Initialize every shape-compatible model tensor from a training "
            "checkpoint while keeping a fresh optimizer, scheduler and step. "
            "Newly introduced modules retain their configured initialization."
        ),
    )
    parser.add_argument(
        "--init-model-state",
        type=str,
        default="",
        help=(
            "Load an exact model-only initialization artifact created by "
            "--export-init-state; optimizer, scheduler and step remain fresh."
        ),
    )
    parser.add_argument(
        "--export-init-state",
        type=str,
        default="",
        help="Build the configured model, export a reusable exact initial state, and exit.",
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
        help=(
            "Load model and optimizer state from --resume, but rebuild only the "
            "scheduler from the current config. This preserves optimizer momentum "
            "when extending a training schedule."
        ),
    )
    args = parser.parse_args()

    if args.reset_solver_on_resume and args.reset_scheduler_on_resume:
        parser.error(
            "--reset-solver-on-resume and --reset-scheduler-on-resume are mutually exclusive"
        )

    cfg = get_default_cfg()
    if args.config:
        user_cfg = load_py_config(args.config)
        cfg = user_cfg
    apply_runtime_cfg(cfg)
    runtime_seed = seed_runtime(cfg)
    datasets = build_datasets(cfg)
    if "train" in datasets:
        train_meta = datasets["train"].metadata
        cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"] = [
            train_meta.categories[i]
            for i in sorted(train_meta.categories.keys(), key=int)
        ]
        cfg["MODEL"]["ROI_RELATION_HEAD"]["RELATION_NAMES"] = [
            train_meta.predicates[i]
            for i in sorted(train_meta.predicates.keys(), key=int)
        ]
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        if not rel_cfg.get("PREDICATE_COUNTS"):
            rel_cfg["PREDICATE_COUNTS"] = predicate_histogram(datasets["train"])
    model = SceneGraphDetector(cfg)
    # print(model)
    dataloaders = build_dataloaders(
        cfg,
        datasets=datasets,
        shuffle_map={"train": True, "val": False, "test": False},
    )
    trainer = Trainer(cfg, model, device=args.device, dataloaders=dataloaders)
    start_epoch = 0
    init_modes = sum(
        bool(value)
        for value in (
            args.init_rpcm,
            args.init_model_checkpoint,
            args.init_model_state,
            args.resume,
        )
    )
    if init_modes > 1:
        raise ValueError(
            "--init-rpcm, --init-model-checkpoint, --init-model-state and "
            "--resume are mutually exclusive"
        )
    if args.export_init_state and init_modes:
        raise ValueError(
            "--export-init-state cannot be combined with an initialization/resume input"
        )
    if args.export_init_state:
        trainer.save_model_initialization(
            args.export_init_state,
            seed=runtime_seed,
            source_config=args.config or "<default>",
        )
        print(f"Exported model initialization: {args.export_init_state}", flush=True)
        return
    if args.init_model_state:
        trainer.load_model_initialization(
            args.init_model_state,
            expected_seed=runtime_seed,
        )
    if args.init_model_checkpoint:
        trainer.load_model_checkpoint_weights(args.init_model_checkpoint)
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
            scheduler_step = int(args.resume_step if args.resume_step is not None else trainer.global_step)
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
