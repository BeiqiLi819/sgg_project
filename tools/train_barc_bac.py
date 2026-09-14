#!/usr/bin/env python3
"""Offline HPRC post-training on frozen PPG Top-10k Dual features.

The hard-only config reproduces the proposal-consistent Tail Aux control.  A
config with ``HPRC.UNIVERSAL.ENABLED`` additionally trains an all-class,
sample-conditioned residual on exactly the same balanced batches.
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
from tqdm import trange


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgg.engine import Trainer  # noqa: E402
from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector  # noqa: E402
from sgg.modeling.detectors.class_channel_order import (  # noqa: E402
    INTERNAL_DETECTOR_CLASS_ORDER,
)
from tools.eval_once import apply_runtime_cfg, load_py_config  # noqa: E402
from tools.export_barc_bac_cache import CACHE_SCHEMA  # noqa: E402


CHECKPOINT_SCHEMA = "ppg-consistent-tail-aux-compact-v1"


def _adapter_state_dict(predictor) -> dict[str, torch.Tensor]:
    prefixes = ("tail_aux_", "universal_aux_", "rel_proto.")
    return {
        key: value
        for key, value in predictor.state_dict().items()
        if key.startswith(prefixes)
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TailFeatureCache:
    def __init__(self, root: Path, manifest: dict, *, seed: int):
        feature_parts = []
        logit_parts = []
        label_parts = []
        for row in manifest["shards"]:
            path = root / str(row["path"])
            if _sha256(path) != str(row["sha256"]):
                raise RuntimeError(f"Tail cache shard hash mismatch: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            labels = payload["labels"].long()
            if labels.shape != (int(row["rows"]),):
                raise RuntimeError(f"Tail cache row mismatch: {path}")
            feature_parts.append(payload["relation_features"])
            logit_parts.append(payload["original_logits"].float())
            label_parts.append(labels)
        self.features = torch.cat(feature_parts, dim=0)
        self.base_logits = torch.cat(logit_parts, dim=0)
        self.labels = torch.cat(label_parts, dim=0)
        if self.labels.eq(0).any():
            raise ValueError("Preservation rows must use label=-1, not background=0")
        self.class_rows = {
            int(class_id): torch.nonzero(self.labels.eq(class_id), as_tuple=False).flatten()
            for class_id in self.labels[self.labels > 0].unique(sorted=True).tolist()
        }
        self.classes = torch.as_tensor(sorted(self.class_rows), dtype=torch.long)
        self.preserve_rows = torch.nonzero(self.labels < 0, as_tuple=False).flatten()
        if self.classes.numel() < 2 or self.preserve_rows.numel() == 0:
            raise RuntimeError("Tail cache requires foreground classes and preservation rows")
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

    def sample(
        self,
        batch_size: int,
        preserve_fraction: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        preserve_count = min(
            max(int(round(batch_size * preserve_fraction)), 0), batch_size - 1
        )
        foreground_count = batch_size - preserve_count
        class_slots = torch.randint(
            self.classes.numel(), (foreground_count,), generator=self.generator
        )
        foreground_rows = []
        for class_id in self.classes[class_slots].tolist():
            candidates = self.class_rows[int(class_id)]
            position = torch.randint(
                candidates.numel(), (1,), generator=self.generator
            ).item()
            foreground_rows.append(int(candidates[position]))
        rows = torch.as_tensor(foreground_rows, dtype=torch.long)
        if preserve_count:
            preserve_positions = torch.randint(
                self.preserve_rows.numel(),
                (preserve_count,),
                generator=self.generator,
            )
            rows = torch.cat(
                [rows, self.preserve_rows.index_select(0, preserve_positions)]
            )
        return (
            self.features.index_select(0, rows).float().to(device),
            self.base_logits.index_select(0, rows).to(device),
            self.labels.index_select(0, rows).to(device),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/star_predcls_dual17500_tail_aux_protected_train.py",
    )
    parser.add_argument(
        "--base-checkpoint",
        default=(
            "outputs/role_gnn_historical_matched/seed1029/dual_role_aware/"
            "model_iter_0017500.pth"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default="outputs/sparse_competition_dual17500_preserve/cache",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--preserve-fraction", type=float, default=0.25)
    parser.add_argument("--tail-lr", type=float, default=1e-3)
    parser.add_argument("--prototype-lr", type=float, default=1e-5)
    parser.add_argument("--preserve-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--save-period", type=int, default=500)
    parser.add_argument("--log-period", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1029)
    parser.add_argument("--save-full-model", action="store_true")
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("PPG-consistent Tail Aux requires CUDA")
    if args.iterations <= 0 or args.batch_size <= 0:
        raise ValueError("iterations and batch size must be positive")
    if not 0.0 < args.preserve_fraction < 1.0:
        raise ValueError("preserve_fraction must be in (0,1)")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Tail Aux output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cache_root = Path(args.cache_dir)
    cache_manifest_path = cache_root / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text())
    if cache_manifest.get("schema") != CACHE_SCHEMA:
        raise ValueError("PPG-consistent Tail Aux requires v2 preservation cache")
    if cache_manifest.get("validation_or_test_labels_used") is not False:
        raise ValueError("Tail cache is not TRAIN-only")
    base_path = Path(args.base_checkpoint)
    if _sha256(base_path) != str(cache_manifest["base_checkpoint_sha256"]):
        raise RuntimeError("Tail cache/base checkpoint SHA mismatch")

    cfg = load_py_config(args.config)
    cfg["MODEL"]["ROI_RELATION_HEAD"]["PREDICATE_COUNTS"] = list(
        cache_manifest["class_counts"]
    )
    apply_runtime_cfg(cfg)
    model = SceneGraphDetector(cfg)
    trainer = Trainer(cfg, model, device=args.device, dataloaders={})
    trainer.load_model_checkpoint_weights(str(base_path))
    predictor = trainer.model.roi_heads.relation.predictor
    for parameter in trainer.model.parameters():
        parameter.requires_grad_(False)
    tail_parameters = list(predictor.tail_aux_head.parameters()) + [
        predictor.tail_aux_logit_scale
    ]
    universal_parameters: list[torch.nn.Parameter] = []
    if predictor.hprc_universal_enabled:
        universal_parameters = list(predictor.universal_aux_head.parameters()) + [
            predictor.universal_aux_logit_scale
        ]
        if not predictor.hprc_universal_zero_init_output:
            raise ValueError(
                "PPG-consistent all-class adaptation requires "
                "HPRC.UNIVERSAL.ZERO_INIT_OUTPUT=True"
            )
        if predictor.hprc_universal_gate_parameterization != "positive_sigmoid":
            raise ValueError(
                "PPG-consistent all-class adaptation requires a positive gate"
            )
    proto_parameters = list(predictor.rel_proto.parameters())
    adapter_parameters = tail_parameters + universal_parameters
    for parameter in adapter_parameters + proto_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": float(args.tail_lr)},
            {"params": proto_parameters, "lr": float(args.prototype_lr)},
        ],
        weight_decay=1e-4,
    )
    proto_anchor = {
        name: parameter.detach().clone()
        for name, parameter in predictor.rel_proto.named_parameters()
    }
    cache = TailFeatureCache(cache_root, cache_manifest, seed=args.seed)
    counts = torch.as_tensor(cache_manifest["class_counts"], dtype=torch.float32)
    prior = counts.clamp_min(1.0) / counts.clamp_min(1.0).sum()
    prior = prior.to(trainer.device)

    # Quantized cache features must still reconstruct the frozen classifier.
    predictor.eval()
    identity_features, identity_base, identity_labels = cache.sample(
        min(args.batch_size, 64), args.preserve_fraction, trainer.device
    )
    with torch.no_grad():
        identity_logits, _ = predictor.rel_proto(identity_features, None)
        identity_corrected, _ = predictor._apply_hprc_adapters(
            identity_logits, identity_features, None
        )
    identity_error = float((identity_corrected - identity_base).abs().max())
    predictor.train()

    log_path = output / "training_log.jsonl"
    running: dict[str, float] = {}
    for iteration in trange(1, args.iterations + 1, desc="PPG-consistent Tail Aux"):
        features, base_logits, labels = cache.sample(
            args.batch_size, args.preserve_fraction, trainer.device
        )
        foreground = labels > 0
        preserve = labels < 0
        logits, proto_losses = predictor.rel_proto(features, labels)
        corrected, _ = predictor._apply_hprc_adapters(logits, features, None)
        ce = F.cross_entropy(corrected[foreground], labels[foreground])
        la = F.cross_entropy(
            corrected[foreground] + 0.5 * prior.clamp_min(1e-12).log(),
            labels[foreground],
        )
        tail_logits = predictor.tail_aux_head(features[foreground])
        tail_targets = predictor._tail_aux_targets(labels[foreground])
        tail_bce = F.binary_cross_entropy_with_logits(
            tail_logits,
            tail_targets,
            pos_weight=predictor.tail_aux_pos_weight.to(trainer.device),
        )
        preserve_kl = F.kl_div(
            F.log_softmax(corrected[preserve], dim=1),
            F.softmax(base_logits[preserve], dim=1),
            reduction="batchmean",
        )
        anchor = sum(
            (parameter - proto_anchor[name]).square().mean()
            for name, parameter in predictor.rel_proto.named_parameters()
        )
        proto = sum(proto_losses.values()) if proto_losses else corrected.sum() * 0.0
        total = (
            ce
            + 0.1 * la
            + 0.2 * tail_bce
            + float(args.preserve_weight) * preserve_kl
            + float(args.anchor_weight) * anchor
            + proto
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite Tail Aux loss at {iteration}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            adapter_parameters + proto_parameters, float(args.grad_clip)
        )
        optimizer.step()
        diagnostics = predictor.hprc_adapter_diagnostics()
        row = {
            "iteration": iteration,
            "loss_total": float(total.detach()),
            "loss_ce": float(ce.detach()),
            "loss_la": float(la.detach()),
            "loss_tail_bce": float(tail_bce.detach()),
            "loss_preserve": float(preserve_kl.detach()),
            "loss_anchor": float(anchor.detach()),
            "loss_proto": float(proto.detach()),
            "gradient_norm": float(grad_norm.detach()),
            "effective_gate": diagnostics["hard_effective_scale"],
            "universal_enabled": bool(predictor.hprc_universal_enabled),
        }
        if predictor.hprc_universal_enabled:
            row.update(
                {
                    "universal_gate_abs_mean": diagnostics[
                        "universal_effective_abs_mean"
                    ],
                    "universal_gate_abs_max": diagnostics[
                        "universal_effective_abs_max"
                    ],
                    "universal_active_classes": diagnostics[
                        "universal_active_classes_1e-6"
                    ],
                }
            )
        with log_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        for key, value in row.items():
            if key != "iteration":
                running[key] = running.get(key, 0.0) + float(value)
        if iteration % args.log_period == 0:
            print(
                {
                    "iteration": iteration,
                    **{key: value / args.log_period for key, value in running.items()},
                },
                flush=True,
            )
            running = {}
        if args.save_period > 0 and iteration % args.save_period == 0:
            torch.save(
                {
                    "schema": CHECKPOINT_SCHEMA,
                    "iteration": iteration,
                    "predictor": _adapter_state_dict(predictor),
                    "cache_manifest_sha256": _sha256(cache_manifest_path),
                    "base_checkpoint_sha256": _sha256(base_path),
                    "identity_max_abs_error": identity_error,
                    "training": vars(args),
                },
                output / f"tail_aux_iter_{iteration:05d}.pth",
            )

    compact_path = output / "tail_aux_final_compact.pth"
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "iteration": args.iterations,
            "predictor": _adapter_state_dict(predictor),
            "cache_manifest_sha256": _sha256(cache_manifest_path),
            "base_checkpoint_sha256": _sha256(base_path),
            "identity_max_abs_error": identity_error,
            "training": vars(args),
        },
        compact_path,
    )
    full_path = None
    if args.save_full_model:
        full_path = output / "model_ppg_consistent_final.pth"
        torch.save(
            {
                "model": trainer.model.state_dict(),
                "detector_class_channel_order": INTERNAL_DETECTOR_CLASS_ORDER,
                "ppg_consistent_tail_aux": {
                    "schema": CHECKPOINT_SCHEMA,
                    "iteration": args.iterations,
                    "base_checkpoint_sha256": _sha256(base_path),
                    "cache_manifest_sha256": _sha256(cache_manifest_path),
                    "identity_max_abs_error": identity_error,
                },
            },
            full_path,
        )
    manifest = {
        "schema": "ppg-consistent-tail-aux-run-v1",
        "base_checkpoint": str(base_path.resolve()),
        "base_checkpoint_sha256": _sha256(base_path),
        "cache_manifest": str(cache_manifest_path.resolve()),
        "cache_manifest_sha256": _sha256(cache_manifest_path),
        "proposal": str(cache_manifest.get("proposal", "released_ppg_top10000")),
        "proposal_contract": cache_manifest.get("proposal_contract"),
        "random_relation_sampling_used": False,
        "unannotated_pairs_used_as_negatives": False,
        "variant": "HU" if predictor.hprc_universal_enabled else "H",
        "all_class_balanced_supervision": bool(predictor.hprc_universal_enabled),
        "identity_max_abs_error": identity_error,
        "compact_checkpoint": compact_path.name,
        "full_model_checkpoint": None if full_path is None else full_path.name,
        "iterations": args.iterations,
        "seed": args.seed,
        "test_untouched": True,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest, flush=True)


if __name__ == "__main__":
    main()
