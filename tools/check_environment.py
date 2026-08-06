#!/usr/bin/env python3
"""Validate the minimal STAR SGG runtime and detect cross-project leakage."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from importlib import metadata
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED = {
    "torch": ("torch", "2.2.2"),
    "torchvision": ("torchvision", "0.17.2"),
    "numpy": ("numpy", "1.26.4"),
    "opencv-python": ("cv2", "4.11.0.86"),
    "Pillow": ("PIL", "11.0.0"),
    "h5py": ("h5py", "3.14.0"),
    "tqdm": ("tqdm", "4.65.2"),
    "mmcv-full": ("mmcv", "1.7.2"),
    # Runtime dependencies imported by mmcv-full itself. They are pinned in
    # requirements.clean.txt and checked here so a partial MMCV installation
    # cannot pass the public Full-route environment audit.
    "addict": ("addict", "2.4.0"),
    "packaging": ("packaging", "24.2"),
    "PyYAML": ("yaml", "6.0.2"),
    "yapf": ("yapf", "0.43.0"),
}

FULL_ROUTE_MODULES = (
    "configs.star_predcls_obb_full",
    "configs.star_sgcls_obb_full",
    "configs.star_sgdet_obb_full",
    "sgg.data.build",
    "sgg.engine.trainer",
    "sgg.evaluation.sgg_eval",
    "sgg.modeling.detectors.scene_graph_detector",
)

UNWANTED = (
    "maskrcnn_benchmark",
    "mmrotate",
    "mmdet",
    "torch_geometric",
    "torch_scatter",
    "torch_sparse",
)


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _module_origin(name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    return None if spec is None else str(spec.origin)


def inspect_environment(require_cuda: bool = False) -> tuple[dict, list[str]]:
    failures: list[str] = []
    packages = {}
    for distribution_name, (module_name, expected_version) in REQUIRED.items():
        version = _distribution_version(distribution_name)
        origin = _module_origin(module_name)
        packages[distribution_name] = {
            "expected": expected_version,
            "installed": version,
            "origin": origin,
        }
        if version is None or origin is None:
            failures.append(f"missing required package: {distribution_name}")
        elif version.split("+")[0] != expected_version:
            failures.append(
                f"version mismatch for {distribution_name}: expected {expected_version}, got {version}"
            )

    contaminated_paths = [
        entry
        for entry in sys.path
        if "/RPCM" in entry or "/SGG-ToolKit" in entry
    ]
    if contaminated_paths:
        failures.append(f"legacy project paths found in sys.path: {contaminated_paths}")

    unwanted = {name: _module_origin(name) for name in UNWANTED}
    present_unwanted = {name: origin for name, origin in unwanted.items() if origin is not None}
    if present_unwanted:
        failures.append(f"unused legacy packages are importable: {present_unwanted}")

    ops = {
        "imported": False,
        "cpu_smoke": False,
        "cuda_available": False,
        "cuda_smoke": False,
        "roi_align_rotated_cpu_smoke": False,
        "roi_align_rotated_cuda_smoke": False,
        "torch_cuda": None,
        "compiled_cuda": None,
        "compiler": None,
    }
    try:
        import torch
        from mmcv.ops import (
            RoIAlignRotated,
            box_iou_rotated,
            get_compiler_version,
            get_compiling_cuda_version,
            nms_rotated,
        )

        ops["imported"] = True
        ops["cuda_available"] = torch.cuda.is_available()
        ops["torch_cuda"] = torch.version.cuda
        ops["compiled_cuda"] = get_compiling_cuda_version()
        ops["compiler"] = get_compiler_version()
        boxes = torch.tensor([[8.0, 8.0, 4.0, 2.0, 0.0]], dtype=torch.float32)
        scores = torch.tensor([0.9], dtype=torch.float32)
        overlap = box_iou_rotated(boxes, boxes)
        kept, indices = nms_rotated(boxes, scores, 0.5)
        ops["cpu_smoke"] = bool(overlap.shape == (1, 1) and kept.shape[0] == 1 and indices.tolist() == [0])
        if not ops["cpu_smoke"]:
            failures.append("mmcv rotated-op CPU smoke test returned unexpected output")
        roi_features = torch.arange(
            64, dtype=torch.float32
        ).reshape(1, 1, 8, 8)
        rois = torch.tensor(
            [[0.0, 4.0, 4.0, 4.0, 4.0, 0.0]], dtype=torch.float32
        )
        roi_align = RoIAlignRotated(
            (2, 2), 1.0, 1, aligned=True, clockwise=False
        )
        roi_output = roi_align(roi_features, rois)
        ops["roi_align_rotated_cpu_smoke"] = bool(
            roi_output.shape == (1, 1, 2, 2)
            and torch.isfinite(roi_output).all().item()
        )
        if not ops["roi_align_rotated_cpu_smoke"]:
            failures.append(
                "mmcv RoIAlignRotated CPU smoke test returned unexpected output"
            )
        if ops["cuda_available"]:
            cuda_boxes = boxes.cuda()
            cuda_scores = scores.cuda()
            cuda_overlap = box_iou_rotated(cuda_boxes, cuda_boxes)
            cuda_kept, cuda_indices = nms_rotated(cuda_boxes, cuda_scores, 0.5)
            ops["cuda_smoke"] = bool(
                cuda_overlap.shape == (1, 1)
                and cuda_kept.shape[0] == 1
                and cuda_indices.cpu().tolist() == [0]
            )
            if not ops["cuda_smoke"]:
                failures.append("mmcv rotated-op CUDA smoke test returned unexpected output")
            cuda_roi_output = roi_align(roi_features.cuda(), rois.cuda())
            ops["roi_align_rotated_cuda_smoke"] = bool(
                cuda_roi_output.shape == (1, 1, 2, 2)
                and torch.isfinite(cuda_roi_output).all().item()
            )
            if not ops["roi_align_rotated_cuda_smoke"]:
                failures.append(
                    "mmcv RoIAlignRotated CUDA smoke test returned unexpected output"
                )
        elif require_cuda:
            failures.append("CUDA is required but torch.cuda.is_available() is false")
    except Exception as exc:  # pragma: no cover - error is reported to the caller
        failures.append(f"mmcv-full rotated ops failed: {type(exc).__name__}: {exc}")

    before = set(sys.modules)
    try:
        module = importlib.import_module("sgg.modeling.core.obb_ops")
        boxes = module.torch.tensor([[8.0, 8.0, 4.0, 2.0, 0.0]])
        module.obb2poly(boxes, version="le90", angle_unit="radian")
    except Exception as exc:  # pragma: no cover - error is reported to the caller
        failures.append(f"local OBB helpers failed: {type(exc).__name__}: {exc}")
    newly_imported_legacy = sorted(
        name for name in set(sys.modules) - before if name.startswith(("mmrotate", "mmdet", "maskrcnn_benchmark"))
    )
    if newly_imported_legacy:
        failures.append(f"local OBB helpers imported legacy modules: {newly_imported_legacy}")

    full_route_imports = {}
    for module_name in FULL_ROUTE_MODULES:
        try:
            module = importlib.import_module(module_name)
            full_route_imports[module_name] = str(
                getattr(module, "__file__", "<namespace>")
            )
        except Exception as exc:  # pragma: no cover - reported to caller
            full_route_imports[module_name] = None
            failures.append(
                "Full train/test route import failed for "
                f"{module_name}: {type(exc).__name__}: {exc}"
            )

    full_config_contract = {}
    for module_name, expected_task in (
        ("configs.star_predcls_obb_full", "predcls"),
        ("configs.star_sgcls_obb_full", "sgcls"),
        ("configs.star_sgdet_obb_full", "sgdet"),
    ):
        module = sys.modules.get(module_name)
        if module is None or not hasattr(module, "cfg"):
            continue
        cfg = module.cfg
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        contract = {
            "task": cfg["MODEL"]["TASK"],
            "predictor": rel_cfg["PREDICTOR"],
            "relation_graph": rel_cfg["RPCM_RELATION_GRAPH_MODE"],
            "train_validation_filter": rel_cfg["TEST_FILTER_METHOD"],
            "rsgp_role_mode": rel_cfg["RSGP_ROLE_MODE"],
        }
        full_config_contract[expected_task] = contract
        if contract["task"] != expected_task:
            failures.append(
                f"Full config task mismatch for {module_name}: {contract['task']}"
            )
        if contract["predictor"] != "RPCM_ORIGINAL_LEGACY":
            failures.append(
                f"Unexpected public Full predictor for {module_name}: "
                f"{contract['predictor']}"
            )
        if contract["relation_graph"] != "dual_view":
            failures.append(
                f"Unexpected public Full relation graph for {module_name}: "
                f"{contract['relation_graph']}"
            )

    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": packages,
        "mmcv_ops": ops,
        "legacy_paths": contaminated_paths,
        "unwanted_packages": present_unwanted,
        "full_route_imports": full_route_imports,
        "full_config_contract": full_config_contract,
        "failures": failures,
    }
    return report, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero for version drift or legacy packages.")
    parser.add_argument("--require-cuda", action="store_true", help="Also require and execute CUDA rotated operators.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    report, failures = inspect_environment(require_cuda=args.require_cuda)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
