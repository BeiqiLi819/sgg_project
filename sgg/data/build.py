from __future__ import annotations

import inspect
from typing import Dict, Iterable, Optional

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from .collate import sgg_collate_fn
from .datasets import DATASETS
from .grouped_batch_sampler import GroupedBatchSampler
from .transforms import (
    Compose,
    NormalizeTransform,
    RandomDirectionalFlip,
    RandomOBBRotate,
    ResizeTransform,
    ShortEdgeResizeTransform,
)


def _sync_model_cfg_from_metadata(cfg: Dict, metadata) -> Dict:
    cfg["MODEL"]["NUM_CLASSES"] = metadata.num_classes
    cfg["MODEL"]["NUM_PREDICATES"] = metadata.num_predicates

    if "ROI_BOX_HEAD" in cfg["MODEL"]:
        cfg["MODEL"]["ROI_BOX_HEAD"]["NUM_CLASSES"] = metadata.num_classes
    if "RELATION_HEAD" in cfg["MODEL"]:
        cfg["MODEL"]["RELATION_HEAD"]["NUM_PREDICATES"] = metadata.num_predicates
    return cfg


def _validate_dataset_kwargs(cls: type, kwargs: Dict, dataset_name: str) -> None:
    signature = inspect.signature(cls.__init__)
    missing = []
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        value = kwargs.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(name)

    if missing:
        missing_args = ", ".join(sorted(missing))
        raise ValueError(
            f"Dataset '{dataset_name}' is missing required config values for: {missing_args}"
        )


def _build_dataset_kwargs(cfg: Dict, split: str) -> tuple[type, Dict]:
    dcfg = cfg["DATASETS"][split.upper()]
    dataset_name = dcfg["NAME"]
    if dataset_name not in DATASETS.keys():
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. Available datasets: {sorted(DATASETS.keys())}"
        )
    cls = DATASETS.get(dataset_name)
    signature = inspect.signature(cls.__init__)
    valid_names = {
        name
        for name, param in signature.parameters.items()
        if name != "self" and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    transform_keys = {
        "IMAGE_SIZE",
        "KEEP_RATIO",
        "RESIZE_MODE",
        "MIN_SIZE",
        "MAX_SIZE",
        "PIXEL_MEAN",
        "PIXEL_STD",
        "TO_RGB",
    }
    kwargs = {
        k.lower(): v
        for k, v in dcfg.items()
        if k != "NAME" and k not in transform_keys and k.lower() in valid_names
    }
    if "split" in signature.parameters:
        kwargs.setdefault("split", split.lower())
    kwargs.setdefault("box_mode", cfg["MODEL"]["BOX_MODE"])
    kwargs.setdefault("task", cfg["MODEL"]["TASK"])
    if "box_angle_unit" in valid_names:
        kwargs.setdefault("box_angle_unit", cfg.get("MODEL", {}).get("OBB_ANGLE_UNIT", "degree"))
    transforms = _build_transforms(dcfg)
    if transforms is not None and "transforms" in valid_names:
        kwargs["transforms"] = transforms
    _validate_dataset_kwargs(cls, kwargs, dataset_name)
    return cls, kwargs


def _build_transforms(dcfg: Dict):
    transforms = []
    resize_mode = str(dcfg.get("RESIZE_MODE", "fit")).strip().lower()
    image_size = dcfg.get("IMAGE_SIZE")
    if resize_mode == "short_edge":
        min_size = dcfg.get("MIN_SIZE")
        max_size = dcfg.get("MAX_SIZE")
        if min_size is None or max_size is None:
            raise ValueError("short_edge resize requires DATASETS.*.MIN_SIZE and MAX_SIZE")
        transforms.append(ShortEdgeResizeTransform(min_size=min_size, max_size=max_size))
    elif resize_mode == "fit" and image_size:
        transforms.append(
            ResizeTransform(
                tuple(int(v) for v in image_size),
                keep_ratio=bool(dcfg.get("KEEP_RATIO", False)),
            )
        )
    elif resize_mode not in {"fit", "none"}:
        raise ValueError(f"Unsupported resize mode: {resize_mode}")
    if dcfg.get("AUGMENT", False):
        transforms.append(
            RandomDirectionalFlip(
                probs=(0.25, 0.25, 0.25),
                directions=("horizontal", "vertical", "diagonal"),
                angle_version=dcfg.get("ANGLE_VERSION", cfg_angle_version(dcfg)),
            )
        )
        transforms.append(
            RandomOBBRotate(
                rotate_ratio=float(dcfg.get("ROTATE_RATIO", 0.5)),
                angles_range=float(dcfg.get("ROTATE_ANGLES_RANGE", 180.0)),
                angle_version=dcfg.get("ANGLE_VERSION", cfg_angle_version(dcfg)),
            )
        )
    mean = dcfg.get("PIXEL_MEAN")
    std = dcfg.get("PIXEL_STD")
    if mean is not None and std is not None:
        transforms.append(NormalizeTransform(mean=mean, std=std, to_rgb=dcfg.get("TO_RGB", True)))
    if not transforms:
        return None
    return Compose(transforms)


def cfg_angle_version(dcfg: Dict) -> str:
    return str(dcfg.get("ANGLE_VERSION", "le90"))


def _split_is_configured(cfg: Dict, split: str) -> bool:
    dcfg = cfg.get("DATASETS", {}).get(split.upper())
    if not dcfg or not dcfg.get("NAME"):
        return False
    required_keys = ("ANN_FILE", "IMAGE_ROOT", "ROIDB_FILE", "DICT_FILE", "IMAGE_FILE")
    return any(bool(dcfg.get(key)) for key in required_keys)


def _resolve_splits(cfg: Dict, splits: Optional[Iterable[str]] = None) -> list[str]:
    if splits is not None:
        return [split.lower() for split in splits]

    ordered = []
    for split in ("train", "val", "test"):
        if _split_is_configured(cfg, split):
            ordered.append(split)
    return ordered


def sync_model_cfg_from_dataset(cfg: Dict, split: str = "train") -> Dict:
    cls, kwargs = _build_dataset_kwargs(cfg, split)
    dataset = cls(**kwargs)
    return _sync_model_cfg_from_metadata(cfg, dataset.metadata)


def build_dataset(cfg: Dict, split: str):
    cls, kwargs = _build_dataset_kwargs(cfg, split)
    dataset = cls(**kwargs)
    _sync_model_cfg_from_metadata(cfg, dataset.metadata)
    return dataset


def build_datasets(cfg: Dict, splits: Optional[Iterable[str]] = None) -> Dict[str, object]:
    resolved_splits = _resolve_splits(cfg, splits)
    datasets = {}
    for split in resolved_splits:
        datasets[split] = build_dataset(cfg, split)
    return datasets


def _build_dataloader_from_dataset(cfg: Dict, dataset, split: str, shuffle: bool = False) -> DataLoader:
    lcfg = cfg["DATALOADER"]
    split_key = f"{split.upper()}_BATCH_SIZE"
    batch_size = int(lcfg.get(split_key, lcfg["BATCH_SIZE"]))
    # Functional aliases for the SGG-Toolkit task scripts.  New configs may
    # still use DATALOADER.* directly; these only take effect when explicitly
    # supplied.
    if split.lower() == "train" and "IMS_PER_BATCH" in cfg.get("SOLVER", {}):
        batch_size = int(cfg["SOLVER"]["IMS_PER_BATCH"])
    elif split.lower() in {"val", "test"} and "IMS_PER_BATCH" in cfg.get("TEST", {}):
        batch_size = int(cfg["TEST"]["IMS_PER_BATCH"])
    collate_fn = lambda batch: sgg_collate_fn(
        batch,
        size_divisible=lcfg.get("SIZE_DIVISIBLE", 0),
    )
    use_aspect_grouping = (
        split.lower() == "train"
        and bool(lcfg.get("ASPECT_RATIO_GROUPING", False))
        and batch_size > 1
    )
    if use_aspect_grouping:
        records = getattr(dataset, "records", None)
        if records is None:
            raise TypeError("ASPECT_RATIO_GROUPING requires a dataset with image-size records")
        group_ids = [int(int(record["width"]) >= int(record["height"])) for record in records]
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        batch_sampler = GroupedBatchSampler(
            sampler=sampler,
            group_ids=group_ids,
            batch_size=batch_size,
            drop_last=False,
        )
        dcfg = cfg["DATASETS"][split.upper()]
        print(
            f"DataLoader[{split}]: batch_size={batch_size}, "
            f"aspect_ratio_grouping=True, resize_mode={dcfg.get('RESIZE_MODE', 'fit')}, "
            f"min_size={dcfg.get('MIN_SIZE')}, max_size={dcfg.get('MAX_SIZE')}",
            flush=True,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=lcfg["NUM_WORKERS"],
            collate_fn=collate_fn,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=lcfg["NUM_WORKERS"],
        collate_fn=collate_fn,
    )


def build_dataloader(cfg: Dict, split: str, shuffle: bool = False):
    dataset = build_dataset(cfg, split)
    return _build_dataloader_from_dataset(cfg, dataset, split=split, shuffle=shuffle)


def build_dataloaders(
    cfg: Dict,
    splits: Optional[Iterable[str]] = None,
    shuffle_map: Optional[Dict[str, bool]] = None,
    datasets: Optional[Dict[str, object]] = None,
) -> Dict[str, DataLoader]:
    resolved_splits = _resolve_splits(cfg, splits)
    datasets = datasets or build_datasets(cfg, resolved_splits)
    shuffle_map = shuffle_map or {}
    return {
        split: _build_dataloader_from_dataset(cfg, datasets[split], split=split, shuffle=shuffle_map.get(split, False))
        for split in resolved_splits
    }
