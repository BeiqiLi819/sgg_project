from __future__ import annotations

import importlib

import torch
from torch.utils.data import SequentialSampler

from sgg.data.grouped_batch_sampler import GroupedBatchSampler
from sgg.data.transforms import ShortEdgeResizeTransform
from sgg.structures.boxes import BoxList


def test_short_edge_resize_matches_600_1000_convention():
    transform = ShortEdgeResizeTransform(min_size=600, max_size=1000)

    square = torch.zeros((3, 1200, 1200), dtype=torch.float32)
    square_target = BoxList(
        torch.tensor([[600.0, 600.0, 200.0, 100.0, 0.0]]),
        (1200, 1200),
        mode="xywha",
    )
    square_image, square_resized = transform(square, square_target)
    assert square_image.shape == (3, 600, 600)
    assert square_resized.size == (600, 600)
    torch.testing.assert_close(
        square_resized.bbox,
        torch.tensor([[300.0, 300.0, 100.0, 50.0, 0.0]]),
    )

    wide = torch.zeros((3, 600, 1800), dtype=torch.float32)
    wide_target = BoxList(
        torch.tensor([[900.0, 300.0, 300.0, 120.0, 0.0]]),
        (1800, 600),
        mode="xywha",
    )
    wide_image, wide_resized = transform(wide, wide_target)
    assert wide_image.shape == (3, 333, 1000)
    assert wide_resized.size == (1000, 333)


def test_grouped_batch_sampler_never_mixes_aspect_groups():
    group_ids = [0, 1, 0, 1, 0, 1, 0]
    sampler = SequentialSampler(group_ids)
    batches = list(
        GroupedBatchSampler(
            sampler=sampler,
            group_ids=group_ids,
            batch_size=2,
            drop_last=False,
        )
    )

    assert len(batches) == 4
    assert sorted(index for batch in batches for index in batch) == list(range(7))
    assert all(len({group_ids[index] for index in batch}) == 1 for batch in batches)


def test_6850_resize_and_grouping_are_isolated_from_paper_rows(monkeypatch):
    monkeypatch.delenv("TRAIN_BATCH_SIZE", raising=False)
    monkeypatch.delenv("DISABLE_CUDNN", raising=False)
    repro_module = importlib.import_module(
        "configs.star_predcls_obb_ablation_dual_6850_repro_train"
    )
    repro_module = importlib.reload(repro_module)
    repro_cfg = repro_module.cfg
    assert repro_cfg["DATALOADER"]["ASPECT_RATIO_GROUPING"] is True
    assert repro_cfg["RUNTIME"]["DISABLE_CUDNN"] is True
    for split in ("TRAIN", "VAL", "TEST"):
        assert repro_cfg["DATASETS"][split]["RESIZE_MODE"] == "short_edge"
        assert repro_cfg["DATASETS"][split]["MIN_SIZE"] == 600
        assert repro_cfg["DATASETS"][split]["MAX_SIZE"] == 1000

    for module_name in (
        "configs.star_predcls_obb_ablation_unified_rca_train",
        "configs.star_predcls_obb_ablation_dual_rca_train",
        "configs.star_predcls_obb_ablation_dual_la_train",
    ):
        module = importlib.import_module(module_name)
        module = importlib.reload(module)
        cfg = module.cfg
        assert cfg["DATALOADER"].get("ASPECT_RATIO_GROUPING", False) is False
        for split in ("TRAIN", "VAL", "TEST"):
            assert cfg["DATASETS"][split].get("RESIZE_MODE", "fit") == "fit"
            assert cfg["DATASETS"][split]["IMAGE_SIZE"] == [1024, 1024]


def test_6850_cudnn_can_be_explicitly_restored(monkeypatch):
    monkeypatch.setenv("DISABLE_CUDNN", "0")
    module = importlib.import_module(
        "configs.star_predcls_obb_ablation_dual_6850_repro_train"
    )
    module = importlib.reload(module)
    assert module.cfg["RUNTIME"]["DISABLE_CUDNN"] is False
