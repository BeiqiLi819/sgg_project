from __future__ import annotations

import json

import torch

from sgg.engine.trainer import Trainer, _ValidationEarlyStopper


def test_trainer_normalizes_cli_device_string_for_profiled_evaluation(tmp_path):
    cfg = {
        "MODEL": {},
        "DATALOADER": {"TRAIN_BATCH_SIZE": 1},
        "SOLVER": {
            "BASE_LR": 1e-3,
            "LR_SCALE_BY_BATCH": False,
            "OUTPUT_DIR": str(tmp_path),
            "SCHEDULE": {"TYPE": "none"},
        },
    }

    trainer = Trainer(cfg, torch.nn.Linear(2, 2), device="cpu")

    assert trainer.device == torch.device("cpu")
    assert trainer.device.type == "cpu"


def test_reset_scheduler_preserves_optimizer_momentum(tmp_path):
    cfg = {
        "MODEL": {},
        "DATALOADER": {"TRAIN_BATCH_SIZE": 1},
        "SOLVER": {
            "BASE_LR": 0.1,
            "LR_SCALE_BY_BATCH": False,
            "OPTIMIZER": "SGD",
            "MOMENTUM": 0.9,
            "OUTPUT_DIR": str(tmp_path),
            "WARMUP_ITERS": 0,
            "STEPS": [100],
            "GAMMA": 0.1,
            "SCHEDULE": {
                "TYPE": "WarmupMultiStepLR",
                "UNIT": "iter",
            },
        },
    }
    model = torch.nn.Linear(2, 1)
    trainer = Trainer(cfg, model, device="cpu")
    model(torch.ones(1, 2)).sum().backward()
    trainer.optimizer.step()

    parameter = next(model.parameters())
    momentum_before = trainer.optimizer.state[parameter][
        "momentum_buffer"
    ].clone()
    optimizer_id = id(trainer.optimizer)

    cfg["SOLVER"]["STEPS"] = [5, 8]
    trainer.reset_scheduler_state(global_step=6)

    assert id(trainer.optimizer) == optimizer_id
    torch.testing.assert_close(
        trainer.optimizer.state[parameter]["momentum_buffer"],
        momentum_before,
    )
    assert trainer.global_step == 6
    assert trainer.scheduler.milestones == [5, 8]
    assert abs(trainer.optimizer.param_groups[0]["lr"] - 0.01) < 1e-12


def test_validation_early_stopper_counts_validation_events():
    stopper = _ValidationEarlyStopper(
        enabled=True,
        metric="HR",
        patience=3,
        min_delta=0.0,
        start_period=120,
    )

    assert stopper.update(118, 0.50) == (False, False)
    assert stopper.update(120, 0.50) == (False, True)
    assert stopper.update(122, 0.49) == (False, False)
    assert stopper.update(124, 0.50) == (False, False)
    assert stopper.update(126, 0.48) == (True, False)
    assert stopper.best == 0.50
    assert stopper.best_epoch == 120
    assert stopper.bad_validations == 3


def test_validation_early_stopper_accepts_hmr_alias():
    stopper = _ValidationEarlyStopper(
        enabled=True,
        metric="hmr",
        patience=1,
    )

    assert stopper.metric == "HR"


def test_validation_early_stopper_state_keeps_current_config():
    source = _ValidationEarlyStopper(
        enabled=True,
        metric="HR",
        patience=3,
    )
    source.update(2, 0.4)
    source.update(4, 0.3)

    restored = _ValidationEarlyStopper(
        enabled=True,
        metric="HR",
        patience=10,
    )
    restored.load_state_dict(source.state_dict())

    assert restored.patience == 10
    assert restored.best == 0.4
    assert restored.best_epoch == 2
    assert restored.bad_validations == 1


def test_trainer_restores_patience_from_validation_history(tmp_path):
    records = [
        {
            "epoch": 120,
            "split": "val",
            "filter_method": "PPG",
            "metrics": {
                "R": {"2000": 0.60},
                "mR": {"2000": 0.30},
                "HR": {"2000": 0.40},
            },
        },
        {
            "epoch": 122,
            "split": "val",
            "filter_method": "PPG",
            "metrics": {
                "R": {"2000": 0.61},
                "mR": {"2000": 0.29},
                "HR": {"2000": 0.39},
            },
        },
        {
            "epoch": 124,
            "split": "val",
            "filter_method": "PPG",
            "metrics": {
                "R": {"2000": 0.62},
                "mR": {"2000": 0.28},
                "HR": {"2000": 0.38},
            },
        },
    ]
    history = tmp_path / "validation_history.jsonl"
    history.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    trainer = Trainer.__new__(Trainer)
    trainer.output_dir = tmp_path
    trainer.cfg = {
        "SOLVER": {"VAL_SPLIT": "val"},
        "MODEL": {
            "ROI_RELATION_HEAD": {
                "TEST_FILTER_METHOD": "PPG",
            }
        },
    }
    trainer.best_metrics = {
        "R": float("-inf"),
        "mR": float("-inf"),
        "HR": float("-inf"),
    }
    trainer.early_stopper = _ValidationEarlyStopper(
        enabled=True,
        metric="HR",
        patience=2,
        start_period=120,
    )

    assert trainer._restore_validation_tracking_from_history(
        max_epoch=124,
        recall_k=2000,
    )
    assert trainer.best_metrics == {
        "R": 0.62,
        "mR": 0.30,
        "HR": 0.40,
    }
    assert trainer.early_stopper.best_epoch == 120
    assert trainer.early_stopper.bad_validations == 2
    assert trainer.early_stopper.should_stop
