"""Set-conditioned residual proposal scorer used by STAR-Q2."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from sgg.modeling.roi_heads.set_context import SET_CONTEXT_FEATURE_NAMES


class SetConditionedPPGResidual(nn.Module):
    """A small residual MLP; its zero output initialization is an exact identity."""

    def __init__(
        self,
        input_dim: int = len(SET_CONTEXT_FEATURE_NAMES),
        hidden_dim: int = 32,
        feature_names: list[str] | tuple[str, ...] | None = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.feature_names = tuple(feature_names or SET_CONTEXT_FEATURE_NAMES)
        if len(self.feature_names) != self.input_dim:
            raise ValueError("feature_names length must equal input_dim")
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.size(1) != self.input_dim:
            raise ValueError(f"Expected context [M,{self.input_dim}], got {tuple(context.shape)}")
        return self.network(context.float()).squeeze(1)

    def scores(self, base_scores: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residual = self(context)
        if residual.shape != base_scores.shape:
            raise ValueError("base score and residual shapes differ")
        return base_scores + residual

    def architecture_manifest(self) -> dict[str, object]:
        return {
            "kind": "set_conditioned_residual_mlp",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_layers": 2,
            "activation": "SiLU",
            "output": "unbounded scalar residual",
            "output_initialization": "all-zero weight and bias",
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: Mapping[str, object]) -> "SetConditionedPPGResidual":
        architecture = payload.get("architecture")
        if not isinstance(architecture, Mapping):
            raise TypeError("STAR-Q2 checkpoint lacks architecture")
        model = cls(
            input_dim=int(architecture["input_dim"]),
            hidden_dim=int(architecture["hidden_dim"]),
            feature_names=architecture.get("feature_names"),
        )
        state = payload.get("residual_state_dict")
        if not isinstance(state, Mapping):
            raise TypeError("STAR-Q2 checkpoint lacks residual_state_dict")
        model.load_state_dict(state)
        return model


def module_state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_set_conditioned_checkpoint(
    path: str | Path,
    model: SetConditionedPPGResidual,
    *,
    metadata: Mapping[str, object],
) -> None:
    torch.save(
        {
            "schema_version": 1,
            "kind": "star_q2_set_conditioned_residual_ppg",
            "architecture": model.architecture_manifest(),
            "residual_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "metadata": dict(metadata),
        },
        Path(path),
    )
