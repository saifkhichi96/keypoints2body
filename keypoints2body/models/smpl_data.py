from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _concat_pose(global_orient: ArrayLike, body_pose: ArrayLike) -> ArrayLike:
    """Concatenate global orientation and body pose arrays."""
    if isinstance(global_orient, torch.Tensor):
        return torch.cat([global_orient, body_pose], dim=-1)
    return np.concatenate([global_orient, body_pose], axis=-1)


def _to_impl(x: ArrayLike, *, device: Optional[torch.device] = None) -> ArrayLike:
    """Move tensor input to device, pass through numpy unchanged."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device) if device is not None else x
    return x


def _detach_impl(x: ArrayLike) -> ArrayLike:
    """Detach tensor input, pass through numpy unchanged."""
    if isinstance(x, torch.Tensor):
        return x.detach()
    return x


@dataclass
class BodyModelParams:
    """Base container for body-model parameters."""

    betas: ArrayLike
    global_orient: ArrayLike
    body_pose: ArrayLike
    transl: Optional[ArrayLike] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pose(self) -> ArrayLike:
        return _concat_pose(self.global_orient, self.body_pose)

    def validate(self) -> None:
        if self.betas is None or self.global_orient is None or self.body_pose is None:
            raise ValueError("betas, global_orient, and body_pose are required")

    def to(self, device: torch.device) -> "BodyModelParams":
        return replace(
            self,
            betas=_to_impl(self.betas, device=device),
            global_orient=_to_impl(self.global_orient, device=device),
            body_pose=_to_impl(self.body_pose, device=device),
            transl=_to_impl(self.transl, device=device)
            if self.transl is not None
            else None,
        )

    def detach(self) -> "BodyModelParams":
        return replace(
            self,
            betas=_detach_impl(self.betas),
            global_orient=_detach_impl(self.global_orient),
            body_pose=_detach_impl(self.body_pose),
            transl=_detach_impl(self.transl) if self.transl is not None else None,
        )


@dataclass
class SMPLData(BodyModelParams):
    """SMPL parameters."""


@dataclass
class SMPLHData(SMPLData):
    """SMPL-H parameters."""

    left_hand_pose: Optional[ArrayLike] = None
    right_hand_pose: Optional[ArrayLike] = None


@dataclass
class SMPLXData(SMPLHData):
    """SMPL-X parameters."""

    expression: Optional[ArrayLike] = None
    jaw_pose: Optional[ArrayLike] = None
    leye_pose: Optional[ArrayLike] = None
    reye_pose: Optional[ArrayLike] = None


@dataclass
class BodyModelFitResult:
    """Typed result for fitting routines."""

    params: BodyModelParams
    vertices: torch.Tensor
    joints: torch.Tensor
    loss: Optional[torch.Tensor] = None
