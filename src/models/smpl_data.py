from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _concat_pose(global_orient: ArrayLike, body_pose: ArrayLike) -> ArrayLike:
    if isinstance(global_orient, torch.Tensor):
        return torch.cat([global_orient, body_pose], dim=-1)
    return np.concatenate([global_orient, body_pose], axis=-1)


@dataclass
class BodyModelParams:
    """Base container for body-model parameters."""

    betas: ArrayLike
    global_orient: ArrayLike
    body_pose: ArrayLike
    transl: Optional[ArrayLike] = None

    @property
    def pose(self) -> ArrayLike:
        return _concat_pose(self.global_orient, self.body_pose)


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
