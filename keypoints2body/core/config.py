from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

ModelType = Literal["smpl", "smplh", "smplx", "mano", "flame"]


@dataclass
class BodyModelConfig:
    """Configuration for loading a body model backend."""

    model_type: ModelType = "smpl"
    model_family: str = "smpl_family"
    gender: str = "neutral"
    ext: Optional[str] = None
    batch_size: int = 1
    model_dir: Path = Path("./data/models/")


@dataclass
class FrameOptimizeConfig:
    """Configuration for single-frame estimation/optimization."""

    estimator_type: Literal["optimization", "learned"] = "optimization"
    input_type: Literal["joints3d", "joints2d", "multiview_joints2d"] = "joints3d"
    coordinate_mode: Literal["camera", "world"] = "world"
    use_lbfgs: bool = True
    step_size: float = 1e-2
    num_iters: int = 100
    num_iters_first: int = 30
    num_iters_followup: int = 10
    joint_loss_weight: float = 600.0
    pose_preserve_weight: float = 5.0
    freeze_betas: bool = False
    shape_prior_weight: float = 5.0
    pose_prior_num_gaussians: int = 8
    joints_category: Literal["SMPL24", "AMASS", "GENERIC"] = "AMASS"


@dataclass
class SequenceOptimizeConfig:
    """Configuration for sequence-level fitting behavior."""

    frame: FrameOptimizeConfig = field(default_factory=FrameOptimizeConfig)
    num_shape_iters: int = 40
    num_shape_frames: int = 50
    use_shape_optimization: bool = True
    use_previous_frame_init: bool = True
    fix_foot: bool = False
    limit_frames: Optional[int] = None
