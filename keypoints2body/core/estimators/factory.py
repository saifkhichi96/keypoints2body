from __future__ import annotations

import torch

from ..config import FrameOptimizeConfig
from .optimization import OptimizationEstimator


class LearnedEstimator:
    """Placeholder for future learned predictors from keypoints to params."""

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "estimator_type='learned' is not implemented yet. "
            "Implement under keypoints2body.core.estimators and wire model loading/inference."
        )


def create_estimator(model, frame_config: FrameOptimizeConfig, device: torch.device):
    """Instantiate an estimator implementation from frame config.

    Args:
        model: Loaded body model module.
        frame_config: Frame-level estimator configuration.
        device: Torch device.

    Returns:
        Estimator object implementing ``fit_frame``.
    """
    if frame_config.estimator_type == "optimization":
        return OptimizationEstimator(
            model=model, frame_config=frame_config, device=device
        )
    if frame_config.estimator_type == "learned":
        return LearnedEstimator(model=model, frame_config=frame_config, device=device)
    raise ValueError(f"Unknown estimator_type: {frame_config.estimator_type}")
