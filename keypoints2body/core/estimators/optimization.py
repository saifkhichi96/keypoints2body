from __future__ import annotations

from typing import Optional

import torch

from ...models.smpl_data import BodyModelFitResult, BodyModelParams
from ..config import FrameOptimizeConfig
from ..fitters.camera_space import CameraSpaceFitter
from ..fitters.world_space import WorldSpaceFitter


class OptimizationEstimator:
    """Adapter that exposes existing optimizer fitters via BodyEstimator interface."""

    def __init__(self, model, frame_config: FrameOptimizeConfig, device: torch.device):
        self.frame_config = frame_config
        if frame_config.coordinate_mode == "camera":
            self._fitter = CameraSpaceFitter(
                smpl_model=model,
                step_size=frame_config.step_size,
                num_iters=frame_config.num_iters,
                use_lbfgs=frame_config.use_lbfgs,
                joints_category=frame_config.joints_category,
                device=device,
                pose_prior_num_gaussians=frame_config.pose_prior_num_gaussians,
            )
        else:
            self._fitter = WorldSpaceFitter(
                smpl_model=model,
                step_size=frame_config.step_size,
                num_iters_first=frame_config.num_iters_first,
                num_iters_followup=frame_config.num_iters_followup,
                use_lbfgs=frame_config.use_lbfgs,
                joints_category=frame_config.joints_category,
                device=device,
                pose_prior_num_gaussians=frame_config.pose_prior_num_gaussians,
            )

    def fit_frame(
        self,
        init_params: BodyModelParams,
        j3d: torch.Tensor,
        conf_3d: Optional[torch.Tensor],
        seq_ind: int,
    ) -> BodyModelFitResult:
        kwargs = dict(
            init_params=init_params,
            j3d=j3d,
            conf_3d=conf_3d,
            seq_ind=seq_ind,
            joint_loss_weight=self.frame_config.joint_loss_weight,
            pose_preserve_weight=self.frame_config.pose_preserve_weight,
            freeze_betas=self.frame_config.freeze_betas,
        )
        return self._fitter.fit_frame(**kwargs)
