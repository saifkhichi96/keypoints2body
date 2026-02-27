from __future__ import annotations

from typing import Optional

import h5py
import torch

from ..models.smpl_data import (
    BodyModelFitResult,
    BodyModelParams,
    FLAMEData,
    MANOData,
    SMPLData,
)
from .config import FrameOptimizeConfig, SequenceOptimizeConfig
from .estimators.factory import create_estimator
from .fitters.world_space import guess_init_transl_from_root
from .shape import optimize_shape_multi_frame


class OptimizeEngine:
    """Coordinator that routes frame fitting to the configured estimator."""

    def __init__(
        self,
        model,
        frame_config: FrameOptimizeConfig,
        device: torch.device,
        model_type: str = "smpl",
    ):
        self.model = model
        self.frame_config = frame_config
        self.device = device
        self.estimator = create_estimator(
            model=model,
            frame_config=frame_config,
            device=device,
            model_type=model_type,
        )

    def fit_frame(
        self,
        init_params: BodyModelParams,
        j3d: torch.Tensor,
        conf_3d: Optional[torch.Tensor],
        seq_ind: int,
    ) -> BodyModelFitResult:
        """Fit one frame using the active estimator.

        Args:
            init_params: Initialization parameters for this frame.
            j3d: Canonical 3D joints tensor with shape ``(1,K,3)``.
            conf_3d: Optional confidence vector for joints.
            seq_ind: Sequence index, used by temporal regularization.

        Returns:
            A fitting result object.
        """
        return self.estimator.fit_frame(
            init_params=init_params,
            j3d=j3d,
            conf_3d=conf_3d,
            seq_ind=seq_ind,
        )


def load_mean_pose_shape(
    mean_file: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load mean pose and mean shape tensors from an H5 file.

    Args:
        mean_file: Path to ``neutral_smpl_mean_params.h5``-style file.
        device: Target torch device.

    Returns:
        Tuple of ``(mean_pose, mean_shape)`` tensors.
    """
    with h5py.File(mean_file, "r") as f:
        init_mean_pose = torch.as_tensor(f["pose"][:]).unsqueeze(0).float().to(device)
        init_mean_shape = torch.as_tensor(f["shape"][:]).unsqueeze(0).float().to(device)
    return init_mean_pose, init_mean_shape


def default_init_params(
    mean_pose: torch.Tensor,
    mean_shape: torch.Tensor,
    joints_frame: torch.Tensor,
    model,
    joints_category: str,
    coordinate_mode: str,
) -> SMPLData:
    """Create default SMPL initialization parameters.

    Args:
        mean_pose: Mean pose tensor.
        mean_shape: Mean shape tensor.
        joints_frame: Frame joints used for translation initialization.
        model: Loaded body model.
        joints_category: Joint category used for indexing.
        coordinate_mode: ``camera`` or ``world``.

    Returns:
        SMPL parameter object initialized from mean values.
    """
    pose = mean_pose.clone().detach()
    betas = mean_shape.clone().detach()
    if coordinate_mode == "world":
        transl = guess_init_transl_from_root(
            model, pose, betas, joints_frame, joints_category=joints_category
        )
    else:
        transl = None
    return SMPLData(
        betas=betas, global_orient=pose[:, :3], body_pose=pose[:, 3:], transl=transl
    )


def default_init_params_for_model(
    model_type: str,
    model,
    joints_frame: torch.Tensor,
    device: torch.device,
    coordinate_mode: str,
) -> BodyModelParams:
    """Create model-aware zero initialization for non-SMPL-family fitters."""
    batch = joints_frame.shape[0]
    num_betas = int(getattr(model, "num_betas", 10))
    transl = joints_frame[:, 0, :].clone().detach() if coordinate_mode == "world" else None

    if model_type == "mano":
        num_hand_joints = int(getattr(model, "NUM_HAND_JOINTS", 15))
        return MANOData(
            betas=torch.zeros((batch, num_betas), device=device),
            global_orient=torch.zeros((batch, 3), device=device),
            body_pose=torch.zeros((batch, 0), device=device),
            transl=transl,
            hand_pose=torch.zeros((batch, num_hand_joints * 3), device=device),
        )
    if model_type == "flame":
        num_expr = int(getattr(model, "num_expression_coeffs", 10))
        return FLAMEData(
            betas=torch.zeros((batch, num_betas), device=device),
            global_orient=torch.zeros((batch, 3), device=device),
            body_pose=torch.zeros((batch, 0), device=device),
            transl=transl,
            expression=torch.zeros((batch, num_expr), device=device),
            jaw_pose=torch.zeros((batch, 3), device=device),
            neck_pose=torch.zeros((batch, 3), device=device),
            leye_pose=torch.zeros((batch, 3), device=device),
            reye_pose=torch.zeros((batch, 3), device=device),
        )
    raise ValueError(f"default_init_params_for_model is unsupported for {model_type}")


def optimize_shape_pass(
    model,
    seq_config: SequenceOptimizeConfig,
    init_mean_shape: torch.Tensor,
    init_mean_pose: torch.Tensor,
    data_tensor: torch.Tensor,
    confidence_input: torch.Tensor,
    device: torch.device,
):
    """Run optional multi-frame shape optimization stage.

    Args:
        model: Loaded body model.
        seq_config: Sequence configuration.
        init_mean_shape: Initial shape tensor.
        init_mean_pose: Initial pose tensor.
        data_tensor: Sequence joints tensor.
        confidence_input: Per-joint confidence tensor.
        device: Torch device.

    Returns:
        Optimized shape tensor (or initial shape when disabled).
    """
    if not seq_config.use_shape_optimization:
        return init_mean_shape

    t_size = data_tensor.shape[0]
    if seq_config.num_shape_frames < 0 or seq_config.num_shape_frames >= t_size:
        frame_indices = list(range(t_size))
    else:
        frame_indices = list(range(seq_config.num_shape_frames))

    return optimize_shape_multi_frame(
        model,
        init_betas=init_mean_shape,
        pose_init=init_mean_pose.repeat(t_size, 1),
        j3d_world=data_tensor,
        joints_category=seq_config.frame.joints_category,
        num_iters=seq_config.num_shape_iters,
        step_size=1e-1,
        use_lbfgs=seq_config.frame.use_lbfgs,
        device=device,
        frame_indices=frame_indices,
        joints3d_conf=confidence_input,
        shape_prior_weight=seq_config.frame.shape_prior_weight,
    )
