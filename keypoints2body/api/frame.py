from __future__ import annotations

from typing import Optional

import torch

from ..core.config import BodyModelConfig, FrameOptimizeConfig, ModelType
from ..core.engine import (
    OptimizeEngine,
    default_init_params,
    default_init_params_for_model,
    load_mean_pose_shape,
)
from ..core.joints.adapters import adapt_layout_and_conf, normalize_joints_frame
from ..models.smpl_data import (
    BodyModelFitResult,
    BodyModelParams,
    FLAMEData,
    MANOData,
    SMPLData,
)
from .model_factory import load_body_model

DEFAULT_MEAN_FILE = "./data/models/neutral_smpl_mean_params.h5"
OPTIMIZATION_BODY_MODELS = {"smpl", "smplh", "smplx", "mano", "flame"}


def optimize_params_frame(
    joints,
    *,
    prev_params: Optional[BodyModelParams] = None,
    body_model: ModelType = "smpl",
    joint_layout: Optional[str] = None,
    model=None,
    config: Optional[FrameOptimizeConfig | dict] = None,
    device=None,
) -> BodyModelFitResult:
    """Optimize body parameters for a single frame of 3D joints.

    Args:
        joints: Frame keypoints with shape ``(K,3)`` or ``(K,4)``.
        prev_params: Optional warm-start parameters from a previous fit.
        body_model: Body model backend. Recognized values are ``smpl``, ``smplh``,
            ``smplx``, ``mano``, and ``flame``.
        joint_layout: Optional explicit layout label for adapter selection.
        model: Optional preloaded body model instance.
        config: Optional frame config or dict equivalent.
        device: Optional torch device specifier.

    Returns:
        Typed fitting result containing optimized parameters, joints, vertices, and loss.
    """
    device = torch.device(device) if device is not None else torch.device("cpu")

    if isinstance(config, dict):
        frame_cfg = FrameOptimizeConfig(**config)
    elif isinstance(config, FrameOptimizeConfig):
        frame_cfg = config
    else:
        frame_cfg = FrameOptimizeConfig()

    if frame_cfg.input_type != "joints3d":
        raise NotImplementedError(
            f"input_type='{frame_cfg.input_type}' is not implemented in this release. "
            "Current APIs support only joints3d."
        )
    if body_model not in OPTIMIZATION_BODY_MODELS:
        raise ValueError(f"Unsupported body_model: {body_model}")
    j3d, conf_3d = normalize_joints_frame(joints)
    if body_model in {"smpl", "smplh", "smplx"}:
        j_np = j3d.squeeze(0).cpu().numpy()[None, ...]
        c_np = conf_3d.cpu().numpy()[None, ...]
        j_np, c_np, out_layout = adapt_layout_and_conf(j_np, c_np, joint_layout)
        j3d = torch.as_tensor(j_np, dtype=torch.float32, device=device)
        conf_3d = torch.as_tensor(c_np[0], dtype=torch.float32, device=device)
        if out_layout not in ("SMPL24", "AMASS"):
            raise ValueError(f"Unsupported output layout after adaptation: {out_layout}")
        frame_cfg.joints_category = out_layout
    else:
        if joint_layout is not None:
            raise ValueError(
                "joint_layout adapters are currently defined for SMPL-family body "
                "layouts only. Use raw MANO/FLAME joint order with joint_layout=None."
            )
        j3d = j3d.to(device)
        conf_3d = conf_3d.to(device)
        frame_cfg.joints_category = "GENERIC"

    if model is None:
        body_cfg = BodyModelConfig(model_type=body_model)
        model = load_body_model(body_cfg, device)

    engine = OptimizeEngine(
        model=model, frame_config=frame_cfg, device=device, model_type=body_model
    )

    if prev_params is None:
        if body_model in {"smpl", "smplh", "smplx"}:
            init_mean_pose, init_mean_shape = load_mean_pose_shape(
                DEFAULT_MEAN_FILE, device
            )
            init_params = default_init_params(
                init_mean_pose,
                init_mean_shape,
                j3d,
                model,
                joints_category=frame_cfg.joints_category,
                coordinate_mode=frame_cfg.coordinate_mode,
            )
        else:
            init_params = default_init_params_for_model(
                model_type=body_model,
                model=model,
                joints_frame=j3d,
                device=device,
                coordinate_mode=frame_cfg.coordinate_mode,
            )
    else:
        expected_type = {
            "smpl": SMPLData,
            "smplh": SMPLData,
            "smplx": SMPLData,
            "mano": MANOData,
            "flame": FLAMEData,
        }[body_model]
        if not isinstance(prev_params, expected_type):
            raise ValueError(
                f"prev_params must be {expected_type.__name__} for body_model={body_model}."
            )
        init_params = prev_params.to(device)
        if frame_cfg.coordinate_mode == "world" and init_params.transl is None:
            if body_model in {"smpl", "smplh", "smplx"}:
                pose = init_params.pose
                if not isinstance(pose, torch.Tensor):
                    pose = torch.as_tensor(pose, dtype=torch.float32, device=device)
                betas = init_params.betas
                if not isinstance(betas, torch.Tensor):
                    betas = torch.as_tensor(betas, dtype=torch.float32, device=device)
                transl = default_init_params(
                    pose,
                    betas,
                    j3d,
                    model,
                    frame_cfg.joints_category,
                    frame_cfg.coordinate_mode,
                ).transl
                init_params = SMPLData(
                    betas=betas,
                    global_orient=pose[:, :3],
                    body_pose=pose[:, 3:],
                    transl=transl,
                    metadata=dict(getattr(init_params, "metadata", {})),
                )
            else:
                init_params.transl = j3d[:, 0, :].detach()

    return engine.fit_frame(
        init_params=init_params, j3d=j3d, conf_3d=conf_3d, seq_ind=0
    )
