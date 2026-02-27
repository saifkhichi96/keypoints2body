from __future__ import annotations

from typing import Optional

import torch

from ..core.config import (
    BodyModelConfig,
    FrameOptimizeConfig,
    ModelType,
    SequenceOptimizeConfig,
)
from ..core.engine import (
    OptimizeEngine,
    default_init_params,
    default_init_params_for_model,
    load_mean_pose_shape,
    optimize_shape_pass,
    upgrade_smpl_family_init_params,
)
from ..core.joints.adapters import (
    adapt_layout_and_conf,
    normalize_sequence_observations,
)
from ..models.smpl_data import (
    BodyModelFitResult,
    BodyModelParams,
    FLAMEData,
    MANOData,
    SMPLData,
    SMPLHData,
    SMPLXData,
)
from .model_factory import load_body_model

DEFAULT_MEAN_FILE = "./data/models/neutral_smpl_mean_params.h5"
OPTIMIZATION_BODY_MODELS = {"smpl", "smplh", "smplx", "mano", "flame"}


def optimize_params_sequence(
    joints_seq,
    *,
    init_params: Optional[BodyModelParams] = None,
    body_model: ModelType = "smpl",
    joint_layout: Optional[str] = None,
    model=None,
    config: Optional[SequenceOptimizeConfig | dict] = None,
    device=None,
) -> list[BodyModelFitResult]:
    """Optimize body parameters for a full motion sequence.

    Args:
        joints_seq: Sequence keypoints as either:
            - array/tensor with shape ``(T,K,3)`` or ``(T,K,4)``, or
            - dict blocks where each value is ``(T,K,3)`` or ``(T,K,4)``.
        init_params: Optional initial parameters for frame 0.
        body_model: Body model backend. Recognized values are ``smpl``, ``smplh``,
            ``smplx``, ``mano``, and ``flame``.
        joint_layout: Optional explicit layout label for adapter selection.
        model: Optional preloaded body model instance.
        config: Optional sequence config or dict equivalent.
        device: Optional torch device specifier.

    Returns:
        Per-frame optimization results in temporal order.
    """
    device = torch.device(device) if device is not None else torch.device("cpu")

    if isinstance(config, dict):
        frame_cfg = FrameOptimizeConfig(**config.get("frame", {}))
        seq_cfg = SequenceOptimizeConfig(
            frame=frame_cfg,
            num_shape_iters=config.get("num_shape_iters", 40),
            num_shape_frames=config.get("num_shape_frames", 50),
            use_shape_optimization=config.get("use_shape_optimization", True),
            use_previous_frame_init=config.get("use_previous_frame_init", True),
            fix_foot=config.get("fix_foot", False),
            limit_frames=config.get("limit_frames", None),
        )
    elif isinstance(config, SequenceOptimizeConfig):
        seq_cfg = config
    else:
        seq_cfg = SequenceOptimizeConfig()

    if seq_cfg.frame.input_type != "joints3d":
        raise NotImplementedError(
            f"input_type='{seq_cfg.frame.input_type}' is not implemented in this release. "
            "Current APIs support only joints3d."
        )
    is_ikgat = seq_cfg.frame.estimator_type == "ikgat"
    if body_model not in OPTIMIZATION_BODY_MODELS:
        raise ValueError(f"Unsupported body_model: {body_model}")
    xyz, conf, model_indices, in_layout = normalize_sequence_observations(
        joints_seq, layout=joint_layout, body_model=body_model
    )
    if body_model in {"smpl", "smplh", "smplx"} and in_layout != "GENERIC":
        xyz_np, conf_np, out_layout = adapt_layout_and_conf(
            xyz.cpu().numpy(), conf.cpu().numpy(), joint_layout
        )
        xyz = torch.as_tensor(xyz_np, dtype=torch.float32, device=device)
        conf = torch.as_tensor(conf_np, dtype=torch.float32, device=device)
        if out_layout not in ("SMPL24", "AMASS"):
            raise ValueError(f"Unsupported output layout after adaptation: {out_layout}")
        seq_cfg.frame.joints_category = out_layout
        model_indices = None
    else:
        if joint_layout is not None and in_layout != "GENERIC":
            raise ValueError(
                "joint_layout adapters are currently defined for SMPL-family body "
                "layouts only. Use raw MANO/FLAME joint order with joint_layout=None."
            )
        xyz = xyz.to(device)
        conf = conf.to(device)
        seq_cfg.frame.joints_category = "GENERIC"
        if model_indices is not None:
            model_indices = model_indices.to(device=device, dtype=torch.long)

    if seq_cfg.limit_frames is not None and seq_cfg.limit_frames > 0:
        xyz = xyz[: seq_cfg.limit_frames]
        conf = conf[: seq_cfg.limit_frames]

    if seq_cfg.fix_foot and xyz.shape[1] > 11:
        conf[:, 7] = 1.5
        conf[:, 8] = 1.5
        conf[:, 10] = 1.5
        conf[:, 11] = 1.5

    if model is None and not is_ikgat:
        body_cfg = BodyModelConfig(model_type=body_model)
        model = load_body_model(body_cfg, device)

    if is_ikgat:
        init_mean_pose = None
        init_mean_shape = None
        betas_opt = None
    elif body_model in {"smpl", "smplh", "smplx"}:
        init_mean_pose, init_mean_shape = load_mean_pose_shape(DEFAULT_MEAN_FILE, device)
        if seq_cfg.frame.joints_category != "GENERIC":
            betas_opt = optimize_shape_pass(
                model=model,
                seq_config=seq_cfg,
                init_mean_shape=init_mean_shape,
                init_mean_pose=init_mean_pose,
                data_tensor=xyz,
                confidence_input=conf[0],
                device=device,
            )
        else:
            betas_opt = init_mean_shape
    else:
        init_mean_pose = None
        init_mean_shape = None
        betas_opt = None

    engine = OptimizeEngine(
        model=model,
        frame_config=seq_cfg.frame,
        device=device,
        model_type=body_model,
    )
    results: list[BodyModelFitResult] = []

    if init_params is None:
        if is_ikgat:
            prev = SMPLData(
                betas=torch.zeros((1, 10), dtype=torch.float32, device=device),
                global_orient=torch.zeros((1, 3), dtype=torch.float32, device=device),
                body_pose=torch.zeros((1, 69), dtype=torch.float32, device=device),
                transl=xyz[0:1, 0, :].detach() if seq_cfg.frame.coordinate_mode == "world" else None,
            )
        elif body_model in {"smpl", "smplh", "smplx"}:
            base_init = default_init_params(
                init_mean_pose,
                betas_opt,
                xyz[0:1],
                model,
                joints_category=seq_cfg.frame.joints_category,
                coordinate_mode=seq_cfg.frame.coordinate_mode,
            )
            prev = upgrade_smpl_family_init_params(
                base_init,
                model_type=body_model,
                model=model,
                device=device,
            )
        else:
            prev = default_init_params_for_model(
                model_type=body_model,
                model=model,
                joints_frame=xyz[0:1],
                device=device,
                coordinate_mode=seq_cfg.frame.coordinate_mode,
            )
    else:
        expected_type = {
            "smpl": SMPLData,
            "smplh": SMPLHData,
            "smplx": SMPLXData,
            "mano": MANOData,
            "flame": FLAMEData,
        }[body_model]
        if not isinstance(init_params, expected_type):
            raise ValueError(
                f"init_params must be {expected_type.__name__} for body_model={body_model}."
            )
        prev = init_params.to(device)

    for idx in range(xyz.shape[0]):
        frame = xyz[idx : idx + 1]
        frame_conf = conf[idx]

        if seq_cfg.frame.coordinate_mode == "world" and prev.transl is None:
            if body_model in {"mano", "flame"}:
                prev.transl = frame[:, 0, :].detach()
            else:
                prev_before = prev
                pose = (
                    prev.pose
                    if isinstance(prev.pose, torch.Tensor)
                    else torch.as_tensor(prev.pose, dtype=torch.float32, device=device)
                )
                betas = (
                    prev.betas
                    if isinstance(prev.betas, torch.Tensor)
                    else torch.as_tensor(prev.betas, dtype=torch.float32, device=device)
                )
                prev = SMPLData(
                    betas=betas,
                    global_orient=pose[:, :3],
                    body_pose=pose[:, 3:],
                    transl=default_init_params(
                        pose,
                        betas,
                        frame,
                        model,
                        seq_cfg.frame.joints_category,
                        seq_cfg.frame.coordinate_mode,
                    ).transl,
                    metadata=dict(getattr(prev, "metadata", {})),
                )
                if isinstance(prev_before, SMPLHData):
                    prev = SMPLHData(
                        betas=prev.betas,
                        global_orient=prev.global_orient,
                        body_pose=prev.body_pose,
                        transl=prev.transl,
                        metadata=prev.metadata,
                        left_hand_pose=prev_before.left_hand_pose,
                        right_hand_pose=prev_before.right_hand_pose,
                    )
                if isinstance(prev_before, SMPLXData):
                    prev = SMPLXData(
                        betas=prev.betas,
                        global_orient=prev.global_orient,
                        body_pose=prev.body_pose,
                        transl=prev.transl,
                        metadata=prev.metadata,
                        left_hand_pose=prev_before.left_hand_pose,
                        right_hand_pose=prev_before.right_hand_pose,
                        expression=prev_before.expression,
                        jaw_pose=prev_before.jaw_pose,
                        leye_pose=prev_before.leye_pose,
                        reye_pose=prev_before.reye_pose,
                    )

        res = engine.fit_frame(
            init_params=prev,
            j3d=frame,
            conf_3d=frame_conf,
            seq_ind=idx,
            target_model_indices=model_indices,
        )
        results.append(res)
        if seq_cfg.use_previous_frame_init:
            prev = res.params

    return results


def optimize_shape_sequence(
    joints_seq,
    *,
    body_model: ModelType = "smpl",
    joint_layout: Optional[str] = None,
    model=None,
    config: Optional[SequenceOptimizeConfig | dict] = None,
    device=None,
) -> BodyModelParams:
    """Run sequence optimization and return the final frame parameters.

    Args:
        joints_seq: Sequence keypoints input.
        body_model: Body model backend identifier.
        joint_layout: Optional explicit layout label.
        model: Optional preloaded body model instance.
        config: Optional sequence config.
        device: Optional torch device specifier.

    Returns:
        Parameter object from the last optimized frame.
    """
    results = optimize_params_sequence(
        joints_seq,
        init_params=None,
        body_model=body_model,
        joint_layout=joint_layout,
        model=model,
        config=config,
        device=device,
    )
    if not results:
        raise ValueError("No frames were optimized")
    return results[-1].params
