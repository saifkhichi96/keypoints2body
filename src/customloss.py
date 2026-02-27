import torch

import config


def gmof(x, sigma):
    """
    Geman-McClure robust error.

    Args:
        x: (...,) error
        sigma: scalar

    Returns:
        (...,) GMoF error
    """
    x_squared = x**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows.

    Args:
        pose: (B, 69) body pose without global orientation

    Returns:
        (B, 4) angle prior per sample
    """
    return (
        torch.exp(
            pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]]
            * torch.tensor([1.0, -1.0, -1.0, -1.0], device=pose.device)
        )
        ** 2
    )


def body_fitting_loss_3d(
    body_pose,
    preserve_pose,
    betas,
    model_joints,
    j3d,
    pose_prior,
    joints3d_conf=None,
    camera_translation=None,
    sigma=100,
    pose_prior_weight=4.78 * 1.5,
    shape_prior_weight=5.0,
    angle_prior_weight=15.2,
    joint_loss_weight=500.0,
    pose_preserve_weight=0.0,
) -> torch.Tensor:
    """
    Core 3D body fitting loss.

    When camera_translation is provided, the model_joints are assumed to be in
    camera coordinates, and we add the translation to bring them to world coords.
    Otherwise, they are assumed to be in world coordinates already.

    Args:
        body_pose: (B, 69) body pose (without global orient)
        preserve_pose: (B, 69) pose to preserve (e.g. previous frame)
        betas: (B, 10)
        model_joints: (B, J, 3) SMPL joints in world coords (subset already)
        j3d: (B, J, 3) target 3D joints in world coords (mapped to same indices)
        pose_prior: MaxMixturePrior
        joints3d_conf: (J,) or (B, J) confidence, optional
        camera_translation: (B, 3) camera translation in world coords, optional
        sigma: GMoF sigma
        pose_prior_weight: weight for pose prior
        shape_prior_weight: weight for shape prior
        angle_prior_weight: weight for angle prior
        joint_loss_weight: weight for 3D joint loss
        pose_preserve_weight: weight for temporal regularizer
    """
    if joints3d_conf is None:
        joints3d_conf = torch.ones(model_joints.shape[1], device=model_joints.device)
    if joints3d_conf.dim() == 1:
        joints3d_conf = joints3d_conf.view(1, -1)

    # World coordinate conversion (if needed)
    if camera_translation is not None:
        model_joints = model_joints + camera_translation

    # 3D joint loss
    joint3d_error = gmof(model_joints - j3d, sigma)  # (B, J, 3)
    joint3d_loss_part = (joints3d_conf**2) * joint3d_error.sum(dim=-1)  # (B, J)
    joint3d_loss = (joint_loss_weight**2) * joint3d_loss_part.sum(dim=-1)  # (B,)

    # Pose prior loss
    pose_prior_loss = (pose_prior_weight**2) * pose_prior(body_pose, betas)

    # Angle prior loss (knees, elbows)
    angle_prior_loss = (angle_prior_weight**2) * angle_prior(body_pose).sum(dim=-1)

    # Shape regularizer
    shape_prior_loss = (shape_prior_weight**2) * (betas**2).sum(dim=-1)

    # Temporal/pose-preserve regularizer
    pose_preserve_loss = (pose_preserve_weight**2) * (
        (body_pose - preserve_pose) ** 2
    ).sum(dim=-1)

    total_loss = (
        joint3d_loss
        + pose_prior_loss
        + angle_prior_loss
        + shape_prior_loss
        + pose_preserve_loss
    )
    return total_loss.sum()


def camera_fitting_loss_3d(
    model_joints,
    camera_t,
    camera_t_est,
    j3d,
    joints_category="SMPL24",
    depth_loss_weight=100.0,
) -> torch.Tensor:
    """
    Loss for fitting camera translation using 3D joints and depth estimate.

    Args:
        model_joints: (B, J, 3) SMPL joints in model coords
        camera_t: (B, 3) camera translation in model coords
        camera_t_est: (B, 3) estimated camera translation in model coords
        j3d: (B, J, 3) target 3D joints in world coords (mapped to same indices)
        joints_category: "SMPL24" or "AMASS" joint indexing for j3d
        depth_loss_weight: weight for depth loss

    Returns:
        total_loss: scalar tensor representing the total loss
    """
    model_joints = model_joints + camera_t

    # get the indexed four
    gt_joints = ["RHip", "LHip", "RShoulder", "LShoulder"]
    gt_joints_ind = [config.JOINT_MAP[joint] for joint in gt_joints]

    if joints_category == "SMPL24":
        select_joints_ind = [config.JOINT_MAP[joint] for joint in gt_joints]
    elif joints_category == "AMASS":
        select_joints_ind = [config.AMASS_JOINT_MAP[joint] for joint in gt_joints]
    else:
        raise ValueError(f"Unknown joints category: {joints_category}")

    j3d_error_loss = (j3d[:, select_joints_ind] - model_joints[:, gt_joints_ind]) ** 2

    # Loss that penalizes deviation from depth estimate
    depth_loss = (depth_loss_weight**2) * (camera_t - camera_t_est) ** 2

    total_loss = j3d_error_loss + depth_loss
    return total_loss.sum()
