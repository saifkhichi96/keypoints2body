import torch

from .constants import AMASS_JOINT_MAP, JOINT_MAP


def gmof(x, sigma):
    """Compute Geman-McClure robust error."""
    x_squared = x**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def angle_prior(pose):
    """Compute elbow/knee bending prior penalty."""
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
    """Compute total 3D body fitting loss."""
    if joints3d_conf is None:
        joints3d_conf = torch.ones(model_joints.shape[1], device=model_joints.device)
    if joints3d_conf.dim() == 1:
        joints3d_conf = joints3d_conf.view(1, -1)

    if camera_translation is not None:
        model_joints = model_joints + camera_translation

    joint3d_error = gmof(model_joints - j3d, sigma)
    joint3d_loss_part = (joints3d_conf**2) * joint3d_error.sum(dim=-1)
    joint3d_loss = (joint_loss_weight**2) * joint3d_loss_part.sum(dim=-1)

    pose_prior_loss = (pose_prior_weight**2) * pose_prior(body_pose, betas)
    angle_prior_loss = (angle_prior_weight**2) * angle_prior(body_pose).sum(dim=-1)
    shape_prior_loss = (shape_prior_weight**2) * (betas**2).sum(dim=-1)
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
    """Compute camera translation fitting loss."""
    model_joints = model_joints + camera_t

    gt_joints = ["RHip", "LHip", "RShoulder", "LShoulder"]
    gt_joints_ind = [JOINT_MAP[joint] for joint in gt_joints]

    if joints_category == "SMPL24":
        select_joints_ind = [JOINT_MAP[joint] for joint in gt_joints]
    elif joints_category == "AMASS":
        select_joints_ind = [AMASS_JOINT_MAP[joint] for joint in gt_joints]
    else:
        raise ValueError(f"Unknown joints category: {joints_category}")

    j3d_error_loss = (j3d[:, select_joints_ind] - model_joints[:, gt_joints_ind]) ** 2
    depth_loss = (depth_loss_weight**2) * (camera_t - camera_t_est) ** 2
    return (j3d_error_loss + depth_loss).sum()


def generic_keypoint_loss_3d(
    model_joints,
    j3d,
    joints3d_conf=None,
    sigma=100.0,
    joint_loss_weight=500.0,
):
    """Compute robust keypoint fitting loss for generic models."""
    if joints3d_conf is None:
        joints3d_conf = torch.ones(model_joints.shape[1], device=model_joints.device)
    if joints3d_conf.dim() == 1:
        joints3d_conf = joints3d_conf.view(1, -1)

    joint3d_error = gmof(model_joints - j3d, sigma)
    joint3d_loss_part = (joints3d_conf**2) * joint3d_error.sum(dim=-1)
    joint3d_loss = (joint_loss_weight**2) * joint3d_loss_part.sum(dim=-1)
    return joint3d_loss.sum()
