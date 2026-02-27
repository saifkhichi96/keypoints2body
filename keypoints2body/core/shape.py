from __future__ import annotations

from typing import Optional

import torch

from .constants import AMASS_IDX, AMASS_JOINT_MAP, AMASS_SMPL_IDX, JOINT_MAP, SMPL_IDX


@torch.no_grad()
def optimize_shape_multi_frame(
    smpl_model,
    init_betas,
    pose_init,
    j3d_world,
    joints_category="orig",
    num_iters=20,
    step_size=1e-1,
    use_lbfgs=True,
    device=torch.device("cpu"),
    frame_indices: Optional[list[int]] = None,
    joints3d_conf: Optional[torch.Tensor] = None,
    shape_prior_weight=5.0,
):
    """Optimize shared shape coefficients across multiple frames.

    Args:
        smpl_model: Loaded body model.
        init_betas: Initial betas tensor.
        pose_init: Initial poses for all frames.
        j3d_world: Target world joints sequence.
        joints_category: Joint indexing category.
        num_iters: Iteration count.
        step_size: Optimizer step size.
        use_lbfgs: Whether to use LBFGS (else Adam).
        device: Torch device.
        frame_indices: Optional frame subset.
        joints3d_conf: Optional per-joint confidence.
        shape_prior_weight: Shape prior weight.

    Returns:
        Optimized betas tensor.
    """
    t_size, j_size, _ = j3d_world.shape

    if frame_indices is None:
        frame_indices = list(range(t_size))

    betas = init_betas.clone().detach().to(device)
    betas.requires_grad = True

    if joints_category == "orig":
        smpl_index = torch.tensor(list(SMPL_IDX), device=device)
        corr_index = torch.tensor(list(SMPL_IDX), device=device)
    elif joints_category == "AMASS":
        smpl_index = torch.tensor(list(AMASS_SMPL_IDX), device=device)
        corr_index = torch.tensor(list(AMASS_IDX), device=device)
    else:
        raise ValueError(f"No such joints category: {joints_category}")

    if joints3d_conf is None:
        joints3d_conf = torch.ones(j_size, device=device)
    elif joints3d_conf.dim() == 1:
        joints3d_conf = joints3d_conf.to(device)

    shape_prior_weight = float(shape_prior_weight)

    def closure():
        optimizer.zero_grad()
        total_loss = betas.new_tensor(0.0)
        for t in frame_indices:
            pose_t = pose_init[t : t + 1].to(device)
            global_orient = pose_t[:, :3]
            body_pose = pose_t[:, 3:]
            smpl_out = smpl_model(
                global_orient=global_orient, body_pose=body_pose, betas=betas
            )
            model_joints = smpl_out.joints

            if joints_category == "orig":
                root_idx_smpl = JOINT_MAP["MidHip"]
                root_idx_target = JOINT_MAP["MidHip"]
            else:
                root_idx_smpl = AMASS_JOINT_MAP["MidHip"]
                root_idx_target = AMASS_JOINT_MAP["MidHip"]

            model_root = model_joints[:, root_idx_smpl, :]
            target_root = j3d_world[t : t + 1, root_idx_target, :].to(device)
            transl_t = target_root - model_root
            model_joints_world = model_joints + transl_t.unsqueeze(1)

            model_joints_sub = model_joints_world[:, smpl_index, :]
            target_j3d_sub = j3d_world[t : t + 1, :, :].to(device)[:, corr_index, :]

            joint3d_error = (model_joints_sub - target_j3d_sub) ** 2
            joint3d_loss = (joints3d_conf[corr_index] ** 2) * joint3d_error.sum(dim=-1)
            joint3d_loss = joint3d_loss.sum()

            shape_reg = (shape_prior_weight**2) * (betas**2).sum()
            total_loss = total_loss + joint3d_loss + shape_reg

        total_loss.backward()
        return total_loss

    if use_lbfgs:
        optimizer = torch.optim.LBFGS(
            [betas], max_iter=num_iters, lr=step_size, line_search_fn="strong_wolfe"
        )
        optimizer.step(closure)
    else:
        optimizer = torch.optim.Adam([betas], lr=step_size, betas=(0.9, 0.999))
        for _ in range(num_iters):
            _ = closure().item()

    return betas.detach()
