from __future__ import annotations

from typing import Optional

import torch

from ...models.smpl_data import BodyModelFitResult, SMPLData
from ..constants import AMASS_IDX, AMASS_JOINT_MAP, AMASS_SMPL_IDX, JOINT_MAP, SMPL_IDX
from ..losses import body_fitting_loss_3d
from ..prior import MaxMixturePrior


def guess_init_transl_from_root(
    smpl_model,
    pose_aa,
    betas,
    j3d_world_frame,
    joints_category="SMPL24",
):
    """Estimate world translation by aligning root joints.

    Args:
        smpl_model: Loaded SMPL model.
        pose_aa: Axis-angle pose tensor.
        betas: Shape tensor.
        j3d_world_frame: Target world-space joints.
        joints_category: Joint indexing category.

    Returns:
        Translation tensor with shape ``(B,3)``.
    """
    global_orient = pose_aa[:, :3]
    body_pose = pose_aa[:, 3:]
    smpl_output = smpl_model(
        global_orient=global_orient, body_pose=body_pose, betas=betas
    )
    model_joints = smpl_output.joints

    if joints_category == "SMPL24":
        root_idx_smpl = JOINT_MAP["MidHip"]
        root_idx_target = JOINT_MAP["MidHip"]
    elif joints_category == "AMASS":
        root_idx_smpl = AMASS_JOINT_MAP["MidHip"]
        root_idx_target = AMASS_JOINT_MAP["MidHip"]
    else:
        raise ValueError(f"Unknown joints category: {joints_category}")

    model_root = model_joints[:, root_idx_smpl, :]
    target_root = j3d_world_frame[:, root_idx_target, :]
    return (target_root - model_root).detach()


class WorldSpaceFitter:
    """Per-frame optimizer operating in world coordinates."""

    def __init__(
        self,
        smpl_model,
        step_size=1e-2,
        num_iters_first=30,
        num_iters_followup=10,
        use_lbfgs=True,
        joints_category="SMPL24",
        device=torch.device("cpu"),
        pose_prior_num_gaussians=8,
    ):
        self.smpl = smpl_model
        self.step_size = step_size
        self.num_iters_first = num_iters_first
        self.num_iters_followup = num_iters_followup
        self.use_lbfgs = use_lbfgs
        self.device = device
        self.joints_category = joints_category

        if joints_category == "SMPL24":
            self.smpl_index = torch.tensor(list(SMPL_IDX), device=device)
            self.corr_index = torch.tensor(list(SMPL_IDX), device=device)
        elif joints_category == "AMASS":
            self.smpl_index = torch.tensor(list(AMASS_SMPL_IDX), device=device)
            self.corr_index = torch.tensor(list(AMASS_IDX), device=device)
        else:
            raise ValueError("No such joints category!")

        self.pose_prior = MaxMixturePrior(
            prior_folder="./data/models/",
            num_gaussians=pose_prior_num_gaussians,
            dtype=torch.float32,
        ).to(device)

    def fit_frame(
        self,
        init_params: SMPLData,
        j3d: torch.Tensor,
        conf_3d: Optional[torch.Tensor] = None,
        seq_ind: int = 0,
        joint_loss_weight: float = 600.0,
        pose_preserve_weight: float = 5.0,
        freeze_betas: bool = False,
    ) -> BodyModelFitResult:
        """Fit one frame in world-space parameterization.

        Args:
            init_params: Initial SMPL parameters (must include transl).
            j3d: Target joints tensor ``(1,K,3)``.
            conf_3d: Optional confidence tensor.
            seq_ind: Sequence index.
            joint_loss_weight: Joint loss weight.
            pose_preserve_weight: Temporal pose-preserve weight.
            freeze_betas: Whether shape should be frozen.

        Returns:
            Optimization result object.
        """
        if init_params.transl is None:
            raise ValueError("init_params.transl must be provided")

        device = self.device
        global_orient = init_params.global_orient.clone().detach().to(device)
        body_pose = init_params.body_pose.clone().detach().to(device)
        betas = init_params.betas.clone().detach().to(device)
        transl = init_params.transl.clone().detach().to(device)

        global_orient.requires_grad_(True)
        body_pose.requires_grad_(True)
        transl.requires_grad_(True)
        preserve_pose = body_pose.clone().detach()

        if conf_3d is None:
            conf_3d = torch.ones(j3d.shape[1], device=device)
        elif conf_3d.dim() == 2:
            conf_3d = conf_3d[0]
        j3d = j3d.to(device)

        betas.requires_grad = not freeze_betas

        def compute_loss():
            smpl_out = self.smpl(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                transl=transl,
            )
            model_joints_world = smpl_out.joints
            model_joints_sub = model_joints_world[:, self.smpl_index, :]
            target_j3d_sub = j3d[:, self.corr_index, :]
            return body_fitting_loss_3d(
                body_pose=body_pose,
                preserve_pose=preserve_pose,
                betas=betas,
                model_joints=model_joints_sub,
                j3d=target_j3d_sub,
                pose_prior=self.pose_prior,
                joints3d_conf=conf_3d[self.corr_index],
                joint_loss_weight=joint_loss_weight,
                pose_preserve_weight=pose_preserve_weight if seq_ind > 0 else 0.0,
            )

        num_iters = self.num_iters_first if seq_ind == 0 else self.num_iters_followup
        params = [global_orient, body_pose, transl]

        if self.use_lbfgs:
            optimizer = torch.optim.LBFGS(
                params,
                max_iter=num_iters,
                lr=self.step_size,
                line_search_fn="strong_wolfe",
            )

            def closure():
                optimizer.zero_grad()
                loss = compute_loss()
                loss.backward()
                return loss

            optimizer.step(closure)
            with torch.no_grad():
                final_loss = compute_loss()
        else:
            optimizer = torch.optim.Adam(params, lr=self.step_size, betas=(0.9, 0.999))
            final_loss = None
            for _ in range(num_iters):
                optimizer.zero_grad()
                loss = compute_loss()
                loss.backward()
                optimizer.step()
                final_loss = loss.detach()

        with torch.no_grad():
            smpl_out = self.smpl(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                transl=transl,
                return_full_pose=False,
            )
            fitted_params = SMPLData(
                betas=betas.detach(),
                global_orient=global_orient.detach(),
                body_pose=body_pose.detach(),
                transl=transl.detach(),
            )

        return BodyModelFitResult(
            params=fitted_params,
            vertices=smpl_out.vertices.detach(),
            joints=smpl_out.joints.detach(),
            loss=final_loss,
        )
