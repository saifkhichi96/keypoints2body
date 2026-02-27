from __future__ import annotations

import logging
from typing import Optional

import torch

from ...models.smpl_data import BodyModelFitResult, SMPLData
from ..constants import AMASS_IDX, AMASS_JOINT_MAP, AMASS_SMPL_IDX, JOINT_MAP, SMPL_IDX
from ..losses import body_fitting_loss_3d, camera_fitting_loss_3d
from ..prior import MaxMixturePrior

logger = logging.getLogger(__name__)


def guess_init_3d(model_joints, j3d, joints_category="SMPL24"):
    """Estimate initial camera translation from torso joints.

    Args:
        model_joints: Predicted model joints tensor.
        j3d: Target 3D joints tensor.
        joints_category: Joint indexing category.

    Returns:
        Initial translation tensor with shape ``(B,3)``.
    """
    gt_joints = ["RHip", "LHip", "RShoulder", "LShoulder"]
    gt_joints_ind = [JOINT_MAP[joint] for joint in gt_joints]

    if joints_category == "SMPL24":
        joints_ind_category = [JOINT_MAP[joint] for joint in gt_joints]
    elif joints_category == "AMASS":
        joints_ind_category = [AMASS_JOINT_MAP[joint] for joint in gt_joints]
    else:
        logger.error("Unknown joints category: %s", joints_category)
        raise ValueError(f"Unknown joints category: {joints_category}")

    sum_init_t = (j3d[:, joints_ind_category] - model_joints[:, gt_joints_ind]).sum(
        dim=1
    )
    return sum_init_t / 4.0


class CameraSpaceFitter:
    """Per-frame optimizer operating in camera coordinates."""

    def __init__(
        self,
        smpl_model,
        step_size=1e-2,
        num_iters=100,
        use_lbfgs=True,
        joints_category="SMPL24",
        device=torch.device("cpu"),
        pose_prior_num_gaussians=8,
    ):
        self.device = device
        self.step_size = step_size
        self.num_iters = num_iters
        self.use_lbfgs = use_lbfgs
        self.pose_prior = MaxMixturePrior(
            prior_folder="./data/models/",
            num_gaussians=pose_prior_num_gaussians,
            dtype=torch.float32,
        ).to(device)
        self.smpl = smpl_model
        self.joints_category = joints_category

        if joints_category == "SMPL24":
            self.smpl_index = SMPL_IDX
            self.corr_index = SMPL_IDX
        elif joints_category == "AMASS":
            self.smpl_index = AMASS_SMPL_IDX
            self.corr_index = AMASS_IDX
        elif joints_category == "GENERIC":
            self.smpl_index = None
            self.corr_index = None
        else:
            raise ValueError("No such joints category!")

    def fit_frame(
        self,
        init_params: SMPLData,
        j3d: torch.Tensor,
        conf_3d: Optional[torch.Tensor] = None,
        seq_ind: int = 0,
        target_model_indices: Optional[torch.Tensor] = None,
        joint_loss_weight: float = 600.0,
        pose_preserve_weight: float = 5.0,
        freeze_betas: bool = True,
        init_cam_t: Optional[torch.Tensor] = None,
    ) -> BodyModelFitResult:
        """Fit one frame in camera-space parameterization.

        Args:
            init_params: Initial SMPL parameters.
            j3d: Target joints tensor ``(1,K,3)``.
            conf_3d: Optional confidence tensor.
            seq_ind: Sequence index.
            joint_loss_weight: Joint loss weight.
            pose_preserve_weight: Temporal pose-preserve weight.
            freeze_betas: Whether shape should be frozen.
            init_cam_t: Optional initial camera translation.

        Returns:
            Optimization result object.
        """
        pose_preserve_weight = pose_preserve_weight if seq_ind > 0 else 0.0

        body_pose = init_params.body_pose.detach().clone()
        global_orient = init_params.global_orient.detach().clone()
        betas = init_params.betas.detach().clone()

        smpl_output = self.smpl(
            global_orient=global_orient, body_pose=body_pose, betas=betas
        )
        model_joints = smpl_output.joints
        if target_model_indices is not None:
            target_model_indices = target_model_indices.to(
                device=model_joints.device, dtype=torch.long
            )

        if init_cam_t is None:
            if target_model_indices is None:
                init_cam_t = guess_init_3d(
                    model_joints, j3d, self.joints_category
                ).detach()
            else:
                model_root = model_joints[:, target_model_indices[0], :]
                target_root = j3d[:, 0, :]
                init_cam_t = (target_root - model_root).detach()
        else:
            init_cam_t = init_cam_t.detach().clone()
        camera_translation = init_cam_t.clone()

        preserve_pose = body_pose.detach().clone()
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]

        if self.use_lbfgs:
            camera_optimizer = torch.optim.LBFGS(
                camera_opt_params,
                max_iter=self.num_iters,
                lr=self.step_size,
                line_search_fn="strong_wolfe",
            )

            def closure():
                camera_optimizer.zero_grad()
                smpl_output = self.smpl(
                    global_orient=global_orient, body_pose=body_pose, betas=betas
                )
                model_joints = smpl_output.joints
                if target_model_indices is None:
                    loss = camera_fitting_loss_3d(
                        model_joints[:, self.smpl_index],
                        camera_translation,
                        init_cam_t,
                        j3d[:, self.corr_index],
                        self.joints_category,
                    )
                else:
                    loss = (
                        (
                            model_joints[:, target_model_indices]
                            + camera_translation
                            - j3d
                        )
                        ** 2
                    ).sum()
                    loss = loss + ((camera_translation - init_cam_t) ** 2).sum() * (
                        100.0**2
                    )
                loss.backward()
                return loss

            camera_optimizer.step(closure)
        else:
            camera_optimizer = torch.optim.Adam(
                camera_opt_params, lr=self.step_size, betas=(0.9, 0.999)
            )
            for _ in range(self.num_iters):
                smpl_output = self.smpl(
                    global_orient=global_orient, body_pose=body_pose, betas=betas
                )
                model_joints = smpl_output.joints
                if target_model_indices is None:
                    loss = camera_fitting_loss_3d(
                        model_joints[:, self.smpl_index],
                        camera_translation,
                        init_cam_t,
                        j3d[:, self.corr_index],
                        self.joints_category,
                    )
                else:
                    loss = (
                        (
                            model_joints[:, target_model_indices]
                            + camera_translation
                            - j3d
                        )
                        ** 2
                    ).sum()
                    loss = loss + ((camera_translation - init_cam_t) ** 2).sum() * (
                        100.0**2
                    )
                camera_optimizer.zero_grad()
                loss.backward()
                camera_optimizer.step()

        body_pose.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        if seq_ind == 0 or not freeze_betas:
            betas.requires_grad = True
            body_opt_params = [body_pose, betas, global_orient, camera_translation]
        else:
            betas.requires_grad = False
            body_opt_params = [body_pose, global_orient, camera_translation]

        if conf_3d is None:
            conf_3d = torch.ones(j3d.shape[1], device=j3d.device)

        if self.use_lbfgs:
            body_optimizer = torch.optim.LBFGS(
                body_opt_params,
                max_iter=self.num_iters,
                lr=self.step_size,
                line_search_fn="strong_wolfe",
            )

            def closure():
                body_optimizer.zero_grad()
                smpl_output = self.smpl(
                    global_orient=global_orient, body_pose=body_pose, betas=betas
                )
                model_joints = smpl_output.joints
                if target_model_indices is None:
                    model_joints_fit = model_joints[:, self.smpl_index]
                    target_j3d = j3d[:, self.corr_index]
                    conf_fit = conf_3d
                else:
                    model_joints_fit = model_joints[:, target_model_indices]
                    target_j3d = j3d
                    conf_fit = conf_3d
                loss = body_fitting_loss_3d(
                    body_pose,
                    preserve_pose,
                    betas,
                    model_joints_fit,
                    target_j3d,
                    self.pose_prior,
                    joints3d_conf=conf_fit,
                    camera_translation=camera_translation,
                    joint_loss_weight=joint_loss_weight,
                    pose_preserve_weight=pose_preserve_weight,
                )
                loss.backward()
                return loss

            body_optimizer.step(closure)
        else:
            body_optimizer = torch.optim.Adam(
                body_opt_params, lr=self.step_size, betas=(0.9, 0.999)
            )
            for _ in range(self.num_iters):
                smpl_output = self.smpl(
                    global_orient=global_orient, body_pose=body_pose, betas=betas
                )
                model_joints = smpl_output.joints
                if target_model_indices is None:
                    model_joints_fit = model_joints[:, self.smpl_index]
                    target_j3d = j3d[:, self.corr_index]
                    conf_fit = conf_3d
                else:
                    model_joints_fit = model_joints[:, target_model_indices]
                    target_j3d = j3d
                    conf_fit = conf_3d
                loss = body_fitting_loss_3d(
                    body_pose,
                    preserve_pose,
                    betas,
                    model_joints_fit,
                    target_j3d,
                    self.pose_prior,
                    joints3d_conf=conf_fit,
                    camera_translation=camera_translation,
                    joint_loss_weight=joint_loss_weight,
                    pose_preserve_weight=pose_preserve_weight,
                )
                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()

        with torch.no_grad():
            smpl_output = self.smpl(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                return_full_pose=False,
            )
            model_joints = smpl_output.joints
            if target_model_indices is None:
                model_joints_fit = model_joints[:, self.smpl_index]
                target_j3d = j3d[:, self.corr_index]
                conf_fit = conf_3d
            else:
                model_joints_fit = model_joints[:, target_model_indices]
                target_j3d = j3d
                conf_fit = conf_3d
            final_loss = body_fitting_loss_3d(
                body_pose,
                preserve_pose,
                betas,
                model_joints_fit,
                target_j3d,
                self.pose_prior,
                joints3d_conf=conf_fit,
                camera_translation=camera_translation,
                joint_loss_weight=600.0,
            )

        fitted_params = SMPLData(
            betas=betas.detach(),
            global_orient=global_orient.detach(),
            body_pose=body_pose.detach(),
            transl=camera_translation.detach(),
        )
        return BodyModelFitResult(
            params=fitted_params,
            vertices=smpl_output.vertices.detach(),
            joints=smpl_output.joints.detach(),
            loss=final_loss,
        )
