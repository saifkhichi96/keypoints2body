import logging
import os
import sys
from typing import List, Optional

import torch

sys.path.append(os.path.dirname(__file__))
import config
from customloss import (
    body_fitting_loss_3d,
    camera_fitting_loss_3d,
)
from models.smpl_data import BodyModelFitResult, SMPLData
from prior import MaxMixturePrior

logger = logging.getLogger(__name__)


@torch.no_grad()
def optimize_shape_multi_frame(
    smpl_model,
    init_betas,
    pose_init,
    j3d_world,
    joints_category="SMPL24",
    num_iters=20,
    step_size=1e-1,
    use_lbfgs=True,
    device=torch.device("cpu"),
    frame_indices: Optional[List[int]] = None,
    joints3d_conf: Optional[torch.Tensor] = None,
    shape_prior_weight=5.0,
):
    """
    Multi-frame shape optimization.

    Args:
        smpl_model: SMPL model (on device)
        init_betas: (1,10) initial shape
        pose_init: (T,72) initial poses (global+body)
        j3d_world: (T,J,3) world 3D joints (dataset layout)
        joints_category: 'SMPL24' or 'AMASS'
        frame_indices: optional subset of frames to use
    """
    T, J, _ = j3d_world.shape

    if frame_indices is None:
        frame_indices = list(range(T))

    betas = init_betas.clone().detach().to(device)
    betas.requires_grad = True

    if joints_category == "SMPL24":
        smpl_index = torch.tensor(list(config.smpl_idx), device=device)
        corr_index = torch.tensor(list(config.smpl_idx), device=device)
    elif joints_category == "AMASS":
        smpl_index = torch.tensor(list(config.amass_smpl_idx), device=device)
        corr_index = torch.tensor(list(config.amass_idx), device=device)
    else:
        raise ValueError(f"No such joints category: {joints_category}")

    if joints3d_conf is None:
        joints3d_conf = torch.ones(J, device=device)
    elif joints3d_conf.dim() == 1:
        joints3d_conf = joints3d_conf.to(device)

    shape_prior_weight = float(shape_prior_weight)

    def closure():
        optimizer.zero_grad()
        total_loss = betas.new_tensor(0.0)

        for t in frame_indices:
            pose_t = pose_init[t : t + 1].to(device)  # (1,72)
            global_orient = pose_t[:, :3]
            body_pose = pose_t[:, 3:]

            smpl_out = smpl_model(
                global_orient=global_orient, body_pose=body_pose, betas=betas
            )
            model_joints = smpl_out.joints  # (1, J_smpl, 3)

            # Align root per frame by translation to remove global position
            if joints_category == "SMPL24":
                root_idx_smpl = config.JOINT_MAP["MidHip"]
                root_idx_target = config.JOINT_MAP["MidHip"]
            else:
                root_idx_smpl = config.AMASS_JOINT_MAP["MidHip"]
                root_idx_target = config.AMASS_JOINT_MAP["MidHip"]

            model_root = model_joints[:, root_idx_smpl, :]  # (1,3)
            target_root = j3d_world[t : t + 1, root_idx_target, :].to(device)  # (1,3)

            transl_t = target_root - model_root  # (1,3)
            model_joints_world = model_joints + transl_t.unsqueeze(1)  # (1,J_smpl,3)

            # Map to observed joint layout
            model_joints_sub = model_joints_world[:, smpl_index, :]
            target_j3d_sub = j3d_world[t : t + 1, :, :].to(device)[:, corr_index, :]

            joint3d_error = (model_joints_sub - target_j3d_sub) ** 2  # (1,J,3)
            joint3d_loss = (joints3d_conf[corr_index] ** 2) * joint3d_error.sum(dim=-1)
            joint3d_loss = joint3d_loss.sum()

            shape_reg = (shape_prior_weight**2) * (betas**2).sum()

            total_loss = total_loss + joint3d_loss + shape_reg

        total_loss.backward()
        return total_loss

    if use_lbfgs:
        optimizer = torch.optim.LBFGS(
            [betas],
            max_iter=num_iters,
            lr=step_size,
            line_search_fn="strong_wolfe",
        )
        optimizer.step(closure)
    else:
        optimizer = torch.optim.Adam([betas], lr=step_size, betas=(0.9, 0.999))
        for _ in range(num_iters):
            loss = closure()
            # detached value only for logging if needed
            _ = loss.item()

    return betas.detach()


@torch.no_grad()
def guess_init_3d(model_joints, j3d, joints_category="SMPL24"):
    """Initialize the camera translation via triangle similarity, by using the torso joints        .
    :param model_joints: SMPL model with pre joints
    :param j3d: 25x3 array of Kinect Joints
    :returns: 3D vector corresponding to the estimated camera translation
    """
    # get the indexed four
    gt_joints = ["RHip", "LHip", "RShoulder", "LShoulder"]
    gt_joints_ind = [config.JOINT_MAP[joint] for joint in gt_joints]

    if joints_category == "SMPL24":
        joints_ind_category = [config.JOINT_MAP[joint] for joint in gt_joints]
    elif joints_category == "AMASS":
        joints_ind_category = [config.AMASS_JOINT_MAP[joint] for joint in gt_joints]
    else:
        logger.error("Unknown joints category: %s", joints_category)
        raise ValueError(f"Unknown joints category: {joints_category}")

    sum_init_t = (j3d[:, joints_ind_category] - model_joints[:, gt_joints_ind]).sum(
        dim=1
    )
    return sum_init_t / 4.0  # init_t


# SMPLIfy 3D
class SMPLify3D:
    """Implementation of SMPLify, use 3D joints."""

    def __init__(
        self,
        smplxmodel,
        step_size=1e-2,
        batch_size=1,
        num_iters=100,
        use_lbfgs=True,
        joints_category="SMPL24",
        device=torch.device("cuda:0"),
    ):
        # Store options
        self.batch_size = batch_size
        self.device = device
        self.step_size = step_size

        self.num_iters = num_iters
        # --- choose optimizer
        self.use_lbfgs = use_lbfgs
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(
            prior_folder=config.GMM_MODEL_DIR, num_gaussians=8, dtype=torch.float32
        ).to(device)

        # reLoad SMPL-X model
        self.smpl = smplxmodel

        # select joints_category
        self.joints_category = joints_category

        if joints_category == "SMPL24":
            self.smpl_index = config.smpl_idx
            self.corr_index = config.smpl_idx
        elif joints_category == "AMASS":
            self.smpl_index = config.amass_smpl_idx
            self.corr_index = config.amass_idx
        else:
            raise ValueError("No such joints category!")

    def __call__(
        self,
        init_params: SMPLData,
        j3d: torch.Tensor,
        init_cam_t: Optional[torch.Tensor] = None,
        conf_3d=1.0,
        seq_ind=0,
        joint_loss_weight=600.0,
        pose_preserve_weight=5.0,
        freeze_betas=True,
    ) -> BodyModelFitResult:
        """Perform body fitting.

        Args:
            init_params: initial SMPL parameters
            init_cam_t: Camera translation estimate
            j3d: joints 3d aka keypoints
            conf_3d: confidence for 3d joints
            seq_ind: index of the sequence
            joint_loss_weight: weight for the 3D joint loss
            pose_preserve_weight: weight for the pose preservation loss (only applied for seq_ind > 0)
            freeze_betas: whether to freeze the betas during optimization

        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
        """
        pose_preserve_weight = pose_preserve_weight if seq_ind > 0 else 0.0

        body_pose = init_params.body_pose.detach().clone()
        global_orient = init_params.global_orient.detach().clone()
        betas = init_params.betas.detach().clone()

        # use guess 3d to get the initial
        smpl_output = self.smpl(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
        )
        model_joints = smpl_output.joints

        if init_cam_t is None:
            init_cam_t = guess_init_3d(model_joints, j3d, self.joints_category).detach()
        else:
            init_cam_t = init_cam_t.detach().clone()
        camera_translation = init_cam_t.clone()

        preserve_pose = body_pose.detach().clone()

        # -------------Step 1: Optimize camera translation and body orientation--------
        # Optimize only camera translation and body orientation
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
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=betas,
                )
                model_joints = smpl_output.joints

                loss = camera_fitting_loss_3d(
                    model_joints[:, self.smpl_index],
                    camera_translation,
                    init_cam_t,
                    j3d[:, self.corr_index],
                    self.joints_category,
                )
                loss.backward()
                return loss

            camera_optimizer.step(closure)
        else:
            camera_optimizer = torch.optim.Adam(
                camera_opt_params, lr=self.step_size, betas=(0.9, 0.999)
            )

            for i in range(self.num_iters):
                smpl_output = self.smpl(
                    global_orient=global_orient, body_pose=body_pose, betas=betas
                )
                model_joints = smpl_output.joints

                loss = camera_fitting_loss_3d(
                    model_joints[:, self.smpl_index],
                    camera_translation,
                    init_cam_t,
                    j3d[:, self.corr_index],
                    self.joints_category,
                )
                camera_optimizer.zero_grad()
                loss.backward()
                camera_optimizer.step()

        # Fix camera translation after optimizing camera
        # --------Step 2: Optimize body joints --------------------------
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        # --- if we use the sequence, fix the shape
        if seq_ind == 0 or not freeze_betas:
            betas.requires_grad = True
            body_opt_params = [body_pose, betas, global_orient, camera_translation]
        else:
            betas.requires_grad = False
            body_opt_params = [body_pose, global_orient, camera_translation]

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

                loss = body_fitting_loss_3d(
                    body_pose,
                    preserve_pose,
                    betas,
                    model_joints[:, self.smpl_index],
                    j3d[:, self.corr_index],
                    self.pose_prior,
                    joints3d_conf=conf_3d,
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

            for i in range(self.num_iters):
                smpl_output = self.smpl(
                    global_orient=global_orient, body_pose=body_pose, betas=betas
                )
                model_joints = smpl_output.joints

                loss = body_fitting_loss_3d(
                    body_pose,
                    preserve_pose,
                    betas,
                    model_joints[:, self.smpl_index],
                    j3d[:, self.corr_index],
                    self.pose_prior,
                    joints3d_conf=conf_3d,
                    camera_translation=camera_translation,
                    joint_loss_weight=joint_loss_weight,
                    pose_preserve_weight=pose_preserve_weight,
                )
                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()

        # Get final loss value
        with torch.no_grad():
            smpl_output = self.smpl(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                return_full_pose=False,
            )
            model_joints = smpl_output.joints

            final_loss = body_fitting_loss_3d(
                body_pose,
                preserve_pose,
                betas,
                model_joints[:, self.smpl_index],
                j3d[:, self.corr_index],
                self.pose_prior,
                joints3d_conf=conf_3d,
                camera_translation=camera_translation,
                joint_loss_weight=600.0,
            )

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        fitted_params = SMPLData(
            betas=betas.detach(),
            global_orient=global_orient.detach(),
            body_pose=body_pose.detach(),
            transl=camera_translation.detach(),
        )

        return BodyModelFitResult(
            params=fitted_params,
            vertices=vertices,
            joints=joints,
            loss=final_loss,
        )
