from __future__ import annotations

import inspect
from typing import Optional

import torch

from ...models.smpl_data import BodyModelFitResult, FLAMEData, MANOData
from ..losses import generic_keypoint_loss_3d


def _forward_filtered(model, kwargs: dict[str, torch.Tensor]):
    params = inspect.signature(model.forward).parameters
    filtered = {k: v for k, v in kwargs.items() if k in params and v is not None}
    return model(**filtered)


class MANOFitter:
    """Optimization fitter for MANO hand model."""

    def __init__(
        self,
        model,
        coordinate_mode: str,
        step_size=1e-2,
        num_iters_first=30,
        num_iters_followup=10,
        use_lbfgs=True,
        device=torch.device("cpu"),
    ):
        self.model = model
        self.coordinate_mode = coordinate_mode
        self.step_size = step_size
        self.num_iters_first = num_iters_first
        self.num_iters_followup = num_iters_followup
        self.use_lbfgs = use_lbfgs
        self.device = device

    def fit_frame(
        self,
        init_params: MANOData,
        j3d: torch.Tensor,
        conf_3d: Optional[torch.Tensor] = None,
        seq_ind: int = 0,
        joint_loss_weight: float = 600.0,
        pose_preserve_weight: float = 5.0,
        freeze_betas: bool = False,
    ) -> BodyModelFitResult:
        if init_params.hand_pose is None:
            raise ValueError("MANOData.hand_pose is required")

        device = self.device
        j3d = j3d.to(device)
        if conf_3d is None:
            conf_3d = torch.ones(j3d.shape[1], device=device)
        elif conf_3d.dim() == 2:
            conf_3d = conf_3d[0]

        global_orient = init_params.global_orient.clone().detach().to(device)
        hand_pose = init_params.hand_pose.clone().detach().to(device)
        betas = init_params.betas.clone().detach().to(device)
        transl = init_params.transl
        if transl is None:
            transl = j3d[:, 0, :].clone().detach()
        transl = transl.clone().detach().to(device)

        preserve_pose = hand_pose.clone().detach()

        global_orient.requires_grad_(True)
        hand_pose.requires_grad_(True)
        transl.requires_grad_(True)
        betas.requires_grad_(not freeze_betas)

        num_iters = self.num_iters_first if seq_ind == 0 else self.num_iters_followup
        params = [global_orient, hand_pose, transl]
        if betas.requires_grad:
            params.append(betas)

        def compute_loss():
            out = _forward_filtered(
                self.model,
                {
                    "global_orient": global_orient,
                    "hand_pose": hand_pose,
                    "betas": betas,
                    "transl": transl if self.coordinate_mode == "world" else None,
                },
            )
            model_joints = out.joints[:, : j3d.shape[1], :]
            if self.coordinate_mode == "camera":
                model_joints = model_joints + transl
            loss = generic_keypoint_loss_3d(
                model_joints=model_joints,
                j3d=j3d,
                joints3d_conf=conf_3d[: model_joints.shape[1]],
                joint_loss_weight=joint_loss_weight,
            )
            loss = loss + 1e-2 * (hand_pose**2).sum() + 5.0 * (betas**2).sum()
            if seq_ind > 0:
                loss = loss + (pose_preserve_weight**2) * (
                    (hand_pose - preserve_pose) ** 2
                ).sum()
            return loss

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
            out = _forward_filtered(
                self.model,
                {
                    "global_orient": global_orient,
                    "hand_pose": hand_pose,
                    "betas": betas,
                    "transl": transl if self.coordinate_mode == "world" else None,
                },
            )

        fitted = MANOData(
            betas=betas.detach(),
            global_orient=global_orient.detach(),
            body_pose=torch.zeros(
                (global_orient.shape[0], 0), device=device, dtype=global_orient.dtype
            ),
            transl=transl.detach(),
            hand_pose=hand_pose.detach(),
        )
        return BodyModelFitResult(
            params=fitted,
            vertices=out.vertices.detach(),
            joints=out.joints.detach(),
            loss=final_loss,
        )


class FLAMEFitter:
    """Optimization fitter for FLAME face model."""

    def __init__(
        self,
        model,
        coordinate_mode: str,
        step_size=1e-2,
        num_iters_first=30,
        num_iters_followup=10,
        use_lbfgs=True,
        device=torch.device("cpu"),
    ):
        self.model = model
        self.coordinate_mode = coordinate_mode
        self.step_size = step_size
        self.num_iters_first = num_iters_first
        self.num_iters_followup = num_iters_followup
        self.use_lbfgs = use_lbfgs
        self.device = device

    def fit_frame(
        self,
        init_params: FLAMEData,
        j3d: torch.Tensor,
        conf_3d: Optional[torch.Tensor] = None,
        seq_ind: int = 0,
        joint_loss_weight: float = 600.0,
        pose_preserve_weight: float = 5.0,
        freeze_betas: bool = False,
    ) -> BodyModelFitResult:
        device = self.device
        j3d = j3d.to(device)
        if conf_3d is None:
            conf_3d = torch.ones(j3d.shape[1], device=device)
        elif conf_3d.dim() == 2:
            conf_3d = conf_3d[0]

        def _clone_opt(x: Optional[torch.Tensor], dim: int) -> torch.Tensor:
            if x is None:
                return torch.zeros((j3d.shape[0], dim), device=device)
            return x.clone().detach().to(device)

        global_orient = init_params.global_orient.clone().detach().to(device)
        betas = init_params.betas.clone().detach().to(device)
        expr_dim = (
            int(init_params.expression.shape[1])
            if init_params.expression is not None
            else int(getattr(self.model, "num_expression_coeffs", 10))
        )
        expression = _clone_opt(init_params.expression, expr_dim)
        jaw_pose = _clone_opt(init_params.jaw_pose, 3)
        neck_pose = _clone_opt(init_params.neck_pose, 3)
        leye_pose = _clone_opt(init_params.leye_pose, 3)
        reye_pose = _clone_opt(init_params.reye_pose, 3)
        transl = init_params.transl
        if transl is None:
            transl = j3d[:, 0, :].clone().detach()
        transl = transl.clone().detach().to(device)

        preserve_jaw = jaw_pose.clone().detach()
        preserve_expr = expression.clone().detach()

        global_orient.requires_grad_(True)
        transl.requires_grad_(True)
        jaw_pose.requires_grad_(True)
        expression.requires_grad_(True)
        neck_pose.requires_grad_(True)
        leye_pose.requires_grad_(True)
        reye_pose.requires_grad_(True)
        betas.requires_grad_(not freeze_betas)

        params = [
            global_orient,
            transl,
            jaw_pose,
            expression,
            neck_pose,
            leye_pose,
            reye_pose,
        ]
        if betas.requires_grad:
            params.append(betas)
        num_iters = self.num_iters_first if seq_ind == 0 else self.num_iters_followup

        def compute_loss():
            out = _forward_filtered(
                self.model,
                {
                    "global_orient": global_orient,
                    "betas": betas,
                    "expression": expression,
                    "jaw_pose": jaw_pose,
                    "neck_pose": neck_pose,
                    "leye_pose": leye_pose,
                    "reye_pose": reye_pose,
                    "transl": transl if self.coordinate_mode == "world" else None,
                },
            )
            model_joints = out.joints[:, : j3d.shape[1], :]
            if self.coordinate_mode == "camera":
                model_joints = model_joints + transl
            loss = generic_keypoint_loss_3d(
                model_joints=model_joints,
                j3d=j3d,
                joints3d_conf=conf_3d[: model_joints.shape[1]],
                joint_loss_weight=joint_loss_weight,
            )
            loss = (
                loss
                + 1e-2 * (jaw_pose**2).sum()
                + 1e-3 * (expression**2).sum()
                + 5.0 * (betas**2).sum()
            )
            if seq_ind > 0:
                loss = loss + (pose_preserve_weight**2) * (
                    ((jaw_pose - preserve_jaw) ** 2).sum()
                    + 0.2 * ((expression - preserve_expr) ** 2).sum()
                )
            return loss

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
            out = _forward_filtered(
                self.model,
                {
                    "global_orient": global_orient,
                    "betas": betas,
                    "expression": expression,
                    "jaw_pose": jaw_pose,
                    "neck_pose": neck_pose,
                    "leye_pose": leye_pose,
                    "reye_pose": reye_pose,
                    "transl": transl if self.coordinate_mode == "world" else None,
                },
            )

        fitted = FLAMEData(
            betas=betas.detach(),
            global_orient=global_orient.detach(),
            body_pose=torch.zeros(
                (global_orient.shape[0], 0), device=device, dtype=global_orient.dtype
            ),
            transl=transl.detach(),
            expression=expression.detach(),
            jaw_pose=jaw_pose.detach(),
            neck_pose=neck_pose.detach(),
            leye_pose=leye_pose.detach(),
            reye_pose=reye_pose.detach(),
        )
        return BodyModelFitResult(
            params=fitted,
            vertices=out.vertices.detach(),
            joints=out.joints.detach(),
            loss=final_loss,
        )
