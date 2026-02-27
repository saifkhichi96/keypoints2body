import numpy as np
import torch

from keypoints2body.models.smpl_data import SMPLData


def test_pose_concat_and_validate():
    params = SMPLData(
        betas=torch.zeros(1, 10),
        global_orient=torch.zeros(1, 3),
        body_pose=torch.zeros(1, 69),
        transl=torch.zeros(1, 3),
    )
    params.validate()
    assert tuple(params.pose.shape) == (1, 72)


def test_to_and_detach_roundtrip():
    params = SMPLData(
        betas=torch.zeros(1, 10, requires_grad=True),
        global_orient=torch.zeros(1, 3, requires_grad=True),
        body_pose=torch.zeros(1, 69, requires_grad=True),
    )
    detached = params.detach()
    assert isinstance(detached.betas, torch.Tensor)
    assert detached.betas.requires_grad is False


def test_numpy_params_supported():
    params = SMPLData(
        betas=np.zeros((1, 10), dtype=np.float32),
        global_orient=np.zeros((1, 3), dtype=np.float32),
        body_pose=np.zeros((1, 69), dtype=np.float32),
    )
    assert params.pose.shape == (1, 72)
