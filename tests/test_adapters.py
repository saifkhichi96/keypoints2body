import numpy as np
import torch

from keypoints2body.core.joints.adapters import (
    adapt_layout,
    normalize_joints_frame,
    normalize_joints_sequence,
)


def test_normalize_joints_frame_k3():
    xyz = np.zeros((22, 3), dtype=np.float32)
    j3d, conf = normalize_joints_frame(xyz)
    assert tuple(j3d.shape) == (1, 22, 3)
    assert tuple(conf.shape) == (22,)


def test_normalize_joints_frame_k4_uses_conf():
    xyzc = np.zeros((22, 4), dtype=np.float32)
    xyzc[:, 3] = 0.7
    j3d, conf = normalize_joints_frame(xyzc)
    assert tuple(j3d.shape) == (1, 22, 3)
    assert torch.allclose(conf, torch.full((22,), 0.7))


def test_normalize_joints_sequence_tk4():
    seq = np.zeros((5, 22, 4), dtype=np.float32)
    seq[:, :, 3] = 0.2
    xyz, conf = normalize_joints_sequence(seq)
    assert tuple(xyz.shape) == (5, 22, 3)
    assert tuple(conf.shape) == (5, 22)


def test_adapt_layout_manny_to_amass():
    seq = np.zeros((3, 25, 3), dtype=np.float32)
    out, layout = adapt_layout(seq, "Manny25")
    assert tuple(out.shape) == (3, 22, 3)
    assert layout == "AMASS"
