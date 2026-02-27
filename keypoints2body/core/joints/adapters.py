from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def _osim_to_smpl_coords(data: np.ndarray) -> np.ndarray:
    """Rotate OpenSim coordinates into SMPL axis convention."""
    rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=data.dtype)
    return data @ rot.T


@dataclass(frozen=True)
class JointLayoutAdapter:
    """Definition of one joint layout and conversion behavior."""

    name: str
    expected_joints: int
    out_layout: str
    mapping: Optional[list[int]] = None
    rotate_osim_to_smpl: bool = False


ADAPTERS = {
    "SMPL24": JointLayoutAdapter("SMPL24", 24, "SMPL24"),
    "AMASS": JointLayoutAdapter("AMASS", 22, "AMASS"),
    "Manny25": JointLayoutAdapter(
        "Manny25",
        25,
        "AMASS",
        [
            0,
            21,
            17,
            1,
            22,
            18,
            3,
            23,
            19,
            5,
            24,
            20,
            6,
            13,
            9,
            8,
            14,
            10,
            15,
            11,
            16,
            12,
        ],
        True,
    ),
    "Halpe26": JointLayoutAdapter("Halpe26", 26, "AMASS", list(range(22)), False),
    "SpineTrack37": JointLayoutAdapter(
        "SpineTrack37",
        37,
        "AMASS",
        [
            0,
            31,
            25,
            1,
            32,
            26,
            3,
            33,
            27,
            5,
            34,
            28,
            6,
            21,
            17,
            8,
            22,
            18,
            23,
            19,
            24,
            20,
        ],
        True,
    ),
}


def resolve_adapter(joint_count: int, layout: Optional[str]) -> JointLayoutAdapter:
    """Resolve adapter by explicit layout or inferred joint count."""
    if layout is not None:
        if layout not in ADAPTERS:
            raise ValueError(f"Unsupported layout: {layout}")
        ad = ADAPTERS[layout]
        if joint_count != ad.expected_joints:
            raise ValueError(
                f"Layout {layout} expects {ad.expected_joints} joints, got {joint_count}"
            )
        return ad

    by_count = {ad.expected_joints: ad for ad in ADAPTERS.values()}
    if joint_count not in by_count:
        raise ValueError(f"Unsupported number of joints: {joint_count}")
    return by_count[joint_count]


def _apply_adapter_points(joints_seq: np.ndarray, ad: JointLayoutAdapter) -> np.ndarray:
    """Apply mapping/rotation rules to point coordinates."""
    out = joints_seq
    if ad.mapping is not None:
        out = out[:, ad.mapping, :]
    if ad.rotate_osim_to_smpl:
        out = _osim_to_smpl_coords(out)
    return out


def _apply_adapter_conf(conf_seq: np.ndarray, ad: JointLayoutAdapter) -> np.ndarray:
    """Apply mapping rules to confidence arrays."""
    out = conf_seq
    if ad.mapping is not None:
        out = out[:, ad.mapping]
    return out


def normalize_joints_frame(
    joints: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize frame joints to canonical tensors.

    Args:
        joints: Input frame joints with shape ``(K,3)`` or ``(K,4)``.

    Returns:
        Tuple of ``(j3d, conf)`` with shapes ``(1,K,3)`` and ``(K,)``.
    """
    if isinstance(joints, np.ndarray):
        jt = torch.as_tensor(joints, dtype=torch.float32)
    elif isinstance(joints, torch.Tensor):
        jt = joints.float()
    else:
        raise ValueError("joints must be numpy array or torch tensor")

    if jt.ndim != 2 or jt.shape[1] not in (3, 4):
        raise ValueError(f"Expected joints shape (K,3) or (K,4), got {tuple(jt.shape)}")

    if jt.shape[1] == 4:
        conf = jt[:, 3].clone()
        xyz = jt[:, :3]
    else:
        xyz = jt
        conf = torch.ones(jt.shape[0], dtype=jt.dtype, device=jt.device)

    return xyz.unsqueeze(0), conf


def normalize_joints_sequence(
    joints_seq: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize sequence joints to canonical tensors.

    Args:
        joints_seq: Input sequence joints with shape ``(T,K,3)`` or ``(T,K,4)``.

    Returns:
        Tuple of ``(j3d, conf)`` with shapes ``(T,K,3)`` and ``(T,K)``.
    """
    if isinstance(joints_seq, np.ndarray):
        jt = torch.as_tensor(joints_seq, dtype=torch.float32)
    elif isinstance(joints_seq, torch.Tensor):
        jt = joints_seq.float()
    else:
        raise ValueError("joints_seq must be numpy array or torch tensor")

    if jt.ndim != 3 or jt.shape[2] not in (3, 4):
        raise ValueError(
            f"Expected joints_seq shape (T,K,3) or (T,K,4), got {tuple(jt.shape)}"
        )

    if jt.shape[2] == 4:
        conf = jt[:, :, 3]
        xyz = jt[:, :, :3]
    else:
        xyz = jt
        conf = torch.ones((jt.shape[0], jt.shape[1]), dtype=jt.dtype, device=jt.device)

    return xyz, conf


def adapt_layout(
    joints_seq: np.ndarray, layout: Optional[str]
) -> tuple[np.ndarray, str]:
    """Adapt sequence coordinates to canonical layout."""
    ad = resolve_adapter(joints_seq.shape[1], layout)
    return _apply_adapter_points(joints_seq, ad), ad.out_layout


def adapt_layout_and_conf(
    joints_seq: np.ndarray,
    conf_seq: np.ndarray,
    layout: Optional[str],
) -> tuple[np.ndarray, np.ndarray, str]:
    """Adapt both coordinates and confidence to canonical layout."""
    ad = resolve_adapter(joints_seq.shape[1], layout)
    return (
        _apply_adapter_points(joints_seq, ad),
        _apply_adapter_conf(conf_seq, ad),
        ad.out_layout,
    )
