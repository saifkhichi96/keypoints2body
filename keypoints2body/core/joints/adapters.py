from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..constants import (
    AMASS_SMPL_IDX,
    SMPL_IDX,
    SMPLX_FACE_IDX_START,
    SMPLX_LEFT_HAND_IDX,
    SMPLX_RIGHT_HAND_IDX,
)


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


def normalize_frame_observations(
    joints: np.ndarray | torch.Tensor | dict,
    *,
    layout: Optional[str],
    body_model: str,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], str]:
    """Normalize one-frame observations with optional block-wise dict input.

    Returns:
        j3d: ``(1,K,3)``
        conf: ``(K,)``
        model_indices: optional ``(K,)`` indices into model joints
        out_layout: canonical layout label
    """
    if not isinstance(joints, dict):
        j3d, conf = normalize_joints_frame(joints)
        return j3d, conf, None, "AUTO"

    def _norm_block(block) -> tuple[torch.Tensor, torch.Tensor]:
        bj, bc = normalize_joints_frame(block)
        return bj[:, :, :], bc

    points_parts: list[torch.Tensor] = []
    conf_parts: list[torch.Tensor] = []
    idx_parts: list[torch.Tensor] = []

    if "body" in joints:
        body_pts, body_conf = _norm_block(joints["body"])
        body_k = body_pts.shape[1]
        if body_k == 24:
            model_idx = torch.tensor(list(SMPL_IDX), dtype=torch.long)
        elif body_k == 22:
            model_idx = torch.tensor(list(AMASS_SMPL_IDX), dtype=torch.long)
        else:
            raise ValueError("body block must have 22 or 24 joints")
        points_parts.append(body_pts)
        conf_parts.append(body_conf)
        idx_parts.append(model_idx)

    if "left_hand" in joints:
        if body_model not in {"smplh", "smplx"}:
            raise ValueError("left_hand block requires body_model='smplh' or 'smplx'")
        lh_pts, lh_conf = _norm_block(joints["left_hand"])
        if lh_pts.shape[1] != len(list(SMPLX_LEFT_HAND_IDX)):
            raise ValueError("left_hand block must have 21 joints")
        points_parts.append(lh_pts)
        conf_parts.append(lh_conf)
        idx_parts.append(torch.tensor(list(SMPLX_LEFT_HAND_IDX), dtype=torch.long))

    if "right_hand" in joints:
        if body_model not in {"smplh", "smplx"}:
            raise ValueError("right_hand block requires body_model='smplh' or 'smplx'")
        rh_pts, rh_conf = _norm_block(joints["right_hand"])
        if rh_pts.shape[1] != len(list(SMPLX_RIGHT_HAND_IDX)):
            raise ValueError("right_hand block must have 21 joints")
        points_parts.append(rh_pts)
        conf_parts.append(rh_conf)
        idx_parts.append(torch.tensor(list(SMPLX_RIGHT_HAND_IDX), dtype=torch.long))

    if "face" in joints:
        if body_model != "smplx":
            raise ValueError("face block requires body_model='smplx'")
        fc_pts, fc_conf = _norm_block(joints["face"])
        fc_k = fc_pts.shape[1]
        points_parts.append(fc_pts)
        conf_parts.append(fc_conf)
        idx_parts.append(
            torch.arange(SMPLX_FACE_IDX_START, SMPLX_FACE_IDX_START + fc_k, dtype=torch.long)
        )

    if not points_parts:
        raise ValueError("dict input must provide at least one of: body, left_hand, right_hand, face")

    j3d = torch.cat(points_parts, dim=1)
    conf = torch.cat(conf_parts, dim=0)
    model_indices = torch.cat(idx_parts, dim=0)
    return j3d, conf, model_indices, "GENERIC"


def normalize_sequence_observations(
    joints_seq: np.ndarray | torch.Tensor | dict,
    *,
    layout: Optional[str],
    body_model: str,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], str]:
    """Normalize sequence observations with optional block-wise dict input."""
    if not isinstance(joints_seq, dict):
        xyz, conf = normalize_joints_sequence(joints_seq)
        return xyz, conf, None, "AUTO"

    def _norm_block_seq(block) -> tuple[torch.Tensor, torch.Tensor]:
        bxyz, bconf = normalize_joints_sequence(block)
        return bxyz, bconf

    pts_parts: list[torch.Tensor] = []
    conf_parts: list[torch.Tensor] = []
    idx_parts: list[torch.Tensor] = []
    t_size: Optional[int] = None

    for key in ("body", "left_hand", "right_hand", "face"):
        if key not in joints_seq:
            continue
        p, c = _norm_block_seq(joints_seq[key])
        if t_size is None:
            t_size = p.shape[0]
        elif p.shape[0] != t_size:
            raise ValueError("all dict sequence blocks must share same T")

        if key == "body":
            if p.shape[1] == 24:
                idx = torch.tensor(list(SMPL_IDX), dtype=torch.long)
            elif p.shape[1] == 22:
                idx = torch.tensor(list(AMASS_SMPL_IDX), dtype=torch.long)
            else:
                raise ValueError("body block must have 22 or 24 joints")
        elif key == "left_hand":
            if body_model not in {"smplh", "smplx"}:
                raise ValueError("left_hand block requires body_model='smplh' or 'smplx'")
            if p.shape[1] != 21:
                raise ValueError("left_hand block must have 21 joints")
            idx = torch.tensor(list(SMPLX_LEFT_HAND_IDX), dtype=torch.long)
        elif key == "right_hand":
            if body_model not in {"smplh", "smplx"}:
                raise ValueError("right_hand block requires body_model='smplh' or 'smplx'")
            if p.shape[1] != 21:
                raise ValueError("right_hand block must have 21 joints")
            idx = torch.tensor(list(SMPLX_RIGHT_HAND_IDX), dtype=torch.long)
        else:
            if body_model != "smplx":
                raise ValueError("face block requires body_model='smplx'")
            idx = torch.arange(SMPLX_FACE_IDX_START, SMPLX_FACE_IDX_START + p.shape[1], dtype=torch.long)

        pts_parts.append(p)
        conf_parts.append(c)
        idx_parts.append(idx)

    if not pts_parts:
        raise ValueError("dict input must provide at least one of: body, left_hand, right_hand, face")

    xyz = torch.cat(pts_parts, dim=1)
    conf = torch.cat(conf_parts, dim=1)
    model_indices = torch.cat(idx_parts, dim=0)
    return xyz, conf, model_indices, "GENERIC"
