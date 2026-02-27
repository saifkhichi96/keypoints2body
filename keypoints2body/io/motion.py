from __future__ import annotations

import csv
import io
import warnings
import zipfile
from pathlib import Path

import numpy as np
import torch

from ..core.joints.adapters import adapt_layout
from ..models.smpl_data import SMPLData


def load_motion_data(
    path: Path, layout: str | None = None
) -> tuple[np.ndarray, str, int]:
    """Load 3D joints from disk and normalize layout.

    Args:
        path: Input file path.
        layout: Optional explicit layout name.

    Returns:
        Tuple ``(joints, layout, num_joints)``.
    """
    ext = path.suffix.lower()
    if ext == ".npy":
        joints = np.load(path)
    elif ext == ".csv":
        with open(path, "r") as f:
            reader = csv.reader(f)
            for _ in range(5):
                next(reader)
            rows = [[float(v) for v in row[2:]] for row in reader]
        joints = [
            [[x, y, z] for x, y, z in zip(row[0::3], row[1::3], row[2::3])]
            for row in rows
        ]
        joints = np.array(joints)
    elif ext == ".npz":
        data = np.load(path)
        if "joints" in data:
            joints = data["joints"]
        else:
            raise ValueError(f"Unsupported .npz format: found keys {list(data.keys())}")
    else:
        raise ValueError(f"Unsupported 3D joints file format: {ext}")

    original_j = joints.shape[1]
    joints, out_layout = adapt_layout(joints, layout)
    if original_j != joints.shape[1]:
        warnings.warn(
            f"Converted input joints from {original_j} to {joints.shape[1]} for layout {out_layout}."
        )
    return joints, out_layout, joints.shape[1]


def write_smplx_zip(
    output_dir: Path,
    poses: np.ndarray,
    betas: np.ndarray,
    transl: np.ndarray,
    zip_name: str = "smpl_params.zip",
    person_idx: int = 0,
) -> Path:
    """Write a sequence of SMPL parameters to renderer-compatible SMPL-X zip."""
    output_dir = output_dir.expanduser()
    zip_path = output_dir / zip_name

    if poses.ndim != 2 or poses.shape[1] != 72:
        raise ValueError(f"Expected poses shape (T,72); got {poses.shape}")

    t_size = poses.shape[0]
    if betas.ndim == 1:
        betas = np.repeat(betas[None, :], t_size, axis=0)
    if betas.shape[0] != t_size:
        raise ValueError(f"Expected betas shape (T,10); got {betas.shape}")
    if transl.shape[0] != t_size:
        raise ValueError(f"Expected transl shape (T,3); got {transl.shape}")

    expression = np.zeros((10,), dtype=np.float32)
    left_hand_pose = np.zeros((15 * 3,), dtype=np.float32)
    right_hand_pose = np.zeros((15 * 3,), dtype=np.float32)
    jaw_pose = np.zeros((3,), dtype=np.float32)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for t in range(t_size):
            global_orient = poses[t, :3].astype(np.float32)
            body_pose = poses[t, 3 : 3 + 21 * 3].astype(np.float32)
            params = {
                "betas": betas[t].astype(np.float32),
                "expression": expression,
                "global_orient": global_orient,
                "body_pose": body_pose,
                "left_hand_pose": left_hand_pose,
                "right_hand_pose": right_hand_pose,
                "jaw_pose": jaw_pose,
                "transl": transl[t].astype(np.float32),
            }
            buf = io.BytesIO()
            np.savez(buf, **params)
            name = f"frame_{t:06d}/person_{person_idx:02d}.npz"
            zf.writestr(name, buf.getvalue())

    return zip_path


def write_smplx_zip_from_smpl_data(
    output_dir: Path,
    smpl_data: SMPLData,
    zip_name: str = "smpl_params.zip",
    person_idx: int = 0,
) -> Path:
    """Export one ``SMPLData`` sequence to SMPL-X zip format."""
    pose = smpl_data.pose
    betas = smpl_data.betas
    transl = smpl_data.transl

    if transl is None:
        raise ValueError("smpl_data.transl is required to export SMPL-X zip")

    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()
    if isinstance(betas, torch.Tensor):
        betas = betas.detach().cpu().numpy()
    if isinstance(transl, torch.Tensor):
        transl = transl.detach().cpu().numpy()

    return write_smplx_zip(
        output_dir=output_dir,
        poses=pose,
        betas=betas,
        transl=transl,
        zip_name=zip_name,
        person_idx=person_idx,
    )
