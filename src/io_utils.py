import io
import warnings
import zipfile
from pathlib import Path

import numpy as np
import torch

from models.smpl_data import SMPLData


def _osim_to_smpl_coords(data: np.ndarray) -> np.ndarray:
    """Convert coordinates from OpenSim world space to SMPL world space.

    Axis conventions:

    OpenSim:
        X: forward
        Y: up
        Z: left

    SMPL:
        X: right
        Y: up
        Z: forward

    This corresponds to a +90° rotation around the Y axis.

    Args:
        data: Array of shape (..., 3) containing 3D points or vectors in OpenSim coordinates.

    Returns:
        Array of the same shape (..., 3) in SMPL coordinates.
    """
    R_osim_to_smpl = np.array(
        [
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
        ],
        dtype=data.dtype,
    )

    return data @ R_osim_to_smpl.T


def _manny2amass(joints: np.ndarray) -> np.ndarray:
    """Map Manny joint layout to AMASS/SMPL joint layout and convert coordinates.

    Args:
        joints: Joint positions of shape (T, J, 3), where J is expected to be 25.

    Returns:
        Joint positions of shape (T, 22, 3) in AMASS/SMPL layout and SMPL coordinates.
    """
    mapping = [
        0,  # MidHip
        21,  # RHip
        17,  # LHip
        1,  # spine1
        22,  # RKnee
        18,  # LKnee
        3,  # spine2
        23,  # RAnkle
        19,  # LAnkle
        5,  # spine3
        24,  # RFoot
        20,  # LFoot
        6,  # Neck
        13,  # Rcollar
        9,  # LCollar
        8,  # Head
        14,  # RShoulder
        10,  # LShoulder
        15,  # RElbow
        11,  # LElbow
        16,  # RWrist
        12,  # LWrist
    ]
    mapped_joints = joints[:, mapping, :]
    return _osim_to_smpl_coords(mapped_joints)


def _halpe2amass(joints: np.ndarray) -> np.ndarray:
    """Map Halpe26 joint layout to AMASS/SMPL joint layout.

    Note:
        This is a placeholder implementation that truncates to the first 22 joints.

    Args:
        joints: Joint positions of shape (T, J, 3), where J is expected to be 26.

    Returns:
        Joint positions of shape (T, 22, 3) in AMASS/SMPL layout.
    """
    # TOOD: Implement correct mapping
    return joints[:, range(22), :]


def _spinetrack2amass(joints: np.ndarray) -> np.ndarray:
    """Map SpineTrack joint layout to AMASS/SMPL joint layout and convert coordinates.

    Args:
        joints: Joint positions of shape (T, J, 3), where J is expected to be 37.

    Returns:
        Joint positions of shape (T, 22, 3) in AMASS/SMPL layout and SMPL coordinates.
    """
    # MidHip, LHip, RHip, spine1, LKnee, RKnee, spine2,
    # LAnkle, RAnkle, spine3, LFoot, RFoot, Neck, LCollar,
    # Rcollar, Head, LShoulder, RShoulder, LElbow, RElbow,
    # LWrist, RWrist
    # mapping = [
    #     19,  # MidHip
    #     11,  # LHip
    #     12,  # RHip
    #     26,  # spine1
    #     13,  # LKnee
    #     14,  # RKnee
    #     28,  # spine2
    #     15,  # LAnkle
    #     16,  # RAnkle
    #     30,  # spine3
    #     20,  # LFoot
    #     21,  # RFoot
    #     18,  # Neck
    #     33,  # LCollar
    #     34,  # Rcollar
    #     36,  # Head
    #     5,  # LShoulder
    #     6,  # RShoulder
    #     7,  # LElbow
    #     8,  # RElbow
    #     9,  # LWrist
    #     10,  # RWrist
    # ]
    mapping = [
        0,  # MidHip
        31,  # LHip
        25,  # RHip
        1,  # spine1
        32,  # LKnee
        26,  # RKnee
        3,  # spine2
        33,  # LAnkle
        27,  # RAnkle
        5,  # spine3
        34,  # LFoot
        28,  # RFoot
        6,  # Neck
        21,  # LCollar
        17,  # Rcollar
        8,  # Head
        22,  # LShoulder
        18,  # RShoulder
        23,  # LElbow
        19,  # RElbow
        24,  # LWrist
        20,  # RWrist
    ]
    mapped_joints = joints[:, mapping, :]
    return _osim_to_smpl_coords(mapped_joints)


def load_motion_data(path: Path) -> tuple[np.ndarray, str, int]:
    """Load 3D joint motion data from disk and normalize to an AMASS/SMPL-style layout.

    Supported file types are NumPy .npy arrays and .csv files with a 5-line header and
    XYZ triplets starting at the third column.

    The function attempts to detect the input skeleton layout by joint count and
    converts supported formats to AMASS/SMPL (22 joints) when needed.

    Args:
        path: Path to the input motion file.

    Returns:
        A tuple of (joints, format, J) where:
            - joints: Joint positions as a NumPy array of shape (T, J, 3).
            - format: Detected/normalized format label (e.g., "AMASS", "orig").
            - J: Number of joints after any conversion.

    Raises:
        ValueError: If the file extension is unsupported or the joint count is unsupported.
    """
    ext = path.suffix.lower()
    if ext == ".npy":
        joints = np.load(path)
    elif ext == ".csv":
        import csv

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
            joints = data["joints"][:, :22]
        else:
            raise ValueError(
                f"Unsupported .npz format: expected 'joints' or 'pose' array, found {list(data.keys())}"
            )
    else:
        raise ValueError(f"Unsupported 3D joints file format: {ext}")

    # Detect skeleton layout and map to AMASS/SMPL as before (with original bugs kept)
    J = joints.shape[1]
    if J == 22:
        format = "AMASS"
    elif J == 24:
        format = "orig"
    elif J == 25:
        warnings.warn("Input has 25 joints; converting from Manny to AMASS.")
        joints = _manny2amass(joints)
        format = "AMASS"
        J = 22
    elif J == 26:
        warnings.warn("Input has 26 joints; converting from Halpe26 to AMASS.")
        joints = _halpe2amass(joints)
        format = "AMASS"
        J = 22
    elif J == 37:
        warnings.warn("Input has 37 joints; converting from SpineTrack to AMASS.")
        joints = _spinetrack2amass(joints)
        format = "AMASS"
        J = 22
    else:
        raise ValueError(f"Unsupported number of joints: {J}")

    return joints, format, J


def write_smplx_zip(
    output_dir: Path,
    poses: np.ndarray,
    betas: np.ndarray,
    transl: np.ndarray,
    zip_name: str = "smpl_params.zip",
    person_idx: int = 0,
) -> Path:
    """
    Convert SMPL outputs to the SMPL-X zip format used by the renderer.

    Notes:
    - SMPL pose has 24 joints (72 params). SMPL-X body_pose expects 21 joints.
      We drop the last two SMPL joints (LHand, RHand) and leave hand poses zero.
    """
    output_dir = output_dir.expanduser()
    zip_path = output_dir / zip_name

    if poses.ndim != 2 or poses.shape[1] != 72:
        raise ValueError(f"Expected poses shape (T,72); got {poses.shape}")

    T = poses.shape[0]
    if betas.ndim == 1:
        betas = np.repeat(betas[None, :], T, axis=0)
    if betas.shape[0] != T:
        raise ValueError(f"Expected betas shape (T,10); got {betas.shape}")
    if transl.shape[0] != T:
        raise ValueError(f"Expected transl shape (T,3); got {transl.shape}")

    expression = np.zeros((10,), dtype=np.float32)
    left_hand_pose = np.zeros((15 * 3,), dtype=np.float32)
    right_hand_pose = np.zeros((15 * 3,), dtype=np.float32)
    jaw_pose = np.zeros((3,), dtype=np.float32)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for t in range(T):
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
    """Write SMPL-X zip directly from a SMPLData object."""

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
