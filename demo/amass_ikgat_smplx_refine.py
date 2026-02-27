from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from keypoints2body import SMPLXData, optimize_params_frame
from keypoints2body.core.config import FrameOptimizeConfig

# AMASS/SMPL22-style hierarchy used by the provided IK-GAT model.
DEFAULT_PARENT_IDS = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
]


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _quat_to_rotvec(quat_xyzw: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert quaternions (J,4) to axis-angle rotvec (J,3)."""
    q = quat_xyzw.astype(np.float32)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.clip(n, eps, None)

    xyz = q[:, :3]
    w = np.clip(q[:, 3], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(np.clip(1.0 - w * w, 0.0, None))

    axis = np.zeros_like(xyz)
    valid = sin_half > eps
    axis[valid] = xyz[valid] / sin_half[valid, None]
    axis[~valid] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return axis * angle[:, None]


def _load_amass_joints(npz_path: Path, limit_frames: int = -1) -> np.ndarray:
    data = np.load(npz_path)
    if "joints" not in data:
        raise KeyError(f"Missing 'joints' key in {npz_path}")
    joints = np.asarray(data["joints"], dtype=np.float32)
    data.close()

    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected joints shape (T,K,3), got {joints.shape}")
    joints = joints[:, :22, :]
    if limit_frames > 0:
        joints = joints[:limit_frames]
    if joints.shape[0] == 0:
        raise ValueError("No frames found after filtering")
    return joints


def _make_smplx_init_from_ikgat(
    quats_xyzw: np.ndarray,
    frame_joints: np.ndarray,
    prev_params: SMPLXData | None,
) -> SMPLXData:
    """Build SMPL-X init params from IK-GAT quaternions for 22 body joints."""
    rotvec = _quat_to_rotvec(quats_xyzw)  # (22, 3)

    global_orient = rotvec[0:1]
    body_pose = np.zeros((1, 69), dtype=np.float32)

    # SMPL body_pose has 23 joints (69D). AMASS provides 21 non-root body joints.
    # Fill the first 21 and keep the last 2 as identity.
    body_pose[0, : 21 * 3] = rotvec[1:22].reshape(-1)

    if prev_params is not None:
        betas = _as_numpy(prev_params.betas).astype(np.float32)
        transl_prev = prev_params.transl
        transl = (
            _as_numpy(transl_prev).astype(np.float32)
            if transl_prev is not None
            else frame_joints[0:1]
        )
    else:
        betas = np.zeros((1, 10), dtype=np.float32)
        transl = frame_joints[0:1].astype(np.float32)

    zeros45 = np.zeros((1, 45), dtype=np.float32)
    zeros3 = np.zeros((1, 3), dtype=np.float32)
    zeros10 = np.zeros((1, 10), dtype=np.float32)

    return SMPLXData(
        betas=betas,
        global_orient=global_orient.astype(np.float32),
        body_pose=body_pose,
        transl=transl,
        left_hand_pose=zeros45,
        right_hand_pose=zeros45,
        expression=zeros10,
        jaw_pose=zeros3,
        leye_pose=zeros3,
        reye_pose=zeros3,
        metadata={},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AMASS -> IK-GAT estimate -> short SMPL-X optimization refine demo"
    )
    parser.add_argument("--amass-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("./demo_out_smplx_refined.npz"))
    parser.add_argument("--limit-frames", type=int, default=-1)
    parser.add_argument("--opt-steps", type=int, default=5)
    parser.add_argument("--model-format", type=str, default="smplx")
    parser.add_argument("--model-type", type=str, default="pos_to_rot6")
    parser.add_argument("--estimators-dir", type=Path, default=Path("./data/estimators"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-adam", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device is not None else None

    if args.opt_steps < 2 or args.opt_steps > 5:
        raise ValueError("--opt-steps should be in [2, 5] for this demo")

    joints_seq = _load_amass_joints(args.amass_file, args.limit_frames)

    est_cfg = FrameOptimizeConfig(
        estimator_type="ikgat",
        input_type="joints3d",
        coordinate_mode="world",
        ikgat_model_dir=args.estimators_dir,
        ikgat_model_format=args.model_format,
        ikgat_model_type=args.model_type,
        ikgat_parent_ids=DEFAULT_PARENT_IDS,
    )

    opt_cfg = FrameOptimizeConfig(
        estimator_type="optimization",
        input_type="joints3d",
        coordinate_mode="world",
        use_lbfgs=not args.use_adam,
        num_iters_first=args.opt_steps,
        num_iters_followup=args.opt_steps,
        joint_loss_weight=600.0,
        pose_preserve_weight=5.0,
        freeze_betas=False,
        joints_category="AMASS",
    )

    prev_est = None
    prev_opt: SMPLXData | None = None
    outputs: list[SMPLXData] = []

    iterator = tqdm(range(joints_seq.shape[0]), desc="Refining", dynamic_ncols=True)
    for t in iterator:
        frame = joints_seq[t]

        # 1) Learned estimator for pose guess (quaternions stored in metadata).
        est_res = optimize_params_frame(
            frame,
            body_model="smplx",
            joint_layout="AMASS",
            prev_params=prev_est,
            config=est_cfg,
            device=device,
        )
        prev_est = est_res.params

        quats = est_res.params.metadata.get("ikgat_quaternions")
        if quats is None:
            raise RuntimeError("IK-GAT output missing metadata['ikgat_quaternions']")

        # 2) Build SMPL-X init from estimator output (hands/face/expression identity).
        init_params = _make_smplx_init_from_ikgat(quats, frame, prev_opt)

        # 3) Short optimization refine (2-5 iterations), warm-started by previous frame.
        refined = optimize_params_frame(
            frame,
            body_model="smplx",
            joint_layout="AMASS",
            prev_params=init_params,
            config=opt_cfg,
            device=device,
        )
        prev_opt = refined.params  # warm-start across frames
        outputs.append(refined.params)

    global_orient = np.concatenate(
        [_as_numpy(p.global_orient).astype(np.float32) for p in outputs], axis=0
    )
    body_pose = np.concatenate(
        [_as_numpy(p.body_pose).astype(np.float32) for p in outputs], axis=0
    )
    betas = np.concatenate(
        [_as_numpy(p.betas).astype(np.float32) for p in outputs], axis=0
    )
    transl = np.concatenate(
        [
            _as_numpy(p.transl).astype(np.float32)
            if p.transl is not None
            else np.zeros((1, 3), dtype=np.float32)
            for p in outputs
        ],
        axis=0,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas,
        transl=transl,
    )
    print(f"Saved refined SMPL-X params to: {args.out}")


if __name__ == "__main__":
    main()
