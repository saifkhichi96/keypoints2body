from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .. import optimize_params_sequence
from ..core.config import FrameOptimizeConfig, SequenceOptimizeConfig

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    """Configure CLI logging format and level."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate keypoints2body on AMASS by fitting each sequence and computing MPJAE."
    )
    parser.add_argument("--amass-root", type=Path, required=True)
    parser.add_argument("--limit-seqs", type=int, default=-1)
    parser.add_argument("--limit-frames", type=int, default=-1)
    parser.add_argument("--skip-start-frames", type=int, default=0)
    parser.add_argument("--num-shape-iters", type=int, default=40)
    parser.add_argument("--num-shape-frames", type=int, default=50)
    parser.add_argument("--num-body-iters-first", type=int, default=100)
    parser.add_argument("--num-body-iters", type=int, default=50)
    parser.add_argument("--fix-shape", action="store_true")
    parser.add_argument("--fix-foot", action="store_true")
    parser.add_argument("--use-adam", action="store_true")
    parser.add_argument("--save-pred-dir", type=Path, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--cuda", dest="cuda", action="store_true", default=torch.cuda.is_available()
    )
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


def discover_amass_npz_files(root: Path) -> list[Path]:
    """Discover evaluation sequence files under root."""
    root = root.expanduser().resolve()
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*.npz") if p.is_file())


def load_amass_sequence(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one AMASS sequence and corresponding pose targets."""
    data = np.load(npz_path)
    required = ["joints", "global_orient", "body_pose"]
    missing = [k for k in required if k not in data]
    if missing:
        data.close()
        raise KeyError(f"Missing keys {missing} in {npz_path}")

    joints = np.asarray(data["joints"], dtype=np.float32)[:, :22, :]
    global_orient = np.asarray(data["global_orient"], dtype=np.float32)
    body_pose = np.asarray(data["body_pose"], dtype=np.float32)
    data.close()

    if global_orient.ndim == 1:
        global_orient = global_orient[None, :]
    if body_pose.ndim == 1:
        body_pose = body_pose[None, :]

    gt_pose = np.concatenate([global_orient, body_pose], axis=1).astype(np.float32)
    t_size = min(joints.shape[0], gt_pose.shape[0])
    if t_size == 0:
        raise ValueError(f"Empty sequence in {npz_path}")
    return joints[:t_size], gt_pose[:t_size]


def rotvec_to_rotmat(rotvec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert axis-angle rotation vectors to rotation matrices."""
    rotvec = np.asarray(rotvec, dtype=np.float32)
    x = rotvec[..., 0]
    y = rotvec[..., 1]
    z = rotvec[..., 2]
    theta2 = x * x + y * y + z * z
    theta = np.sqrt(theta2)
    theta_safe = np.where(theta > eps, theta, 1.0).astype(np.float32, copy=False)

    sin_t = np.sin(theta_safe)
    cos_t = np.cos(theta_safe)
    a = sin_t / theta_safe
    b = (1.0 - cos_t) / (theta_safe * theta_safe)

    small = theta <= eps
    if np.any(small):
        t2 = theta2[small]
        a[small] = 1.0 - t2 / 6.0 + (t2 * t2) / 120.0
        b[small] = 0.5 - t2 / 24.0 + (t2 * t2) / 720.0

    xy = x * y
    xz = x * z
    yz = y * z
    xx = x * x
    yy = y * y
    zz = z * z

    r = np.empty(rotvec.shape[:-1] + (3, 3), dtype=np.float32)
    r[..., 0, 0] = 1.0 - b * (yy + zz)
    r[..., 0, 1] = b * xy - a * z
    r[..., 0, 2] = b * xz + a * y
    r[..., 1, 0] = b * xy + a * z
    r[..., 1, 1] = 1.0 - b * (xx + zz)
    r[..., 1, 2] = b * yz - a * x
    r[..., 2, 0] = b * xz - a * y
    r[..., 2, 1] = b * yz + a * x
    r[..., 2, 2] = 1.0 - b * (xx + yy)
    return r


def compute_angular_error_deg(
    pred_rotvec: np.ndarray, gt_rotvec: np.ndarray
) -> np.ndarray:
    """Compute angular error in degrees between rotations."""
    r_pred = rotvec_to_rotmat(pred_rotvec)
    r_gt = rotvec_to_rotmat(gt_rotvec)
    trace = np.sum(r_pred * r_gt, axis=(-1, -2))
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    return np.degrees(np.arccos(cos_theta))


def evaluate_pose_pair(
    pred_pose: np.ndarray, gt_pose: np.ndarray
) -> tuple[float, float, int]:
    """Evaluate MPJAE-style metric for predicted vs. GT poses."""
    n = min(gt_pose.shape[0], pred_pose.shape[0])
    d = min(gt_pose.shape[1], pred_pose.shape[1])
    d = (d // 3) * 3
    gt_pose = gt_pose[:n, :d]
    pred_pose = pred_pose[:n, :d]

    gt_rotvec = gt_pose.reshape(n, d // 3, 3)
    pred_rotvec = pred_pose.reshape(n, d // 3, 3)

    angles_deg = compute_angular_error_deg(pred_rotvec, gt_rotvec)
    angle_sum = float(np.sum(angles_deg, dtype=np.float64))
    angle_count = int(angles_deg.size)
    return angle_sum / angle_count, angle_sum, angle_count


def fit_sequence(
    joints: np.ndarray, args: argparse.Namespace, device: torch.device
) -> np.ndarray:
    """Fit one sequence and return concatenated predicted poses."""
    frame_cfg = FrameOptimizeConfig(
        coordinate_mode="world",
        use_lbfgs=not args.use_adam,
        num_iters_first=args.num_body_iters_first,
        num_iters_followup=args.num_body_iters,
        joints_category="AMASS",
        freeze_betas=args.fix_shape,
    )
    seq_cfg = SequenceOptimizeConfig(
        frame=frame_cfg,
        num_shape_iters=args.num_shape_iters,
        num_shape_frames=args.num_shape_frames,
        use_shape_optimization=not args.fix_shape,
        fix_foot=args.fix_foot,
        limit_frames=args.limit_frames if args.limit_frames > 0 else None,
    )

    results = optimize_params_sequence(
        joints,
        body_model="smpl",
        joint_layout="AMASS",
        config=seq_cfg,
        device=device,
    )
    return np.concatenate(
        [r.params.pose.detach().cpu().numpy() for r in results], axis=0
    )


def save_prediction_pose(
    pred_pose: np.ndarray, src_path: Path, dataset_root: Path, save_root: Path
) -> None:
    """Save predicted pose array to compressed npz file."""
    save_root.mkdir(parents=True, exist_ok=True)
    try:
        rel = src_path.resolve().relative_to(dataset_root.resolve())
        out_path = save_root / rel
    except ValueError:
        out_path = save_root / src_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, pose=pred_pose)


def main() -> None:
    """Run end-to-end evaluation CLI pipeline."""
    args = parse_args()
    configure_logging(args.log_level)

    if args.skip_start_frames < 0:
        raise ValueError("--skip-start-frames must be >= 0.")

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.cuda and torch.cuda.is_available()
        else torch.device("cpu")
    )

    files = discover_amass_npz_files(args.amass_root)
    if args.limit_seqs > 0:
        files = files[: args.limit_seqs]
    if not files:
        raise FileNotFoundError(f"No .npz files found under {args.amass_root}")

    total_angle_sum = 0.0
    total_angle_count = 0
    num_success = 0
    num_failed = 0

    iterator = tqdm(files, desc="Evaluating", dynamic_ncols=True)
    for seq_path in iterator:
        try:
            joints, gt_pose = load_amass_sequence(seq_path)
            if args.limit_frames > 0:
                joints = joints[: args.limit_frames]
                gt_pose = gt_pose[: args.limit_frames]

            pred_pose = fit_sequence(joints=joints, args=args, device=device)

            if args.skip_start_frames > 0:
                if pred_pose.shape[0] <= args.skip_start_frames:
                    raise ValueError(
                        f"Sequence too short after skipping {args.skip_start_frames} frame(s): {pred_pose.shape[0]} available."
                    )
                pred_pose_eval = pred_pose[args.skip_start_frames :]
                gt_pose_eval = gt_pose[args.skip_start_frames :]
            else:
                pred_pose_eval = pred_pose
                gt_pose_eval = gt_pose

            seq_mpjae, seq_angle_sum, seq_angle_count = evaluate_pose_pair(
                pred_pose_eval, gt_pose_eval
            )
            total_angle_sum += seq_angle_sum
            total_angle_count += seq_angle_count
            num_success += 1

            dataset_mpjae = total_angle_sum / total_angle_count
            iterator.set_postfix(
                seq_mpjae=f"{seq_mpjae:.3f}", dataset_mpjae=f"{dataset_mpjae:.3f}"
            )

            if args.save_pred_dir is not None:
                save_prediction_pose(
                    pred_pose=pred_pose,
                    src_path=seq_path,
                    dataset_root=args.amass_root,
                    save_root=args.save_pred_dir,
                )
        except Exception as exc:
            num_failed += 1
            logger.warning("Failed on %s: %s", seq_path, exc)
            if args.fail_fast:
                raise

    if total_angle_count == 0:
        raise RuntimeError("No valid sequence was evaluated.")

    dataset_mpjae = total_angle_sum / total_angle_count
    logger.info(
        "Finished evaluation: success=%d failed=%d dataset_MPJAE=%.6f deg",
        num_success,
        num_failed,
        dataset_mpjae,
    )
    print(f"Dataset MPJAE(global_orient + body_pose): {dataset_mpjae:.6f} deg")


if __name__ == "__main__":
    main()
