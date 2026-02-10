import argparse
import logging
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import smplx
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import config
from io_utils import load_motion_data, write_smplx_zip
from smplify import SMPLify3D, optimize_shape_multi_frame

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit SMPL parameters to a sequence of 3D joints."
    )
    parser.add_argument(
        "--file",
        dest="input_file",
        type=Path,
        required=True,
        help="Path to 3D joints (.npy or .csv). Shape (T,J,3).",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=1,
        help="Input batch size.",
    )
    parser.add_argument(
        "--num-shape-iters",
        type=int,
        default=40,
        help="Number of LBFGS iterations for multi-frame shape optimization.",
    )
    parser.add_argument(
        "--num-shape-frames",
        type=int,
        default=50,
        help="Max number of frames to use for shape optimization (-1 for all).",
    )
    parser.add_argument(
        "--num-iters",
        dest="num_iters",
        type=int,
        default=30,
        help="Number of SMPLify iterations for each optimization stage.",
    )
    parser.add_argument(
        "--cuda",
        dest="cuda",
        action="store_true",
        default=torch.cuda.is_available(),
        help="Enable CUDA if available.",
    )
    parser.add_argument(
        "--cpu",
        dest="cuda",
        action="store_false",
        help="Force CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--gpu-id",
        dest="gpu_id",
        type=int,
        default=0,
        help="GPU id to use when CUDA is enabled.",
    )
    parser.add_argument(
        "--use-adam",
        dest="use_adam",
        action="store_true",
        help="Use Adam optimizer in SMPLify.",
    )
    parser.add_argument(
        "--fix-foot",
        dest="fix_foot",
        action="store_true",
        help="Increase confidence for foot and ankle joints.",
    )
    parser.add_argument(
        "--work-dir",
        dest="work_dir",
        type=Path,
        default=Path("./work_dirs"),
        help="Folder where results will be saved.",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit the number of frames to process (-1 for all).",
    )
    return parser.parse_args()


def main():
    opt = parse_args()
    configure_logging(opt.log_level)
    logger.debug("Parsed options: %s", opt)

    if opt.cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{opt.gpu_id}")
    else:
        device = torch.device("cpu")
    logger.info(
        "Using device: %s (cuda available: %s)", device, torch.cuda.is_available()
    )
    if torch.cuda.is_available():
        logger.info("CUDA device name: %s", torch.cuda.get_device_name(opt.gpu_id))

    # Load SMPL model
    smpl_dir = Path(config.smpl_dir).expanduser()
    logger.info("Loading SMPL model from %s", smpl_dir)
    smpl_model = smplx.create(
        str(smpl_dir),
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        batch_size=opt.batch_size,
    ).to(device)

    # Load motion data
    input_path = Path(opt.input_file).expanduser().absolute()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info("Loading 3D joints from %s", input_path)
    joints, skeleton, J = load_motion_data(input_path)  # (T, J, 3)
    T = joints.shape[0]

    if opt.limit > 0 and opt.limit < T:
        joints = joints[: opt.limit]
        T = opt.limit

    logger.info("Loaded %d frames in %s format with %d joints", T, skeleton, J)

    # Create input tensor
    data_tensor = torch.as_tensor(joints).float().to(device)

    if skeleton in ["AMASS", "orig"]:
        confidence_input = torch.ones(J, device=device)
        if opt.fix_foot:
            confidence_input[7] = 1.5
            confidence_input[8] = 1.5
            confidence_input[10] = 1.5
            confidence_input[11] = 1.5
    else:
        raise ValueError(f"Unsupported joint category: {skeleton}")

    # Load mean pose/shape
    smpl_mean_file = Path(config.SMPL_MEAN_FILE).expanduser()
    logger.info("Loading mean pose/shape from %s", smpl_mean_file)
    with h5py.File(smpl_mean_file, "r") as f:
        init_mean_pose = torch.as_tensor(f["pose"][:]).unsqueeze(0).float().to(device)
        init_mean_shape = torch.as_tensor(f["shape"][:]).unsqueeze(0).float().to(device)

    # Prepare output directory
    purename = input_path.stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dir_save = opt.work_dir.expanduser() / purename / f"{timestamp}"
    dir_save.mkdir(parents=True, exist_ok=True)
    logger.info("Saving results to %s", dir_save)

    # Stage 1: Multi-frame shape optimization
    logger.info("Starting multi-frame shape optimization...")
    if opt.num_shape_frames < 0 or opt.num_shape_frames >= T:
        frame_indices = list(range(T))
    else:
        frame_indices = list(range(opt.num_shape_frames))
    logger.info("Running shape optimization on %d / %d frames", len(frame_indices), T)
    betas_opt = optimize_shape_multi_frame(
        smpl_model,
        init_betas=init_mean_shape,
        pose_init=init_mean_pose.repeat(T, 1),
        j3d_world=data_tensor,
        joints_category=skeleton,
        num_iters=opt.num_shape_iters,
        step_size=1e-1,
        use_lbfgs=not opt.use_adam,
        device=device,
        frame_indices=frame_indices,
        joints3d_conf=confidence_input,
        shape_prior_weight=5.0,
    )
    logger.info("Shape optimization finished.")

    # Stage 2: Per-frame pose/translation optimization
    logger.info("Initializing SMPLify3D...")
    smplify = SMPLify3D(
        smplxmodel=smpl_model,
        batch_size=opt.batch_size,
        joints_category=skeleton,
        use_lbfgs=not opt.use_adam,
        num_iters=opt.num_iters,
        device=device,
    )
    logger.info("SMPLify3D initialized.")

    pbar = tqdm(
        range(T),
        dynamic_ncols=True,
        desc="Fitting SMPL to 3D joints",
        total=T,
    )

    poses_out = []
    betas_out_all = []
    transl_out_all = []

    pred_pose = torch.zeros(opt.batch_size, 72, device=device)
    pred_betas = torch.zeros(opt.batch_size, 10, device=device)
    pred_transl = torch.zeros(opt.batch_size, 3, device=device)
    keypoints_3d = torch.zeros(opt.batch_size, J, 3, device=device)

    prev_pose = init_mean_pose.to(device)
    prev_betas = betas_opt.to(device)
    prev_transl = torch.tensor([0.0, 0.0, 0.0], device=device)

    for idx in pbar:
        keypoints_3d[0, :, :].copy_(data_tensor[idx])

        # Initialize with previous estimates
        pred_pose[0, :] = prev_pose
        pred_betas[0, :] = prev_betas
        pred_transl[0, :] = prev_transl

        # Optimize
        (
            new_opt_vertices,
            new_opt_joints,
            new_opt_pose,
            new_opt_betas,
            new_opt_transl,
            new_opt_joint_loss,
        ) = smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_transl.detach(),
            keypoints_3d,
            conf_3d=confidence_input,
            seq_ind=idx,
        )

        poses_out.append(new_opt_pose.detach().cpu().numpy())
        betas_out_all.append(new_opt_betas.detach().cpu().numpy())
        transl_out_all.append(new_opt_transl.detach().cpu().numpy())

        pbar.set_postfix(
            {
                "joint_loss": f"{new_opt_joint_loss.item():.4f}",
            },
            refresh=False,
        )

        # Update previous estimates
        prev_pose = new_opt_pose.detach()
        prev_betas = new_opt_betas.detach()
        prev_transl = new_opt_transl.detach()

    # Concatenate frame results
    poses_out = np.concatenate(poses_out, axis=0)  # (T,72)
    betas_out_all = np.concatenate(betas_out_all, axis=0)  # (T,10)
    transl_out_all = np.concatenate(transl_out_all, axis=0)  # (T,3)

    # Write SMPL-X sequence
    zip_path = write_smplx_zip(
        dir_save,
        poses_out,
        betas_out_all,
        transl_out_all,
    )
    logger.info("Done. Saved SMPL-X parameters to %s", zip_path)


if __name__ == "__main__":
    main()
