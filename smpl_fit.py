import argparse
import logging
import os
import sys
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
import smplx
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import config
from smplify import SMPLify3D

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
        "--batch-size",
        dest="batch_size",
        type=int,
        default=1,
        help="Input batch size.",
    )
    parser.add_argument(
        "--num-smplify-iters",
        dest="num_smplify_iters",
        type=int,
        default=100,
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
        "--num-joints",
        dest="num_joints",
        type=int,
        default=22,
        help="Number of joints in the input data.",
    )
    parser.add_argument(
        "--joint-category",
        dest="joint_category",
        type=str,
        default="AMASS",
        choices=["AMASS", "orig"],
        help="Joint mapping to use.",
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
        "--file",
        "--files",
        dest="input_file",
        type=str,
        default="test_motion.npy",
        help="Input .npy file name or path (extension optional).",
    )
    parser.add_argument(
        "--smpl-dir",
        dest="smpl_dir",
        type=Path,
        default=Path(config.smpl_dir),
        help="Path to SMPL model directory.",
    )
    parser.add_argument(
        "--smpl-mean-file",
        dest="smpl_mean_file",
        type=Path,
        default=Path(config.SMPL_MEAN_FILE),
        help="Path to SMPL mean parameters file (.h5).",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
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

    # Load mean pose/shape
    smpl_mean_file = Path(config.SMPL_MEAN_FILE).expanduser()
    logger.info("Loading mean pose/shape from %s", smpl_mean_file)
    with h5py.File(smpl_mean_file, "r") as f:
        init_mean_pose = torch.as_tensor(f["pose"][:]).unsqueeze(0).float().to(device)
        init_mean_shape = torch.as_tensor(f["shape"][:]).unsqueeze(0).float().to(device)

    pred_pose = torch.zeros(opt.batch_size, 72, device=device)
    pred_betas = torch.zeros(opt.batch_size, 10, device=device)
    pred_cam_t = torch.zeros(opt.batch_size, 3, device=device)
    keypoints_3d = torch.zeros(opt.batch_size, opt.num_joints, 3, device=device)

    logger.info("Initializing SMPLify3D...")
    smplify = SMPLify3D(
        smplxmodel=smplmodel,
        batch_size=opt.batch_size,
        joints_category=opt.joint_category,
        num_iters=opt.num_smplify_iters,
        device=device,
    )
    logger.info("SMPLify3D initialized.")

    input_path = Path(opt.input_file).expanduser().absolute()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    purename = input_path.stem

    logger.info("Loading 3D joints from %s", input_path)
    data = np.load(input_path)
    data_tensor = torch.from_numpy(data).float().to(device)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dir_save = opt.work_dir.expanduser() / purename / f"{timestamp}"
    dir_save.mkdir(parents=True, exist_ok=True)

    num_seqs = data.shape[0]
    logger.info("Processing %d frames", num_seqs)

    pbar = tqdm(
        range(num_seqs),
        dynamic_ncols=True,
        desc="Fitting SMPL to 3D joints",
        total=num_seqs,
    )

    if opt.joint_category == "AMASS":
        confidence_input = torch.ones(opt.num_joints, device=device)
        if opt.fix_foot:
            confidence_input[7] = 1.5
            confidence_input[8] = 1.5
            confidence_input[10] = 1.5
            confidence_input[11] = 1.5
    else:
        raise ValueError(f"Unsupported joint category: {opt.joint_category}")

    prev_pose = init_mean_pose.to(device)
    prev_betas = init_mean_shape.to(device)
    prev_cam = torch.tensor([0.0, 0.0, 0.0], device=device)

    for idx in pbar:
        keypoints_3d[0, :, :].copy_(data_tensor[idx])

        pred_betas[0, :] = prev_betas
        pred_pose[0, :] = prev_pose
        pred_cam_t[0, :] = prev_cam

        (
            new_opt_vertices,
            new_opt_joints,
            new_opt_pose,
            new_opt_betas,
            new_opt_cam_t,
            new_opt_joint_loss,
        ) = smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            keypoints_3d,
            conf_3d=confidence_input,
            seq_ind=idx,
        )

        param = {
            "beta": new_opt_betas.detach().cpu().numpy(),
            "pose": new_opt_pose.detach().cpu().numpy(),
            "cam": new_opt_cam_t.detach().cpu().numpy(),
        }

        root_position = keypoints_3d[0, 0, :].detach().cpu().numpy()
        param["root"] = root_position

        pbar.set_postfix(
            {
                "joint_loss": f"{new_opt_joint_loss.item():.4f}",
            },
            refresh=False,
        )

        joblib.dump(param, dir_save / f"{idx:04d}.pkl", compress=3)

        prev_betas = new_opt_betas.detach()
        prev_pose = new_opt_pose.detach()
        prev_cam = new_opt_cam_t.detach()


if __name__ == "__main__":
    main()
