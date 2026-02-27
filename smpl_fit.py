import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from keypoints2body import optimize_params_sequence
from keypoints2body.core.config import FrameOptimizeConfig, SequenceOptimizeConfig
from keypoints2body.io import load_motion_data, write_smplx_zip

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    """Configure script logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args():
    """Parse CLI arguments for sequence fitting script."""
    parser = argparse.ArgumentParser(
        description="Fit SMPL parameters to a sequence of 3D joints."
    )
    parser.add_argument("--file", dest="input_file", type=Path, required=True)

    parser.add_argument(
        "--coordinate-mode",
        choices=["camera", "world"],
        default="world",
        help="Choose camera-space or world-space fitter.",
    )

    parser.add_argument("--num-shape-iters", type=int, default=40)
    parser.add_argument("--num-shape-frames", type=int, default=50)

    parser.add_argument(
        "--num-iters",
        type=int,
        default=30,
        help="Used in camera mode only.",
    )
    parser.add_argument(
        "--num-body-iters-first",
        type=int,
        default=100,
        help="Used in world mode only.",
    )
    parser.add_argument(
        "--num-body-iters",
        type=int,
        default=50,
        help="Used in world mode only.",
    )

    parser.add_argument(
        "--cuda", dest="cuda", action="store_true", default=torch.cuda.is_available()
    )
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    parser.add_argument("--gpu-id", type=int, default=0)

    parser.add_argument("--use-adam", action="store_true")
    parser.add_argument(
        "--fix-shape", action="store_true", help="World mode: keep betas fixed."
    )
    parser.add_argument("--fix-foot", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=Path("./work_dirs"))
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--joint-layout", type=str, default=None)
    return parser.parse_args()


def main():
    """Run sequence fitting script using keypoints2body APIs."""
    opt = parse_args()
    configure_logging(opt.log_level)

    device = (
        torch.device(f"cuda:{opt.gpu_id}")
        if opt.cuda and torch.cuda.is_available()
        else torch.device("cpu")
    )
    input_path = Path(opt.input_file).expanduser().absolute()
    joints, skeleton, _ = load_motion_data(input_path, layout=opt.joint_layout)

    if opt.limit > 0 and opt.limit < joints.shape[0]:
        joints = joints[: opt.limit]

    if opt.coordinate_mode == "camera":
        frame_cfg = FrameOptimizeConfig(
            coordinate_mode="camera",
            use_lbfgs=not opt.use_adam,
            num_iters=opt.num_iters,
            joints_category=skeleton,
            freeze_betas=False,
        )
        use_shape_optimization = True
    else:
        frame_cfg = FrameOptimizeConfig(
            coordinate_mode="world",
            use_lbfgs=not opt.use_adam,
            num_iters_first=opt.num_body_iters_first,
            num_iters_followup=opt.num_body_iters,
            joints_category=skeleton,
            freeze_betas=opt.fix_shape,
        )
        use_shape_optimization = not opt.fix_shape

    seq_cfg = SequenceOptimizeConfig(
        frame=frame_cfg,
        num_shape_iters=opt.num_shape_iters,
        num_shape_frames=opt.num_shape_frames,
        use_shape_optimization=use_shape_optimization,
        fix_foot=opt.fix_foot,
    )

    results = optimize_params_sequence(
        joints,
        body_model="smpl",
        joint_layout=skeleton,
        config=seq_cfg,
        device=device,
    )

    poses_out = np.concatenate(
        [r.params.pose.detach().cpu().numpy() for r in results], axis=0
    )
    betas_out = np.concatenate(
        [r.params.betas.detach().cpu().numpy() for r in results], axis=0
    )
    transl_out = np.concatenate(
        [r.params.transl.detach().cpu().numpy() for r in results], axis=0
    )

    dir_save = (
        opt.work_dir.expanduser() / input_path.stem / time.strftime("%Y%m%d_%H%M%S")
    )
    dir_save.mkdir(parents=True, exist_ok=True)
    zip_path = write_smplx_zip(dir_save, poses_out, betas_out, transl_out)
    logger.info("Done. Saved SMPL-X parameters to %s", zip_path)


if __name__ == "__main__":
    main()
