import argparse
import io
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import smplx
import torch
import trimesh
from tqdm import tqdm

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - optional dependency
    imageio = None

sys.path.append(str(Path(__file__).parent / "src"))
import config  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render SMPL sequence results produced by fit_seq.py"
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to folder containing per-frame .pkl files.",
    )
    parser.add_argument(
        "--smpl-dir",
        type=Path,
        default=Path(config.smpl_dir),
        help="Path to SMPL model directory.",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="neutral",
        choices=["neutral", "male", "female"],
        help="SMPL gender to load.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to run SMPL on.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU id to use when device is cuda.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second used for time estimation.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=800,
        help="Render width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Render height in pixels.",
    )
    parser.add_argument(
        "--write-frames",
        type=Path,
        default=None,
        help="Optional output folder to store rendered PNG frames.",
    )
    parser.add_argument(
        "--write-video",
        type=Path,
        default=None,
        help="Optional output video path (e.g., output.mp4). Requires imageio.",
    )
    parser.add_argument(
        "--display-frame",
        type=int,
        default=None,
        help="Show a specific frame interactively instead of offscreen rendering.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_frame_params(pkl_path: Path):
    params = joblib.load(pkl_path)
    return params["pose"], params["beta"], params["cam"]


def build_smpl_model(
    model_dir: Path, gender: str, batch_size: int, device: torch.device
):
    return smplx.create(
        model_dir,
        model_type="smpl",
        gender=gender,
        ext="pkl",
        batch_size=batch_size,
    ).to(device)


def smpl_output_to_mesh(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )


def render_frame_offscreen(
    mesh: trimesh.Trimesh, width: int, height: int
) -> np.ndarray:
    if imageio is None:
        raise ImportError(
            "imageio is required for rendering. Install with `pip install imageio`."
        )
    scene = trimesh.Scene(mesh)
    # visible=False avoids creating a viewer window per frame
    png_bytes = scene.save_image(resolution=(width, height), visible=False)
    frame = imageio.imread(io.BytesIO(png_bytes))
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    return frame


def main():
    args = parse_args()
    configure_logging(args.log_level)

    device = torch.device(
        f"cuda:{args.gpu_id}"
        if args.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")

    if (args.write_frames or args.write_video) and imageio is None:
        raise ImportError(
            "imageio is required for rendering with trimesh. Install with `pip install imageio imageio-ffmpeg`."
        )

    results_dir = args.results_dir.expanduser()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    pkl_files = sorted(results_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {results_dir}")

    logger.info("Found %d frames in %s", len(pkl_files), results_dir)

    smpl_model = build_smpl_model(
        args.smpl_dir.expanduser(), args.gender, batch_size=1, device=device
    )
    smpl_faces = smpl_model.faces

    frame_dir = None
    video_writer = None

    if args.write_frames:
        frame_dir = args.write_frames.expanduser()
        frame_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Rendering frames to %s", frame_dir)

    if args.write_video:
        video_path = args.write_video.expanduser()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=args.fps)
        logger.info("Writing video to %s", video_path)

    display_index = args.display_frame
    if display_index is not None and (
        display_index < 0 or display_index >= len(pkl_files)
    ):
        raise ValueError(
            f"display-frame {display_index} is out of range for {len(pkl_files)} frames"
        )

    progress_iter = tqdm(
        enumerate(pkl_files),
        total=len(pkl_files),
        desc="Rendering frames",
        dynamic_ncols=True,
    )

    try:
        for idx, pkl_path in progress_iter:
            pose_np, betas_np, cam_np = load_frame_params(pkl_path)
            pose = torch.from_numpy(pose_np).float().to(device)
            betas = torch.from_numpy(betas_np).float().to(device)
            cam = torch.from_numpy(cam_np).float().to(device)

            with torch.no_grad():
                output = smpl_model(
                    betas=betas,
                    global_orient=pose[:, :3],
                    body_pose=pose[:, 3:],
                    transl=cam,
                    return_verts=True,
                )
            vertices = output.vertices.detach().cpu().numpy().squeeze()
            mesh = smpl_output_to_mesh(vertices, smpl_faces)

            if frame_dir or video_writer:
                frame_np = render_frame_offscreen(mesh, args.width, args.height)
                if frame_dir:
                    frame_path = frame_dir / f"{idx:04d}.png"
                    imageio.imwrite(frame_path, frame_np)
                if video_writer:
                    video_writer.append_data(frame_np)

            if display_index is not None and idx == display_index:
                mesh.show()
                break
    finally:
        if video_writer:
            video_writer.close()

    logger.info("Done.")


if __name__ == "__main__":
    main()
