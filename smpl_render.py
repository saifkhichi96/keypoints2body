import argparse
import io
import logging
import os
import os.path as osp
import sys
from pathlib import Path

if not os.environ.get("DISPLAY"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import imageio.v2 as imageio
import joblib
import numpy as np
import pyrender
import smplx
import toml
import torch
import trimesh
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent / "src"))
import config  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render an SMPL sequence either composited onto a background video "
            "or as a mesh-only video."
        )
    )

    parser.add_argument(
        "results_dir", type=Path, help="Directory containing .pkl SMPL parameter files."
    )
    parser.add_argument(
        "--smpl-dir",
        type=Path,
        default=Path(config.smpl_dir),
        help="Directory with SMPL model files.",
    )
    parser.add_argument("--gender", type=str, default="neutral")

    parser.add_argument(
        "--background-video",
        type=Path,
        help="Background video to composite the mesh onto.",
    )
    parser.add_argument(
        "--camera-file",
        type=Path,
        help="Camera calibration (.toml) for background compositing.",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        help="Camera entry name inside the calibration file. Defaults to the background video stem.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        required=True,
        help="Output video path (mp4, gif, etc.).",
    )

    parser.add_argument(
        "--width", type=int, default=800, help="Render width (mesh-only mode)."
    )
    parser.add_argument(
        "--height", type=int, default=800, help="Render height (mesh-only mode)."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS to use when no background video is provided.",
    )

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--gpu-id", type=int, default=0)

    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def configure_logging(level):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_frame_params(pkl_path: Path):
    """Load SMPL pose/beta/translation."""
    data = joblib.load(pkl_path)
    pose = data["pose"]
    betas = data["beta"]
    transl = data.get("transl", data.get("cam"))
    return pose, betas, transl


def load_camera(camera_name, camera_file):
    if camera_file.suffix.lower() != ".toml":
        raise ValueError("Camera file must be a .toml calibration file")

    cam = toml.load(camera_file)[camera_name]

    K = np.array(cam["matrix"], dtype=np.float32)

    rvec = np.array(cam["rotation"])
    R, _ = cv2.Rodrigues(rvec)
    R = R.astype(np.float32)

    t = np.array(cam["translation"], dtype=np.float32).reshape(3)

    return K, R, t


def load_frame_iterator(results_dir: Path):
    """
    Return an iterator over (pose, betas, transl) tuples and the number of frames.

    Prefers per-frame .pkl files; falls back to results.npz produced by older runs.
    """
    pkls = sorted(results_dir.glob("*.pkl"))
    if pkls:

        def iterator():
            for pkl_path in pkls:
                yield load_frame_params(pkl_path)

        logger.info("Found %d frame PKL files in %s", len(pkls), results_dir)
        return iterator(), len(pkls)

    npz_file = results_dir / "results.npz"
    if not npz_file.exists():
        raise FileNotFoundError(f"No frame PKLs or results.npz found in {results_dir}")

    data = np.load(npz_file)
    poses = data["pose"]

    betas_arr = None
    for key in ("betas", "beta"):
        if key in data.files:
            betas_arr = data[key]
            break

    cams = None
    for key in ("transl", "cam"):
        if key in data.files:
            cams = data[key]
            break

    if betas_arr is None or cams is None:
        raise KeyError("results.npz must contain betas/beta and transl/cam arrays")

    num_frames = poses.shape[0]
    logger.info("Found %d frames in %s", num_frames, npz_file)

    def iterator():
        for idx in range(num_frames):
            pose = poses[idx]
            betas = betas_arr[min(idx, betas_arr.shape[0] - 1)]
            transl = cams[min(idx, cams.shape[0] - 1)]

            if pose.ndim == 1:
                pose = pose[None, ...]
            if betas.ndim == 1:
                betas = betas[None, ...]
            if transl.ndim == 1:
                transl = transl[None, ...]

            yield pose, betas, transl

    return iterator(), num_frames


def render_smpl_reprojection(
    verts_world, faces, K, R, t, background_img, mesh_color=[0.7, 0.7, 0.75, 1.0]
):
    """
    Full correct reprojection into background image using PyRender.
    """

    H, W = background_img.shape[:2]

    # Conversion matrices
    M_osim_to_smpl = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)

    M_smpl_to_osim = M_osim_to_smpl.T  # inverse

    # Convert camera rotation from OpenSim to SMPL coordinates
    R = R @ M_smpl_to_osim

    # Camera translation stays valid (OpenCV convention)
    t = t.copy()

    # ---------------------------------------
    # 1) World → Camera coordinates
    # ---------------------------------------
    verts_cam = (R @ verts_world.T).T + t.reshape(1, 3)

    # OpenCV uses +Z forward; PyRender uses +Z backward
    verts_cam[:, 2] *= -1

    # ---------------------------------------
    # 2) Build trimesh mesh
    # ---------------------------------------
    mesh_tm = trimesh.Trimesh(vertices=verts_cam, faces=faces, process=False)

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=mesh_color, roughnessFactor=0.4, metallicFactor=0.1
    )

    mesh = pyrender.Mesh.from_trimesh(mesh_tm, material=material, smooth=True)

    # ---------------------------------------
    # 3) Build scene
    # ---------------------------------------
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.2, 0.2, 0.2])
    scene.add(mesh)

    # ---------------------------------------
    # 4) Camera intrinsics
    # ---------------------------------------
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    cam_pose = np.eye(4)
    scene.add(camera, pose=cam_pose)

    # ---------------------------------------
    # 5) Lighting
    # ---------------------------------------
    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    fill_light = pyrender.PointLight(color=np.ones(3), intensity=1.5)

    scene.add(key_light, pose=np.eye(4))
    scene.add(fill_light, pose=np.eye(4))

    # ---------------------------------------
    # 6) Render
    # ---------------------------------------
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    rgb, depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.RGBA)
    renderer.delete()

    # Make mask upside down before compositing
    rgb = np.flipud(rgb)

    # ---------------------------------------
    # 7) Alpha composite over background
    # ---------------------------------------
    alpha = rgb[..., 3:] / 255.0
    out = (1 - alpha) * background_img + alpha * rgb[..., :3]

    return out.astype(np.uint8)


def render_frame_offscreen(
    mesh: trimesh.Trimesh, width: int, height: int
) -> np.ndarray:
    # Some environments need pyglet headless for trimesh offscreen rendering
    try:  # pragma: no cover - optional dependency
        import pyglet

        pyglet.options["headless"] = True
        os.environ.setdefault("PYGLET_HEADLESS", "true")
    except Exception:
        pass

    scene = trimesh.Scene(mesh)
    png_bytes = scene.save_image(resolution=(width, height), visible=False)
    frame = imageio.imread(io.BytesIO(png_bytes))
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    return frame


def main():
    args = parse_args()
    configure_logging(args.log_level)

    if args.background_video and not args.camera_file:
        raise ValueError("--camera-file is required when using --background-video")

    device = torch.device(
        f"cuda:{args.gpu_id}"
        if args.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")

    # ------------------------------------------------------------
    # Load frames
    # ------------------------------------------------------------
    frame_iter, num_frames = load_frame_iterator(args.results_dir)

    # ------------------------------------------------------------
    # Load SMPL model
    # ------------------------------------------------------------
    smpl = smplx.create(
        args.smpl_dir, model_type="smpl", gender=args.gender, ext="pkl", batch_size=1
    ).to(device)
    faces = smpl.faces

    # ------------------------------------------------------------
    # Background + camera
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # Determine rendering mode
    # ------------------------------------------------------------
    background_mode = args.background_video is not None
    if background_mode:
        bg_reader = imageio.get_reader(args.background_video)
        fps = bg_reader.get_meta_data().get("fps", args.fps)

        camera_name = (
            args.camera_name or osp.splitext(osp.basename(args.background_video))[0]
        )
        K, R, t = load_camera(camera_name, args.camera_file)
        logger.info("Loaded camera intrinsics + extrinsics for '%s'", camera_name)

        writer = imageio.get_writer(args.output_video, fps=fps)
        logger.info(f"Writing composited video → {args.output_video}")

        bg_iter = iter(bg_reader)
    else:
        writer = imageio.get_writer(args.output_video, fps=args.fps)
        logger.info(f"Writing mesh-only video → {args.output_video}")

    # ------------------------------------------------------------
    # Per-frame loop
    # ------------------------------------------------------------
    for idx, frame_params in tqdm(enumerate(frame_iter), total=num_frames):
        pose_np, betas_np, transl_np = frame_params
        pose = torch.tensor(pose_np).float().to(device)
        betas = torch.tensor(betas_np).float().to(device)
        transl = torch.tensor(transl_np).float().to(device)

        with torch.no_grad():
            out = smpl(
                betas=betas,
                global_orient=pose[:, :3],
                body_pose=pose[:, 3:],
                transl=transl,
                return_verts=True,
            )
        verts_world = out.vertices.detach().cpu().numpy().squeeze()

        if background_mode:
            try:
                bg = next(bg_iter)
            except StopIteration:
                logger.warning("Background video ended early")
                break

            frame = render_smpl_reprojection(
                verts_world=verts_world, faces=faces, K=K, R=R, t=t, background_img=bg
            )
        else:
            mesh_tm = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
            frame = render_frame_offscreen(mesh_tm, args.width, args.height)

        writer.append_data(frame)

    writer.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
