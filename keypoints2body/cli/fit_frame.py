from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .. import optimize_params_frame


def main() -> None:
    """CLI entrypoint for one-frame fitting."""
    parser = argparse.ArgumentParser(
        description="Optimize body-model parameters for one frame."
    )
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Input .npy frame joints, shape (K,3|4)",
    )
    parser.add_argument("--layout", type=str, default=None)
    parser.add_argument(
        "--body-model",
        type=str,
        default="smpl",
        choices=["smpl", "smplh", "smplx", "mano", "flame"],
        help="Body model backend to use.",
    )
    args = parser.parse_args()

    joints = np.load(args.file)
    if joints.ndim == 3:
        joints = joints[0]
    result = optimize_params_frame(
        joints,
        body_model=args.body_model,
        joint_layout=args.layout,
    )
    print(result.params)


if __name__ == "__main__":
    main()
