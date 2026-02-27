from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .. import optimize_params_frame


def main() -> None:
    """CLI entrypoint for one-frame fitting."""
    parser = argparse.ArgumentParser(description="Optimize SMPL params for one frame.")
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Input .npy frame joints, shape (K,3|4)",
    )
    parser.add_argument("--layout", type=str, default=None)
    args = parser.parse_args()

    joints = np.load(args.file)
    if joints.ndim == 3:
        joints = joints[0]
    result = optimize_params_frame(joints, joint_layout=args.layout)
    print(result.params)


if __name__ == "__main__":
    main()
