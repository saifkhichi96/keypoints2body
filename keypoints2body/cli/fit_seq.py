from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .. import optimize_params_sequence


def main() -> None:
    """CLI entrypoint for sequence fitting."""
    parser = argparse.ArgumentParser(
        description="Optimize body-model parameters for a sequence."
    )
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Input .npy seq joints, shape (T,K,3|4)",
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
    results = optimize_params_sequence(
        joints,
        body_model=args.body_model,
        joint_layout=args.layout,
    )
    print(f"optimized {len(results)} frames")


if __name__ == "__main__":
    main()
