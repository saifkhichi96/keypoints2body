from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .. import optimize_params_sequence


def main() -> None:
    """CLI entrypoint for sequence fitting."""
    parser = argparse.ArgumentParser(description="Optimize SMPL params for a sequence.")
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Input .npy seq joints, shape (T,K,3|4)",
    )
    parser.add_argument("--layout", type=str, default=None)
    args = parser.parse_args()

    joints = np.load(args.file)
    results = optimize_params_sequence(joints, joint_layout=args.layout)
    print(f"optimized {len(results)} frames")


if __name__ == "__main__":
    main()
