from pathlib import Path

import numpy as np
import pytest

from keypoints2body import optimize_params_frame


def _has_models() -> bool:
    root = Path("./data/models")
    return root.exists()


@pytest.mark.skipif(
    not _has_models(), reason="SMPL models not present in test environment"
)
def test_optimize_params_frame_smoke():
    joints = np.zeros((22, 3), dtype=np.float32)
    result = optimize_params_frame(joints, joint_layout="AMASS")
    assert result.params is not None
