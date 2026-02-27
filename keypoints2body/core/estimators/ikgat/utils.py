from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    """Normalize quaternion tensor along the last axis."""
    return F.normalize(q, p=2.0, dim=-1)


def quat_to_6d(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion tensor (..., 4) to 6D rotation representation."""
    q = normalize_quat(q)
    x, y, z, w = q.unbind(-1)

    c1_x = 1 - 2 * (y * y + z * z)
    c1_y = 2 * (x * y + z * w)
    c1_z = 2 * (x * z - y * w)

    c2_x = 2 * (x * y - z * w)
    c2_y = 1 - 2 * (x * x + z * z)
    c2_z = 2 * (y * z + x * w)

    return torch.stack([c1_x, c1_y, c1_z, c2_x, c2_y, c2_z], dim=-1)


def rot6_to_quat_torch(rot_6d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert 6D rotation representation (..., 6) to quaternion (..., 4)."""
    b1 = F.normalize(rot_6d[..., :3], dim=-1)
    a2 = rot_6d[..., 3:]
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    r = torch.stack([b1, b2, b3], dim=-1)

    trace = r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]
    qw = torch.sqrt(torch.clamp(1.0 + trace, min=eps)) * 0.5
    qx = (r[..., 2, 1] - r[..., 1, 2]) / (4.0 * qw + eps)
    qy = (r[..., 0, 2] - r[..., 2, 0]) / (4.0 * qw + eps)
    qz = (r[..., 1, 0] - r[..., 0, 1]) / (4.0 * qw + eps)

    q = torch.stack([qx, qy, qz, qw], dim=-1)
    return F.normalize(q, dim=-1)


def quat_from_6d(rot_6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotations (..., 6) to quaternions (..., 4) in numpy."""
    t = torch.from_numpy(rot_6d.astype(np.float32))
    q = rot6_to_quat_torch(t)
    return q.cpu().numpy()
