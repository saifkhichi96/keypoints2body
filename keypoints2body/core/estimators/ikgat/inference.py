from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ....models.smpl_data import BodyModelFitResult, BodyModelParams
from .gan_regressor import GATRotationRegressor
from .utils import quat_from_6d, quat_to_6d

logger = logging.getLogger(__name__)


class InputFormat(Enum):
    POS = 3
    POS_ROT6 = 9


class OutputFormat(Enum):
    ROT6 = 6


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def axes_to_rot6(a1: torch.Tensor, a2_raw: torch.Tensor) -> torch.Tensor:
    """Convert two predicted axes into orthonormal 6D rotation representation."""
    a1 = F.normalize(a1, dim=-1)
    a2 = F.normalize(a2_raw - (a1 * a2_raw).sum(dim=-1, keepdim=True) * a1, dim=-1)
    return torch.cat([a1, a2], dim=-1)


@dataclass
class IKGATConfig:
    model_dir: Path = Path("./data/estimators")
    model_format: str = "manny"
    model_type: str = "pos-rot6_to_rot6"
    parent_ids: Optional[list[int]] = None
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4


class RotationRegressionWrapper(nn.Module):
    """Loads a trained IK-GAT model and exposes numpy in/out inference."""

    def __init__(self, model_path: Path, parent_ids: list[int], **kwargs):
        super().__init__()
        if not model_path.exists():
            raise FileNotFoundError(f"IKGAT model file not found: {model_path}")

        self.device = get_device()
        self.input_format = kwargs.pop("input_format", InputFormat.POS_ROT6)
        self.output_format = kwargs.pop("output_format", OutputFormat.ROT6)

        self.model = GATRotationRegressor(
            parent_ids,
            input_dim=self.input_format.value,
            output_dim=self.output_format.value,
            **kwargs,
        )

        state_dict = torch.load(str(model_path), map_location=self.device)
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(
        self,
        positions: np.ndarray,
        quaternions: Optional[np.ndarray],
    ) -> torch.Tensor:
        root_position = positions[0:1, :]
        pos_relative = positions - root_position
        pos_relative = torch.from_numpy(pos_relative.astype(np.float32))

        if self.input_format == InputFormat.POS:
            x = pos_relative
        elif self.input_format == InputFormat.POS_ROT6:
            if quaternions is None:
                raise ValueError(
                    "quaternions are required for model_type='pos-rot6_to_rot6'"
                )
            rot6 = quat_to_6d(torch.from_numpy(quaternions.astype(np.float32)))
            x = torch.cat((pos_relative, rot6), dim=-1)
        else:
            raise NotImplementedError(f"Unsupported input format: {self.input_format}")

        return x.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def forward(
        self,
        positions: np.ndarray,
        quaternions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        inputs = self.preprocess(positions, quaternions)
        pred = self.model(inputs)

        if self.output_format != OutputFormat.ROT6:
            raise NotImplementedError(f"Unsupported output format: {self.output_format}")

        a1_pred = pred[:, :, :3]
        a2_pred = pred[:, :, 3:]
        pred_6d = axes_to_rot6(a1_pred, a2_pred)
        return quat_from_6d(pred_6d[0].cpu().numpy())


def _detect_config(config_path: Path) -> dict:
    default_config = {"hidden_dim": 128, "num_layers": 3, "num_heads": 4}
    if not config_path.exists():
        logger.info("No config.json found at %s; using defaults", config_path)
        return default_config

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        if not isinstance(config, dict):
            return default_config
        for key in ("hidden_dim", "num_layers", "num_heads"):
            if key not in config:
                return default_config
        return config
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to load IKGAT config at %s: %s", config_path, exc)
        return default_config


def create_rotation_regressor(cfg: IKGATConfig) -> RotationRegressionWrapper:
    supported_formats = {"manny", "smplx"}
    if cfg.model_format not in supported_formats:
        raise ValueError(
            f"Unsupported model_format: {cfg.model_format}. Supported: {sorted(supported_formats)}"
        )

    supported_types = {
        "pos_to_rot6": (InputFormat.POS, OutputFormat.ROT6),
        "pos-rot6_to_rot6": (InputFormat.POS_ROT6, OutputFormat.ROT6),
    }
    if cfg.model_type not in supported_types:
        raise ValueError(
            f"Unsupported model_type: {cfg.model_type}. Supported: {sorted(supported_types)}"
        )
    if not cfg.parent_ids:
        raise ValueError("ikgat_parent_ids must be provided for estimator_type='ikgat'")

    model_path = cfg.model_dir / "ikgat" / cfg.model_type / f"{cfg.model_format}.pth"
    model_cfg = _detect_config(model_path.parent / "config.json")
    model_cfg.update(
        {
            "input_format": supported_types[cfg.model_type][0],
            "output_format": supported_types[cfg.model_type][1],
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads,
        }
    )
    return RotationRegressionWrapper(model_path=model_path, parent_ids=cfg.parent_ids, **model_cfg)


class IKGATEstimator:
    """Learned IK estimator that predicts per-joint quaternions from 3D joints."""

    def __init__(self, frame_config, device: torch.device):
        self.device = device
        cfg = IKGATConfig(
            model_dir=Path(frame_config.ikgat_model_dir),
            model_format=frame_config.ikgat_model_format,
            model_type=frame_config.ikgat_model_type,
            parent_ids=frame_config.ikgat_parent_ids,
            hidden_dim=frame_config.ikgat_hidden_dim,
            num_layers=frame_config.ikgat_num_layers,
            num_heads=frame_config.ikgat_num_heads,
        )
        self.wrapper = create_rotation_regressor(cfg)

    def fit_frame(
        self,
        init_params: BodyModelParams,
        j3d: torch.Tensor,
        conf_3d: Optional[torch.Tensor],
        seq_ind: int,
        target_model_indices: Optional[torch.Tensor] = None,
    ) -> BodyModelFitResult:
        del conf_3d, seq_ind, target_model_indices
        positions = j3d[0].detach().cpu().numpy()

        quaternions = None
        if self.wrapper.input_format == InputFormat.POS_ROT6:
            meta = getattr(init_params, "metadata", {}) or {}
            quaternions = meta.get("ikgat_quaternions")
            if quaternions is None:
                raise ValueError(
                    "ikgat model_type='pos-rot6_to_rot6' requires init_params.metadata['ikgat_quaternions']"
                )

        pred_quat = self.wrapper(positions=positions, quaternions=quaternions)
        meta = dict(getattr(init_params, "metadata", {}))
        meta["ikgat_quaternions"] = pred_quat
        params = init_params.detach()
        params.metadata = meta

        joints = j3d.detach()
        vertices = j3d.detach()
        return BodyModelFitResult(params=params, joints=joints, vertices=vertices, loss=None)
