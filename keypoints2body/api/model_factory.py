from __future__ import annotations

from pathlib import Path

import smplx
import torch

from ..core.config import BodyModelConfig


def load_body_model(
    config: BodyModelConfig,
    device: torch.device,
):
    """Load a body model module with ``smplx.create``.

    Args:
        config: Model-loading configuration.
        device: Torch device where the model should live.

    Returns:
        A body model module returned by ``smplx.create`` moved to ``device``.
    """
    model_dir = Path(config.model_dir).expanduser()
    return smplx.create(
        str(model_dir),
        model_type=config.model_type,
        gender=config.gender,
        ext=config.ext,
        batch_size=config.batch_size,
    ).to(device)
