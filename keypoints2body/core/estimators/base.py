from __future__ import annotations

from typing import Optional, Protocol

import torch

from ...models.smpl_data import BodyModelFitResult, BodyModelParams


class BodyEstimator(Protocol):
    """Common interface for both optimization and learned estimators."""

    def fit_frame(
        self,
        init_params: BodyModelParams,
        j3d: torch.Tensor,
        conf_3d: Optional[torch.Tensor],
        seq_ind: int,
        target_model_indices: Optional[torch.Tensor] = None,
    ) -> BodyModelFitResult: ...
