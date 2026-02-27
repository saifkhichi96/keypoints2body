from importlib.metadata import PackageNotFoundError, version

from .api.frame import optimize_params_frame
from .api.sequence import optimize_params_sequence, optimize_shape_sequence
from .models.smpl_data import (
    BodyModelFitResult,
    BodyModelParams,
    FLAMEData,
    MANOData,
    SMPLData,
    SMPLHData,
    SMPLXData,
)

try:
    __version__ = version("keypoints2body")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "optimize_params_frame",
    "optimize_params_sequence",
    "optimize_shape_sequence",
    "BodyModelFitResult",
    "BodyModelParams",
    "MANOData",
    "FLAMEData",
    "SMPLData",
    "SMPLHData",
    "SMPLXData",
]
