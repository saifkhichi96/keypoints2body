![keypoints2body header](docs/assets/header.svg)

# keypoints2body

`keypoints2body` is a Python library for optimizing SMPL-family body model parameters
from 3D joints for both single frames and motion sequences.

## Install

```bash
pip install -e .
```

## Quick usage

```python
import numpy as np
from keypoints2body import optimize_params_frame

joints = np.zeros((22, 3), dtype=np.float32)
result = optimize_params_frame(joints, joint_layout="AMASS")
```

## Documentation

Full documentation lives under [`docs/`](docs/) and is intended for Sphinx.

Suggested starting points:

- [Getting started](docs/getting_started.rst)
- [Library usage](docs/usage.rst)
- [CLI reference](docs/cli.rst)
- [Architecture](docs/architecture.rst)
- [API reference](docs/api.rst)
- [Contributor guide](docs/contributing.rst)

## CLI

```bash
keypoints2body-fit-frame --help
keypoints2body-fit-seq --help
keypoints2body-eval --help
```

Project script:

```bash
python smpl_fit.py --help
```
