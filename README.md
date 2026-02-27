# keypoints2body

[![PyPI version](https://img.shields.io/pypi/v/keypoints2body.svg)](https://pypi.org/project/keypoints2body/)
[![Python versions](https://img.shields.io/pypi/pyversions/keypoints2body.svg)](https://pypi.org/project/keypoints2body/)
[![Documentation Status](https://readthedocs.org/projects/keypoints2body/badge/?version=latest)](https://keypoints2body.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![keypoints2body header](docs/assets/header.svg)

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
