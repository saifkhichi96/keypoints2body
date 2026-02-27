Getting Started
===============

Requirements
------------

- Python 3.10+
- PyTorch
- ``smplx`` package
- Body model files and priors under ``./data/models``

Supported body models
---------------------

The public APIs accept ``body_model`` with one of:

- ``"smpl"``
- ``"smplh"``
- ``"smplx"``

Required model assets
---------------------

At runtime, the package expects these assets under ``./data/models`` by default:

- Body model files used by ``smplx.create``
- Pose prior GMM file(s), e.g. ``gmm_08.pkl``
- Mean pose/shape file, e.g. ``neutral_smpl_mean_params.h5``

A practical layout is:

.. code-block:: text

   data/models/
   ├── gmm_08.pkl
   ├── neutral_smpl_mean_params.h5
   ├── SMPL_NEUTRAL.pkl
   ├── SMPLH_NEUTRAL.pkl
   └── SMPLX_NEUTRAL.npz

If your assets are elsewhere, pass explicit configuration (e.g. ``BodyModelConfig``)
so model loading points to your actual paths/extensions.

Install
-------

Editable install from repo root:

.. code-block:: bash

   pip install -e .

Optional extras:

.. code-block:: bash

   pip install -e .[dev]
   pip install -e .[docs]

Minimal Example
---------------

.. code-block:: python

   import numpy as np
   from keypoints2body import optimize_params_frame

   joints = np.zeros((22, 3), dtype=np.float32)
   result = optimize_params_frame(joints, body_model="smpl", joint_layout="AMASS")
   print(result.params.pose.shape)

Warm-start next frame
---------------------

.. code-block:: python

   result1 = optimize_params_frame(joints_frame1, joint_layout="AMASS")
   result2 = optimize_params_frame(
       joints_frame2,
       joint_layout="AMASS",
       prev_params=result1.params,
   )
