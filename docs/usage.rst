Library Usage
=============

Public API
----------

The package exports three main functions:

- ``optimize_params_frame``
- ``optimize_params_sequence``
- ``optimize_shape_sequence``

Body model selection
--------------------

Recognized model backends:

- ``body_model="smpl"``
- ``body_model="smplh"``
- ``body_model="smplx"``
- ``body_model="mano"``
- ``body_model="flame"``

Current optimization estimator support:

- Fully supported in optimization APIs: ``smpl``, ``smplh``, ``smplx``
- Registered for model loading and type system: ``mano``, ``flame``

Returned ``result.params`` type depends on model:

- ``SMPLData`` for SMPL
- ``SMPLHData`` for SMPL-H
- ``SMPLXData`` for SMPL-X
- ``MANOData`` for MANO (when estimator integration is enabled)
- ``FLAMEData`` for FLAME (when estimator integration is enabled)

Frame Optimization
------------------

.. code-block:: python

   import numpy as np
   from keypoints2body import optimize_params_frame

   joints = np.random.randn(22, 3).astype("float32")
   result = optimize_params_frame(
       joints,
       body_model="smpl",
       joint_layout="AMASS",
   )

Supported frame inputs:

- ``(K, 3)`` xyz
- ``(K, 4)`` xyz + confidence score

Sequence Optimization
---------------------

.. code-block:: python

   import numpy as np
   from keypoints2body import optimize_params_sequence

   joints_seq = np.random.randn(120, 22, 3).astype("float32")
   results = optimize_params_sequence(
       joints_seq,
       body_model="smplx",
       joint_layout="AMASS",
   )

Supported sequence inputs:

- ``(T, K, 3)``
- ``(T, K, 4)``

Joint layout adapters
---------------------

The library supports these layouts via adapter registry:

- ``AMASS`` (22 joints)
- ``SMPL24`` (24 joints)
- ``Manny25`` (25 joints)
- ``Halpe26`` (26 joints)
- ``SpineTrack37`` (37 joints)

Confidence handling
-------------------

- If the last channel exists (shape ``...,4``), channel 4 is used as per-joint confidence.
- Otherwise confidence defaults to ``1.0`` for all joints.

Configuration
-------------

Use ``FrameOptimizeConfig`` and ``SequenceOptimizeConfig`` for explicit control.

.. code-block:: python

   from keypoints2body.core.config import FrameOptimizeConfig, SequenceOptimizeConfig

   frame_cfg = FrameOptimizeConfig(
       coordinate_mode="world",
       num_iters_first=60,
       num_iters_followup=20,
       joints_category="AMASS",
   )
   seq_cfg = SequenceOptimizeConfig(
       frame=frame_cfg,
       num_shape_iters=30,
       use_shape_optimization=True,
       fix_foot=True,
   )

   results = optimize_params_sequence(
       joints_seq,
       config=seq_cfg,
       body_model="smpl",
       joint_layout="AMASS",
   )
