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

- Fully supported in optimization APIs: ``smpl``, ``smplh``, ``smplx``, ``mano``, ``flame``

Returned ``result.params`` type depends on model:

- ``SMPLData`` for SMPL
- ``SMPLHData`` for SMPL-H
- ``SMPLXData`` for SMPL-X
- ``MANOData`` for MANO
- ``FLAMEData`` for FLAME

For parameter details, see :doc:`body_models`.

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
- dict blocks for richer models:
  - ``{"body": (K,3|4), "left_hand": (21,3|4), "right_hand": (21,3|4), "face": (F,3|4)}``

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
- dict blocks:
  - ``{"body": (T,K,3|4), "left_hand": (T,21,3|4), "right_hand": (T,21,3|4), "face": (T,F,3|4)}``

Model-Specific Examples
-----------------------

SMPL (body-only, AMASS layout)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from keypoints2body import optimize_params_frame

   body22 = np.random.randn(22, 3).astype("float32")
   result = optimize_params_frame(
       body22,
       body_model="smpl",
       joint_layout="AMASS",
   )

SMPLH (body + hands via dict blocks)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from keypoints2body import optimize_params_frame

   obs = {
       "body": np.random.randn(22, 3).astype("float32"),
       "left_hand": np.random.randn(21, 3).astype("float32"),
       "right_hand": np.random.randn(21, 3).astype("float32"),
   }
   result = optimize_params_frame(
       obs,
       body_model="smplh",
       joint_layout=None,
   )

SMPLX (body + hands + face via dict blocks)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from keypoints2body import optimize_params_frame

   obs = {
       "body": np.random.randn(22, 3).astype("float32"),
       "left_hand": np.random.randn(21, 3).astype("float32"),
       "right_hand": np.random.randn(21, 3).astype("float32"),
       "face": np.random.randn(68, 3).astype("float32"),
   }
   result = optimize_params_frame(
       obs,
       body_model="smplx",
       joint_layout=None,
   )

MANO (hand-only)
~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from keypoints2body import optimize_params_frame

   hand21 = np.random.randn(21, 3).astype("float32")
   result = optimize_params_frame(
       hand21,
       body_model="mano",
       joint_layout=None,
   )

FLAME (face-only)
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from keypoints2body import optimize_params_frame

   face68 = np.random.randn(68, 3).astype("float32")
   result = optimize_params_frame(
       face68,
       body_model="flame",
       joint_layout=None,
   )

Joint layout adapters
---------------------

For SMPL-family body fitting, the library supports these layouts via adapter registry:

- ``AMASS`` (22 joints)
- ``SMPL24`` (24 joints)
- ``Manny25`` (25 joints)
- ``Halpe26`` (26 joints)
- ``SpineTrack37`` (37 joints)

For MANO/FLAME, provide joints in model-native order and keep ``joint_layout=None``.
For SMPLH/SMPLX full-body+hands+face fitting, prefer dict block input with ``joint_layout=None``.

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
