Body Models
===========

This page summarizes the body-model backends available in ``keypoints2body``
and the parameter dataclasses returned by the optimization APIs.

Model overview
--------------

- ``smpl``: full-body model with body pose + shape.
- ``smplh``: SMPL body + articulated hands.
- ``smplx``: SMPLH + face/expression components.
- ``mano``: hand-only model.
- ``flame``: head/face model.

In code, these are selected with the ``body_model`` argument in public APIs.

Observation formats
-------------------

Two input styles are supported:

- Flat array/tensor keypoints (for body-centric layouts such as ``AMASS``/``SMPL24``).
- Block-wise dict observations for richer constraints:
  - ``body``
  - ``left_hand``
  - ``right_hand``
  - ``face``

For full SMPLH/SMPLX fitting, use the block-wise dict style so hand/face terms are
directly constrained in the optimization loss.

Common parameter fields
-----------------------

All model dataclasses inherit from ``BodyModelParams`` and include:

- ``betas``: shape coefficients.
- ``global_orient``: global/root orientation (axis-angle).
- ``body_pose``: body pose block.
- ``transl``: optional translation.
- ``metadata``: optional dictionary for trace/debug metadata.

SMPL family
-----------

SMPLData
~~~~~~~~

Class: ``SMPLData``

- Uses common fields only:
  - ``betas``
  - ``global_orient``
  - ``body_pose``
  - ``transl`` (optional)

SMPLHData
~~~~~~~~~

Class: ``SMPLHData`` (extends ``SMPLData``)

- Adds:
  - ``left_hand_pose``
  - ``right_hand_pose``

SMPLXData
~~~~~~~~~

Class: ``SMPLXData`` (extends ``SMPLHData``)

- Adds:
  - ``expression``
  - ``jaw_pose``
  - ``leye_pose``
  - ``reye_pose``

MANO
----

Class: ``MANOData``

- Uses common base fields.
- Adds:
  - ``hand_pose``

Notes:

- MANO fitting expects hand keypoints in model-native order.
- Keep ``joint_layout=None`` when calling API functions with ``body_model="mano"``.

FLAME
-----

Class: ``FLAMEData``

- Uses common base fields.
- Adds:
  - ``expression``
  - ``jaw_pose``
  - ``neck_pose``
  - ``leye_pose``
  - ``reye_pose``

Notes:

- FLAME fitting expects face keypoints/landmarks in model-native order.
- Keep ``joint_layout=None`` when calling API functions with ``body_model="flame"``.

Returned fit result
-------------------

All fitting APIs return ``BodyModelFitResult``:

- ``params``: one of ``SMPLData``, ``SMPLHData``, ``SMPLXData``, ``MANOData``, ``FLAMEData``.
- ``vertices``: model vertices tensor.
- ``joints``: model joints tensor.
- ``loss``: final loss scalar (if available).

Example
-------

.. code-block:: python

   import numpy as np
   from keypoints2body import optimize_params_frame

   joints = np.zeros((21, 3), dtype=np.float32)
   result = optimize_params_frame(joints, body_model="mano", joint_layout=None)
   print(type(result.params).__name__)  # MANOData
