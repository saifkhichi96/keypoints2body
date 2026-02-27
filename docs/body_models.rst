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

Parameter Shapes
----------------

Shape notation:

- ``B``: batch size (typically 1 in frame fitting).
- ``Nb``: number of shape coefficients (``model.num_betas``, often 10).
- ``Ne``: number of expression coefficients (``model.num_expression_coeffs``, often 10).

Common fields (``BodyModelParams``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Field
     - Shape
     - Notes
   * - ``betas``
     - ``(B, Nb)``
     - Shape coefficients.
   * - ``global_orient``
     - ``(B, 3)``
     - Root orientation in axis-angle.
   * - ``body_pose``
     - ``(B, D_body)``
     - Model-dependent body block. For MANO/FLAME this is currently ``(B, 0)`` in this codebase.
   * - ``transl``
     - ``(B, 3)`` or ``None``
     - Translation (world/camera depending on coordinate mode).
   * - ``metadata``
     - dict
     - Optional trace/debug metadata.

Model-specific fields
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 14 20 20 46

   * - Model
     - Dataclass
     - Extra fields
     - Extra field shapes
   * - ``smpl``
     - ``SMPLData``
     - None
     - N/A
   * - ``smplh``
     - ``SMPLHData``
     - ``left_hand_pose``, ``right_hand_pose``
     - Each ``(B, 45)`` (15 joints * 3 axis-angle)
   * - ``smplx``
     - ``SMPLXData``
     - ``left_hand_pose``, ``right_hand_pose``, ``expression``, ``jaw_pose``, ``leye_pose``, ``reye_pose``
     - Hand poses: ``(B, 45)`` each; ``expression``: ``(B, Ne)``; jaw/eyes: ``(B, 3)`` each
   * - ``mano``
     - ``MANOData``
     - ``hand_pose``
     - ``(B, 45)`` (15 joints * 3 axis-angle)
   * - ``flame``
     - ``FLAMEData``
     - ``expression``, ``jaw_pose``, ``neck_pose``, ``leye_pose``, ``reye_pose``
     - ``expression``: ``(B, Ne)``; jaw/neck/eyes: ``(B, 3)`` each

Fitting input expectations (important)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 14 28 58

   * - Model
     - Preferred observation input
     - Notes
   * - ``smpl``
     - Flat array/tensor ``(K,3|4)`` or ``(T,K,3|4)``
     - ``joint_layout`` adapters (AMASS/SMPL24/etc.) are supported.
   * - ``smplh``
     - Dict blocks: ``body`` + hands
     - For full hand-constrained fitting, provide ``left_hand`` and ``right_hand`` blocks.
   * - ``smplx``
     - Dict blocks: ``body`` + hands + face
     - For full-body/hand/face constraints, use block-wise dict and ``joint_layout=None``.
   * - ``mano``
     - Flat hand keypoints or dict-style hand blocks
     - Keep model-native order and ``joint_layout=None``.
   * - ``flame``
     - Flat face keypoints or dict-style face blocks
     - Keep model-native order and ``joint_layout=None``.

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
