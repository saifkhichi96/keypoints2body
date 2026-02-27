Architecture
============

This page is written for contributors extending internals, adding new model types,
or changing optimization behavior.

Design goals
------------

- Keep a stable, small public API surface in ``keypoints2body.api``.
- Isolate optimization mechanics under ``keypoints2body.core``.
- Keep data contracts explicit and typed via ``keypoints2body.models``.
- Keep I/O and file-format concerns out of optimization internals.

Package topology
----------------

.. mermaid::

   graph TD
     A[keypoints2body.api] --> B[keypoints2body.core.engine]
     A --> C[keypoints2body.api.model_factory]
     B --> E[keypoints2body.core.estimators.factory]
     E --> D[keypoints2body.core.estimators.optimization]
     D --> F[keypoints2body.core.fitters.camera_space]
     D --> G[keypoints2body.core.fitters.world_space]
     B --> H[keypoints2body.core.shape]
     B --> I[keypoints2body.core.joints.adapters]
     F --> J[keypoints2body.core.losses]
     G --> J
     F --> K[keypoints2body.core.prior]
     G --> K
     A --> L[keypoints2body.models.smpl_data]
     M[keypoints2body.io.motion] --> I
     N[keypoints2body.cli.*] --> A

Layering contract
-----------------

- ``api/*`` may depend on ``core/*``, ``models/*``, ``io/*``.
- ``core/*`` may depend on ``models/*`` and 3rd-party compute libs.
- ``models/*`` should stay dependency-light and free of optimizer logic.
- ``cli/*`` should remain thin wrappers over ``api/*``.
- Avoid importing ``cli``/``io`` from ``core``.

If you need a new dependency path, document why in this page and keep it one-way.

Core runtime flow
-----------------

Single frame flow
~~~~~~~~~~~~~~~~~

.. mermaid::

   sequenceDiagram
     participant U as User Code / CLI
     participant API as api.frame.optimize_params_frame
     participant AD as core.joints.adapters
     participant MF as api.model_factory
     participant EN as core.engine.OptimizeEngine
     participant ES as core.estimators.*
     participant FT as core.fitters.*

     U->>API: joints + options
     API->>AD: normalize_joints_frame + adapt_layout_and_conf
     API->>MF: load_body_model (if model not provided)
     API->>EN: construct engine(frame config)
     API->>EN: fit_frame(init_params, j3d, conf)
     EN->>ES: dispatch by estimator_type
     ES->>FT: (optimization) dispatch by coordinate_mode
     FT-->>EN: BodyModelFitResult
     EN-->>API: BodyModelFitResult
     API-->>U: BodyModelFitResult

Sequence flow
~~~~~~~~~~~~~

.. mermaid::

   flowchart TD
     A[optimize_params_sequence] --> B[normalize_joints_sequence]
     B --> C[adapt_layout_and_conf]
     C --> D[load mean pose/shape]
     D --> E{use_shape_optimization?}
     E -- yes --> F[optimize_shape_pass]
     E -- no --> G[reuse mean betas]
     F --> H[create engine]
     G --> H
     H --> I[frame loop]
     I --> J[fit_frame]
     J --> K{use_previous_frame_init?}
     K -- yes --> L[prev = result.params]
     K -- no --> M[keep initial params]
     L --> I
     M --> I
     I --> N[collect BodyModelFitResult list]

Module responsibilities
-----------------------

``keypoints2body.api``
~~~~~~~~~~~~~~~~~~~

- Public functional interfaces and argument normalization.
- Converts user-facing options/config dictionaries into typed configs.
- Enforces high-level input contracts and error messages.

``keypoints2body.core.engine``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Chooses estimator implementation from ``FrameOptimizeConfig.estimator_type``.
- Loads default initialization values (mean pose/shape).
- Orchestrates shape-pass + frame optimization for sequence APIs.

``keypoints2body.core.estimators``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Defines a common estimator interface for current and future methods.
- ``optimization`` estimator adapts existing iterative fitters.
- ``learned`` estimator path is reserved for future direct predictors.

``keypoints2body.core.fitters``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``camera_space`` implements camera-translation based fitting.
- ``world_space`` implements explicit world-translation fitting.
- Both return ``BodyModelFitResult`` and share loss/prior infrastructure.

``keypoints2body.core.joints.adapters``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Normalizes input tensor shapes:
  - frame: ``(K,3|4) -> (1,K,3) + (K,)``
  - seq: ``(T,K,3|4) -> (T,K,3) + (T,K)``
- Converts incoming layouts to canonical targets via registry.
- Applies mapping consistently to both coordinates and confidence.

``keypoints2body.models.smpl_data``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Contains typed parameter containers and fit result container.
- Shared contract between API, core, and downstream consumers.
- ``metadata`` field is the intended extension point for trace/debug context.

Data contracts
--------------

Input joints
~~~~~~~~~~~~

- Frame APIs accept ``np.ndarray``/``torch.Tensor`` with shape ``(K,3)`` or ``(K,4)``.
- Sequence APIs accept shape ``(T,K,3)`` or ``(T,K,4)``.
- If channel 4 exists, it is interpreted as confidence.

Result contract
~~~~~~~~~~~~~~~

``BodyModelFitResult`` always includes:

- ``params``: model parameter dataclass instance
- ``vertices``: torch tensor
- ``joints``: torch tensor
- ``loss``: scalar tensor (or ``None`` for some optimizer branches)

Extension guide
---------------

Add a new joint layout
~~~~~~~~~~~~~~~~~~~~~~

1. Add entry in ``core.joints.adapters.ADAPTERS``.
2. Provide mapping/coordinate transform rules.
3. Ensure both points and confidence are transformed identically.
4. Add adapter tests.

Add a new coordinate mode/fitter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Add new fitter module under ``core/fitters`` with ``fit_frame`` signature.
2. Update ``OptimizeEngine`` dispatch in ``core.engine``.
3. Add config flag(s) in ``FrameOptimizeConfig``.
4. Add sequence + frame smoke tests.

Add a new body-model family variant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Extend model dataclasses if new params are required.
2. Update ``api.model_factory`` loading logic.
3. Register capability expectations (loading-only vs optimization-ready).
4. Ensure return typing maps correctly from ``body_model`` input.
5. Add docs/examples and API tests.

Contributor checklist
---------------------

Before opening a PR touching core architecture:

- Preserve layering boundaries described above.
- Update this page if flows/modules changed.
- Add tests for any new adapter/fitter/config path.
- Verify CLI wrappers still call package APIs (not core internals directly).
