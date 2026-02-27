Contributing
============

This page describes architectural conventions and roadmap guidance for
`keypoints2body` contributors.

Scope direction
---------------

`keypoints2body` is intended to support:

- Multiple body model families (SMPL family first, additional backends later)
- Multiple input modalities
  - 3D joints (current)
  - 2D joints (planned)
  - multi-view 2D joints (planned)
- Multiple estimation paradigms
  - optimization-based fitters (current)
  - learned predictors (planned)

Current architecture contracts
------------------------------

- ``keypoints2body.api`` is the stable public surface.
- ``keypoints2body.core`` owns estimation orchestration.
- ``keypoints2body.models`` owns data contracts.
- ``keypoints2body.io`` owns file I/O and exports.
- ``keypoints2body.cli`` should call ``api`` only.

Extension points
----------------

1. New body model families
~~~~~~~~~~~~~~~~~~~~~~~~~~

Target modules:

- ``keypoints2body.api.model_factory``
- ``keypoints2body.models``
- potentially ``keypoints2body.core.backends`` (future)

Recommended path:

1. Add model family identifier/config.
2. Implement loader and parameter conventions.
3. Extend parameter dataclasses when needed.
4. Add loading and smoke-fit tests.

2. Learned estimators
~~~~~~~~~~~~~~~~~~~~~

Target modules:

- ``keypoints2body.core.estimators``

Current state:

- ``FrameOptimizeConfig.estimator_type`` supports ``optimization`` and ``learned``.
- ``learned`` currently raises ``NotImplementedError`` by design.

Recommended path:

1. Implement learned estimator class.
2. Wire it in ``core.estimators.factory``.
3. Keep ``BodyModelFitResult`` output contract.
4. Add deterministic inference tests.

3. New input modalities (2D/multiview)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Current state:

- ``FrameOptimizeConfig.input_type`` exists.
- Current APIs support ``joints3d`` only and fail fast otherwise.

Recommended path:

1. Add modality-specific adapters under ``core.joints``.
2. Add modality-specific constraints/losses.
3. Keep top-level API signatures stable.
4. Add unit tests for each accepted shape contract.

Roadmap
-------

v0.1.x
~~~~~~

- Stabilize package naming and docs.
- Keep optimization-based 3D fitting robust.
- Expand tests for adapters/config/engine behavior.

v0.2
~~~~

- Add backend registry to decouple model-family specifics.
- Add first non-SMPL-family backend integration.
- Add backend capability matrix in docs.

v0.3
~~~~

- Implement first learned predictor pipeline.
- Keep unified output contract with optimization estimators.
- Add mixed strategy (predict-init + optimize-refine).

v0.4
~~~~

- Add ``joints2d`` and ``multiview_joints2d`` input paths.
- Add camera/projective constraints and multi-view fusion.
- Add modality-aware examples and docs.

Contributor checklist
---------------------

Before merging architecture-affecting changes:

- Preserve layering boundaries.
- Keep API return contracts stable.
- Update architecture docs when flows/modules change.
- Add tests for new adapter/fitter/config/backend paths.
- Keep CLI wrappers routed through package APIs.
