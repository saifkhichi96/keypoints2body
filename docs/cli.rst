CLI Reference
=============

The package provides three console scripts:

- ``keypoints2body-fit-frame``
- ``keypoints2body-fit-seq``
- ``keypoints2body-eval``

Fit One Frame
-------------

.. code-block:: bash

   keypoints2body-fit-frame --file ./frame.npy --layout AMASS

Fit Sequence
------------

.. code-block:: bash

   keypoints2body-fit-seq --file ./sequence.npy --layout AMASS

Evaluate on AMASS
-----------------

.. code-block:: bash

   keypoints2body-eval \
     --amass-root /path/to/amass_npz \
     --num-shape-iters 40 \
     --num-body-iters-first 100 \
     --num-body-iters 50
