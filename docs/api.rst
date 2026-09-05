.. _api:

====================
Full API reference
====================

The public API lives in the :mod:`vectorer` package. The autodoc-generated
pages below document every public module, class, and function.

------------------
Package
------------------

The :mod:`vectorer` package is a thin namespace that re-exports the public
symbols from the modules below (``IncrementalPipeline``, ``BatchPipeline``,
``RecordLinker``, ``distributed_batch_er``, the comparison set, the scorer,
etc.).  Each re-export is documented in its owning module below, so the
package-level listing is intentionally empty to avoid duplicating them.

.. automodule:: vectorer
   :no-members:

------------------
Pipelines
------------------

.. automodule:: vectorer.incremental
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: vectorer.batch
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: vectorer.link
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: vectorer.distributed
   :members:
   :undoc-members:
   :show-inheritance:

------------------
Core components
------------------

.. automodule:: vectorer.records
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: vectorer.embeddings
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: vectorer.vectorstores
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: vectorer.blocking
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: vectorer.comparisons
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: vectorer.sim
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: vectorer.scoring
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: vectorer.classification
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: vectorer.clustering
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: vectorer.pins
   :members:
   :undoc-members:
   :show-inheritance: