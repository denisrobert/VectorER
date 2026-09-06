vector-er
=========

.. image:: _static/logo.svg
   :alt: vector-er logo (the glyph ER with a vector macron)
   :width: 120px
   :align: center

:mod:`vectorer` is a framework for **embedding-and-vector-based entity
resolution**: incremental (streaming) resolution, batch deduplication, and
two-database record linkage, built on a vectorized Fellegi-Sunter scoring
engine with no SQL dependency.

This site is the **API reference**.  For tutorials, the design rationale, and
practical recipes, see the docs in the
`repository <https://github.com/denisrobert/VectorER>`_ (``README.md``,
``.docs/``).

.. note::
   Use the **version selector** in the top bar to switch between releases —
   `latest`, `stable`, or any published ``v0.x.y``.  The version being viewed
   is also shown in the page header/footer.  Curated release notes live in the
   repository's ``CHANGELOG.md``.

Installation
------------

.. code-block:: bash

   pip install vectorer          # core
   pip install "vectorer[embedding]"   # sentence-transformers model

Modes
-----

The framework provides three operation modes built on shared search, scoring
and clustering machinery:

* **Incremental** (:mod:`vectorer.incremental`) — resolve records one at a time
  against a reference store.
* **Batch** (:mod:`vectorer.batch`) — deduplicate / cluster a whole dataset.
* **Link** (:mod:`vectorer.link`) — link records across two differently-schemed
  databases (mergers / cross-enterprise collaboration).

API reference
-------------

.. toctree::
   :maxdepth: 2

   api

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`