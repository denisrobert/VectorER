"""Extensible Fellegi-Sunter comparison set -- native, fully vectorized.

The comparison set spans the standard attribute-comparison families of record
linkage, implemented **natively in NumPy with no SQL engine**: each comparison
is a list of *levels*, and every level is a vectorized NumPy predicate
evaluated over whole batches of pairs (see :mod:`vectorer.sim`).  The set is
name-keyed through :class:`ComparisonRegistry` and currently covers 19
options.

The module is a small package: shared building blocks live in ``_core.py``
(``PairValues``, ``Level`` / ``ComparisonSpec``, the default ``m/u`` scheme and
the level/prescore toolkit); the comparison families are grouped by shared
machinery in ``_distance_families.py``, ``_temporal_families.py``,
``_structural_families.py`` and ``_custom.py``; the registry and the
declared-comparison helpers live in ``_registry.py`` / ``_comparison.py``.

Performance
-----------
Each comparison carries an optional *pre-score* step (:attr:`ComparisonSpec.prescore`)
that computes shared score arrays (e.g. one Jaro-Winkler pass instead of one per
threshold) once per batch; every level then reads those cached arrays.  Levels
are ordered by decreasing agreement, and the first matching level decides the
pair's bayes factor -- the **last level must be the ELSE fallback**
(``test=None``), holding all pairs that match no earlier level (contract
detailed on :class:`ComparisonSpec`).
"""

from ._core import (
    Cache,
    ComparisonSpec,
    Level,
    PairValues,
    apply_default_mu,
    build_spec,
    _and_test,
    _cached_epoch_diff_test,
    _cached_threshold_test,
    _distance_prescore,
    _else_level,
    _exact_test,
    _isnan,
    _jw_prescore,
    _null_test,
)
from ._registry import (
    REGISTRY,
    ComparisonRegistry,
    RegistryEntry,
    available_comparisons,
    comparison_catalog,
    register_comparison,
)
from ._comparison import (
    Comparison,
    comparison_fields,
    comparison_from_dict,
    comparison_set,
    comparison_to_dict,
    make_comparison,
    make_comparisons,
)
from ._custom import time_decay_wrapper, time_decayed_comparison_builder
from ._distance_families import exact_match_spec

__all__ = [
    "Comparison",
    "ComparisonRegistry",
    "ComparisonSpec",
    "Level",
    "PairValues",
    "REGISTRY",
    "RegistryEntry",
    "available_comparisons",
    "comparison_catalog",
    "comparison_fields",
    "comparison_from_dict",
    "comparison_set",
    "comparison_to_dict",
    "make_comparison",
    "make_comparisons",
    "register_comparison",
    "time_decay_wrapper",
    "time_decayed_comparison_builder",
]