"""Fellegi-Sunter scoring engine (native NumPy, no SQL engine).

Two stages consume the comparison set:

* *train* - estimate the comparison-level ``m`` (match) / ``u`` (non-match)
  probabilities and the base prior.  Two estimators are provided:

  - :meth:`FellegiSunterScorer.calibrate_from_pairs` - supervised calibration
    from labelled match/non-match pairs;
  - :meth:`FellegiSunterScorer.fit_em` - unsupervised expectation maximisation
    on a (near-duplicate-bearing) population, mirroring the standard
    Fellegi-Sunter parameter-estimation workflow (u estimated by
    random sampling of pairs, m and the prior fit by EM over blocking-rule
    candidate pairs).

* *infer* - score ``(query, candidate)`` pairs in one pass.

Inference is entirely vectorized: every comparison's levels are evaluated as
NumPy predicates over whole batches of pairs (see :mod:`vectorer.sim`), each
pair is assigned its highest-priority level, and the posterior is the sigmoid
of ``log(prior odds) + sum(log(m/u))`` -- the same match-weight algebra
expressed without SQL.  "Vectoring" here is over the *batch* (all
of a query's candidates, or every canopy pair) rather than row-by-row.

By default the match function is *reflexive* (:attr:`idempotent`): a pair whose
compared fields are content-identical scores exactly ``1.0``.  This guarantees
the ``r ~ r`` property of the Swoosh Union-Class ICAR construction, which would
otherwise fail for "thin" records (few non-null comparison fields) whose
self-posterior -- all null levels carrying no evidence -- sits at the prior.
Pass ``idempotent=False`` to recover the raw calibrated posterior for
identical-content pairs.
"""

from ._capture_recapture import (
    CaptureRecapturePrior,
    estimate_prior_capture_recapture,
)
from ._constants import DEFAULT_PRIOR, DEFAULT_THRESHOLD
from ._scorer import FellegiSunterScorer
from ._splink import import_splink_scorer
from ._weights import WeightTable

__all__ = [
    "CaptureRecapturePrior",
    "DEFAULT_PRIOR",
    "DEFAULT_THRESHOLD",
    "FellegiSunterScorer",
    "WeightTable",
    "estimate_prior_capture_recapture",
    "import_splink_scorer",
]